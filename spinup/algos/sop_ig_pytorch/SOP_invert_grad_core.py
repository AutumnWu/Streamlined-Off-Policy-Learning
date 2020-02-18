import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal
from spinup.algos.sac_pytorch.SOP_core_auto import Mlp, weights_init_

class TanhGaussianPolicyIG(Mlp): # for SOP invert gradient, with no tanh
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0,
            device="cpu",
    ):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )

        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        ## this is the layer that gives log_std, init this layer with small weight and bias
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        ## action limit: for example, humanoid has an action limit of -0.4 to 0.4
        self.action_limit = action_limit
        self.apply(weights_init_)
        self.device = device

    def get_env_action(self, obs_np,
        deterministic=False, 
        fixed_sigma_value=0.3
        ):
        """
        Get an action that can be used to forward one step in the environment
        :param obs_np: observation got from environment, in numpy form
        :param action_limit: for scaling the action from range (-1,1) to, for example, range (-3,3)
        :param deterministic: if true then policy make a deterministic action, instead of sample an action
        :return: action in numpy format, can be directly put into env.step()
        """
        ## convert observations to pytorch tensors first
        ## and then use the forward method
        obs_tensor = torch.Tensor(obs_np).unsqueeze(0).to(self.device)
        action_tensor = self.forward_inverting_gradient(obs_tensor, deterministic=deterministic,
                                     fixed_sigma_value=fixed_sigma_value, need_invert_gradient=False)[0].detach()
        ## convert action into the form that can put into the env and scale it
        action_np = action_tensor.cpu().numpy().reshape(-1)
        return action_np

    def forward_inverting_gradient(
            self,
            obs,
            deterministic=False,
            fixed_sigma_value=0.3,
            need_invert_gradient=True,
    ):
        """
        this is basically forward, but with the inverting gradient method
        """
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        """
        The mean below is the value that we will apply the inverting gradient method to 
        """
        mean = self.last_fc_layer(h)
        last_layer_output_mean = mean

        # fixed sigma
        std = torch.zeros(mean.size()).to(self.device)
        std += fixed_sigma_value

        if need_invert_gradient: # for policy loss
            self.mean1 = mean
            mean = torch.Tensor(mean.data)
            self.mean2 = mean
            self.mean2.requires_grad = True # this has to be true
        else: # for Q loss
            pass

        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            action = normal.rsample()

        # the environment will do a hard clip on each action dimension (not on the entire action)
        return (
            action * self.action_limit, last_layer_output_mean
            )

    def inverting_gradient(self):
        """
        call this method after policy loss' backward()
        :return:
        """
        """
        let mean be the 'p' value in Inverting Gradient paper
        now we have grad_p, we need to compute the ratio
        SUPER IMPORTANT, BECAUSE OF HOW PYTORCH LOSS.BACKWARD, OPTIMIZER WORK
        THE 'grad' STORED ARE ACTUALLY THE NEGATE OF THE GRADIENT
        SO WE MUST TAKE A MINUS HERE IF WE WANT GREATER THAN ZERO, LESS THAN ZERO THING TO MAKE SENSE
        """
        grad = -self.mean2.grad.data # this is actually 256 * 7

        greater_than_zero = (grad > 0).float()
        less_than_zero = 1 - greater_than_zero

        # worry about action limit later. Here max and min are always -1, 1
        p_max = 1
        p_min = -1
        p_range = p_max - p_min

        """
        # this ratio is a multiplier of grad_p, if the ratio is negative, then gradient is inverted
        """
        ratios = greater_than_zero * ((p_max - self.mean2.data) / p_range) + \
                 less_than_zero * ((self.mean2.data - p_min) / p_range)

        """
        mean1 is connected to parameters in the policy network
        we are going to do gradient ascent on mean1 * gradient of Q w.r.t mean1 * ratio
        """
        new_obj = self.mean1 * self.mean2.grad.data * ratios
        # we should not have minus sign in the following line.
        loss = new_obj.sum()
        loss.backward()
        return

"""
following are computed with 1-step
DDDDDDDD
tensor([[-3.0713e-03, -7.4410e-02, -1.2484e-02, -1.1516e-01, -1.2680e-01,
         -3.2257e-01, -2.9342e-03, -7.0171e-05, -5.3233e-02, -1.2562e-01,
         -6.5742e-04, -5.0180e-03, -1.1655e-02, -7.9040e-02, -1.2425e-01,
         -1.9538e-02, -1.7279e-01, -4.8323e-03, -1.5001e-02, -4.6700e-03,
         -1.1387e-01, -1.0115e-03, -1.5606e-01, -3.6727e-02, -2.7707e-03,
         -2.4942e-01, -4.8595e-02, -9.4311e-02, -6.4797e-03, -5.0762e-02,
         -1.0167e-01, -6.8327e-04],
seems to be correct
"""