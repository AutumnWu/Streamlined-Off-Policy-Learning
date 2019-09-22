import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        """
        ## init buffers as numpy arrays
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        data will get stored in the pointer's location
        data should NOT be in tensor format.
        it's easier if you get data from environment
        then just store them with the geiven format
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        ## move the pointer to store in next location in buffer
        self.ptr = (self.ptr+1) % self.max_size
        ## keep track of the current buffer size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        ## sample with replacement from buffer
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        return the log probability of a value
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        # use arctanh formula to compute arctanh(value)
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - \
               torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        Implement: tanh(mu + sigma * eksee)
        with eksee~N(0,1)
        z here is mu+sigma+eksee
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal( ## this part is eksee~N(0,1)
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample()
        )

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        ## here we use ModuleList so that the layers in it can be
        ## detected by .parameters() call
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        ## initialize each hidden layer
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

        ## init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(in_size, output_size)

        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output

class TanhGaussianPolicySACAdapt(Mlp):
    """
    TODO for this class, we might want to eventually combine
    this with the TanhGaussianPolicy class
    but for now let's just use it as a separate class
    this class is only used in the SAC adapt version
    (Though I think it might also help boost performance in our
    old SAC)
    The only difference is that we now do action squeeze and
    compute action log prob using the method described in
    "Enforcing Action Bounds" section of SAC adapt paper
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0
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

    def get_env_action(self, obs_np, 
        deterministic=False, 
        removed_tanh=False, 
        fixed_sigma=False, 
        clip_action=False, 
        b=None, 
        beta_action=False, 
        beta=None,
        SOP=False,
        record_mu=False
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
        obs_tensor = torch.Tensor(obs_np).unsqueeze(0)
        #if removed_tanh:
        #    action_tensor = self.forward(obs_tensor, deterministic=deterministic,
        #                             return_log_prob=False, fixed_sigma=fixed_sigma, removed_tanh=True)[0].detach()
        #else:
        if record_mu:
            action, mean, _, _, _, pre_tanh_value = self.forward(obs_tensor, deterministic=deterministic, removed_tanh=removed_tanh, return_log_prob=False, 
                                     fixed_sigma=fixed_sigma, clip_action=clip_action, b=b,
                                     beta_action=beta_action, beta=beta,
                                     SOP=SOP,
                                     record_mu=record_mu)
            action_tensor = action.detach()
            mu = mean.detach().numpy()
            pre_tanh_value = pre_tanh_value.detach().numpy()
            action_np = action_tensor.numpy().reshape(-1)
            return action_np, mu, pre_tanh_value
        else:
            action_tensor  = self.forward(obs_tensor, deterministic=deterministic, removed_tanh=removed_tanh, return_log_prob=False, 
                                     fixed_sigma=fixed_sigma, clip_action=clip_action, b=b,
                                     beta_action=beta_action, beta=beta,
                                     SOP=SOP,
                                     record_mu=record_mu)[0].detach()
        ## convert action into the form that can put into the env and scale it

            action_np = action_tensor.numpy().reshape(-1)
            return action_np

    def forward(
            self,
            obs,
            batch_size = 256,
            deterministic=False,
            return_log_prob=True,
            fixed_sigma=False,
            removed_tanh=False,
            closed_entropy=False,
            tanh_entropy=False,
            mu_l2=False,
            mu_l1=False,
            mu_l4=False,
            entropy_net=None,
            penal_a=False,
            clip_action=False,
            b=None,
            beta_action=False,
            beta=None,
            SOP=False,
            record_mu=False
    ):
        """
        :param obs: Observation
        :param reparameterize: if True, use the reparameterization trick
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        mean = self.last_fc_layer(h)

        if fixed_sigma:
            std = torch.zeros(mean.size())
            std += 0.3
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)

        else:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)

        if clip_action:
            zeros = torch.zeros(mean.size())
            normal = Normal(zeros, std)
            mean = torch.clamp(mean,min=-b,max=b)

        elif beta_action:
            zeros = torch.zeros(mean.size())
            normal = Normal(zeros, std)
            mean = mean/beta

        elif SOP:
            zeros = torch.zeros(mean.size())
            normal = Normal(zeros, std)
            K = torch.tensor(mean.size()[1])
            abs_mean = torch.abs(mean)
            Gs = torch.sum(abs_mean, dim=1).view(-1,1) ######
            Gs = Gs/K
            Gs = Gs/beta
            mean = mean/Gs
        else:
            normal = Normal(mean, std)

        if closed_entropy:
            entropy = normal.entropy()
            entropy = torch.sum(entropy, dim=1).view(-1,1)

        if tanh_entropy:
            pair = torch.stack((mean,std), 2).reshape(batch_size, -1, 2)
            entropy = entropy_net(pair)
            entropy = torch.sum(entropy, dim=1).view(-1,1)
            #print (entropy.size())

        if mu_l2:
            entropy = - torch.norm(mean, dim=1).view(-1,1)

        if mu_l1:
            entropy = - torch.norm(mean, p=1, dim=1).view(-1,1)

        if mu_l4:
            entropy = - torch.norm(mean, p=4, dim=1).view(-1,1)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            if clip_action:
                pre_tanh_value = mean + normal.rsample()
            elif beta_action:
                pre_tanh_value = mean + normal.rsample()
            elif SOP:
                pre_tanh_value = mean + normal.rsample()
            else:
                pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if penal_a:
            penalty = - torch.norm(pre_tanh_value, p=1, dim=1).view(-1,1)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            if removed_tanh:
                log_prob -= torch.log(1 - pre_tanh_value.pow(2) + ACTION_BOUND_EPSILON)
            else:
                log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        if closed_entropy:
            return (
                action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value, entropy
            )
        elif tanh_entropy:
            return (
                action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value, entropy
            )
        elif removed_tanh:
            return (
                pre_tanh_value * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
            )
        elif mu_l1:
            return (
                action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value, entropy
            )
        elif mu_l2:
            return (
                action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value, entropy
            )
        elif mu_l4:
            return (
                action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value, entropy
            )
        elif penal_a:
            return (
                action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value, penalty
            )
        else:
            return (
                action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
            )


def soft_update_model1_with_model2(model1, model2, rou):
    """
    see openai spinup sac psudocode line 16, used to update target_value_net
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(
            rou*model1_param.data + (1-rou)*model2_param.data
        )
