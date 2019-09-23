import gym
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import time
from spinup.algos.sac_pytorch.SOP_core_auto import TanhGaussianPolicySACAdapt, Mlp, soft_update_model1_with_model2
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
import sys
"""
SOP with auto ERE
"""

class StagePriorityReplayBuffer_autoeta:
    """
    weighted buffer that basically
    gives more probability of sampling for more recent data
    """
    def __init__(self, obs_dim, act_dim, size, n_epoch):
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

        # performance buffer is used to adapt eta value
        self.epoch_performance_buf = np.zeros(n_epoch, dtype=np.float32)
        self.current_epoch = 0
        self.max_history_improvement = 1e-5

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

    def store_epoch_performance(self, test_score):
        self.epoch_performance_buf[self.current_epoch] = test_score
        self.current_epoch += 1

    def get_auto_eta(self, eta_init=0.994, baseline_epoch=100, ave_size=20):
        """
        This method assumes that the fastest learning occurs at the beginning of
        the learning. And it computes the baseline improvement based on the
        (baseline_improvement) = (mean return of 80-100 epoches) - (mean return of first 20 epoches)
        (recent_improvement) = (mean return of most recent 20 epoches) - (mean return of 100 to 80 epoches older)
        interpolation = recent_improvement / baseline_improvement
        then we interpolate between 0.994 and 1.0 with this interpolation
        """
        if self.current_epoch < baseline_epoch: # if don't have enough data
            return eta_init
        # if have enough data already
        baseline_improvement = self.epoch_performance_buf[baseline_epoch - ave_size:baseline_epoch].mean() \
                               - self.epoch_performance_buf[0:ave_size].mean()
        current_performance = self.epoch_performance_buf[self.current_epoch - ave_size:self.current_epoch].mean()
        previous_performance = self.epoch_performance_buf[self.current_epoch-100:self.current_epoch - 100 + ave_size].mean()
        recent_improvement = current_performance - previous_performance
        interpolation = recent_improvement/baseline_improvement
        if interpolation < 0:
            interpolation = 0
        auto_eta = eta_init * interpolation + 1 * (1-interpolation)
        return auto_eta

    def get_auto_eta_with_recent_baseline(self, eta_init=0.994, compare_interval=200):
        """
        This method uses improvement in slightly older data (instead of in the beginning stage)
        as the baseline improvement

        (baseline_improvement) = (mean return of 80-100 epoches) - (mean return of first 20 epoches)
        (recent_improvement) = (mean return of most recent 20 epoches) - (mean return of 100 to 80 epoches older)
        interpolation = recent_improvement / baseline_improvement
        then we interpolate between 0.994 and 1.0 with this interpolation
        """
        half_compare_interval = int(compare_interval/2)
        onethird_compare_interval = int(compare_interval/3)
        if self.current_epoch < half_compare_interval+onethird_compare_interval: # if don't have enough data
            return eta_init

        baseline_start = self.current_epoch - compare_interval #
        if baseline_start < 0:
            baseline_start = 0

        current_performance = self.epoch_performance_buf[self.current_epoch - onethird_compare_interval:self.current_epoch].mean()
        previous_performance_istart = self.current_epoch-half_compare_interval-int(onethird_compare_interval/2)
        previous_performance_iend = self.current_epoch - half_compare_interval + int(onethird_compare_interval/2)
        previous_performance = self.epoch_performance_buf[previous_performance_istart:previous_performance_iend].mean()
        baseline_performance = self.epoch_performance_buf[baseline_start:baseline_start+onethird_compare_interval].mean()

        recent_improvement = current_performance - previous_performance
        older_improvement = previous_performance - baseline_performance

        if older_improvement == 0:
            interpolation = 1
        else:
            interpolation = recent_improvement/older_improvement

        if interpolation < 0:
            interpolation = 0
        if interpolation > 1:
            interpolation = 1
        auto_eta = eta_init * interpolation + 1 * (1-interpolation)
        return auto_eta

    def get_auto_eta_max_history(self, eta_init=0.994, eta_final=0.999, baseline_epoch=100, ave_size=20):
        """
        100 epoch is 500,000 data
        This method uses the maximum improvement in history as the baseline
        by default:
        recent_performance = ave performance of last 20 epoches
        previous_performance = ave performance of last 80-100 epoches
        recent_improvement = recent_performance - previous_performance
        baseline_improvement = max recent_improvement over history
        interpolation = recent_improvement / baseline_improvement
        then we interpolate between 0.994 and 0.999 with this interpolation (can be a hyperparam)
        """
        if self.current_epoch < baseline_epoch: # if don't have enough data
            return eta_init
        # if have enough data already
        current_performance = self.epoch_performance_buf[self.current_epoch - ave_size:self.current_epoch].mean()
        previous_performance = self.epoch_performance_buf[self.current_epoch-100:self.current_epoch - 100 + ave_size].mean()
        recent_improvement = current_performance - previous_performance
        if recent_improvement > self.max_history_improvement:
            self.max_history_improvement = recent_improvement

        interpolation = recent_improvement/self.max_history_improvement
        # clip to range (0, 1)
        interpolation = np.clip(interpolation, a_min=0, a_max=1)
        auto_eta = eta_init * interpolation + eta_final * (1-interpolation)
        return auto_eta

    def sample_uniform_batch(self, batch_size=32):
        ## sample with replacement from buffer
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_priority_only_batch(self, force_priority_count, batch_size=32):
        ## EVERYTHING IS PRIORITY ONLY
        ## force priority count, when given, will override this replay buffer's default priority count
        recent_data_size = batch_size
        ## max index used for generate data indecies in a batch
        max_index = min(int(force_priority_count),self.size)
        ## relative indecies for selected recent data
        recent_relative_idxs = -np.random.randint(0, max_index, size=recent_data_size)
        recent_idxs = (self.ptr-1 + recent_relative_idxs)%self.size
        return dict(obs1=self.obs1_buf[recent_idxs],
                    obs2=self.obs2_buf[recent_idxs],
                    acts=self.acts_buf[recent_idxs],
                    rews=self.rews_buf[recent_idxs],
                    done=self.done_buf[recent_idxs])

def compute_current_eta(eta_initial, eta_final, current_timestep, total_timestep):
    ## linearly anneal eta as trained on more timesteps
    current_eta = eta_initial + (eta_final - eta_initial) * current_timestep/total_timestep
    return current_eta

def get_ck_list_exp(replay_size ,num_updates, eta_current, update_order):
    ck_list = np.zeros(num_updates, dtype=int)
    for k in range(num_updates):  ## compute ck for each k, using formula for old data first update
        ck_list[k] = int(replay_size * eta_current ** (k * 1000 / num_updates))
    if update_order == 'new_first':
        ck_list = np.flip(ck_list, axis=0)
    elif update_order == 'random':
        ck_list = np.random.permutation(ck_list)
    else:  ## 'old_first'
        pass
    return ck_list

def sop_aere(env_fn, hidden_sizes=[256, 256], seed=0,
             steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
             polyak=0.995, lr=3e-4, alpha=0, beta=1.2, batch_size=256, start_steps=10000,
             max_ep_len=1000, save_freq=1, dont_save=False, logger_store_freq=500,
             auto_alpha=False,
             update_order='old_first',
             auto_eta_mode=2, aeta_init=0.994, aeta_final=0.999,
             aeta_baseline_epoch=100, aeta_ave_size=20, c_min=5000, update_no_random=False,
             fixed_sigma_value_init=0.3, fixed_sigma_value_anneal_final=-1, grad_clip=-1,
             logger_kwargs=dict(), ):
    """
    Largely following OpenAI documentation
    But slightly different from tensorflow implementation
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        hidden_sizes: number of entries is number of hidden layers
            each entry in this list indicate the size of that hidden layer.
            applies to all networks

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch. Note the epoch here is just logging epoch
            so every this many steps a logging to stdouot and also output file will happen
            note: not to be confused with training epoch which is a term used often in literature for all kinds of
            different things

        epochs (int): Number of epochs to run and train agent. Usage of this term can be different in different
            algorithms, use caution. Here every epoch you get new logs

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration. However during testing the action always come from policy

        max_ep_len (int): Maximum length of trajectory / episode / rollout. Environment will get reseted if
        timestep in an episode excedding this number

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        logger_kwargs (dict): Keyword args for EpochLogger.

    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running on device:" ,device)

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env, test_env = env_fn(), env_fn()

    ## seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## seed environment along with env action space so that everything about env is seeded
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    test_env.seed(seed)
    test_env.action_space.np_random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # if environment has a smaller max episode length, then use the environment's max episode length
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()

    # init auto eta buffer
    replay_buffer = StagePriorityReplayBuffer_autoeta(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, n_epoch=epochs)

    def test_agent(n=5):
        """
        This will test the agent's performance by running n episodes
        During the runs, the agent only take deterministic action, so the
        actions are not drawn from a distribution, but just use the mean
        :param n: number of episodes to run the agent
        """
        ep_return_list = np.zeros(n)
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a = policy_net.get_env_action(o, deterministic=True, fixed_sigma=True, SOP=True, mod1=True, beta=beta)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            ep_return_list[j] = ep_ret
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    """init all networks"""
    # see line 1
    policy_net = TanhGaussianPolicySACAdapt(obs_dim, act_dim, hidden_sizes, action_limit=act_limit, device=device).to(device)
    q1_net = Mlp(obs_dim+act_dim,1,hidden_sizes).to(device)
    q2_net = Mlp(obs_dim+act_dim,1,hidden_sizes).to(device)

    q1_target_net = Mlp(obs_dim+act_dim,1,hidden_sizes).to(device)
    q2_target_net = Mlp(obs_dim+act_dim,1,hidden_sizes).to(device)

    # see line 2: copy parameters from value_net to target_value_net
    q1_target_net.load_state_dict(q1_net.state_dict())
    q2_target_net.load_state_dict(q2_net.state_dict())

    # set up optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(),lr=lr)
    q1_optimizer = optim.Adam(q1_net.parameters(),lr=lr)
    q2_optimizer = optim.Adam(q2_net.parameters(),lr=lr)

    # mean squared error loss for v and q networks
    mse_criterion = nn.MSELoss()

    # Main loop: collect experience in env and update/log each epoch
    # NOTE: t here is the current number of total timesteps used
    # it is not the number of timesteps passed in the current episode
    current_update_index = 0

    # for ere recording performance part, we record the total episode return over n episodes in the past 5000 timestep
    # then we get an average episode return, and use it as the epoch performance, store it in the buffer.
    # need to reset them at every 5000 timestep
    recent_total_ep_return = 0
    recent_n_episode = 0
    # flush for better hpc debugging
    sys.stdout.flush()
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        # sigma anneal
        if fixed_sigma_value_anneal_final >= 0:
            interpolation = t/total_steps
            current_sigma = (1-interpolation) * fixed_sigma_value_init + interpolation * fixed_sigma_value_anneal_final
        else:
            current_sigma = fixed_sigma_value_init

        # interact with env
        if t > start_steps:
            a = policy_net.get_env_action(o, deterministic=False, fixed_sigma=True, SOP=True, mod1=True, beta=beta, fixed_sigma_value=current_sigma)
        else:
            a = env.action_space.sample()
        # Step the env, get next observation, reward and done signal

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience (observation, action, reward, next observation, done) to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            Quoted from the original SAC paper: 'In practice, we take a single environment step
            followed by one or several gradient step' after a single environment step,
            the number of gradient steps is 1 for SAC. (see paper for reference)
            """
            ## first compute the current eta
            if auto_eta_mode == 1:
                eta_current = replay_buffer.get_auto_eta_with_recent_baseline(eta_init=aeta_init)
            elif auto_eta_mode == 2:
                eta_current = replay_buffer.get_auto_eta_max_history(eta_init=aeta_init, eta_final=aeta_final,
                                                                     baseline_epoch=aeta_baseline_epoch,
                                                                     ave_size=aeta_ave_size)
            else:
                eta_current = replay_buffer.get_auto_eta(eta_init=aeta_init,
                                                         baseline_epoch=aeta_baseline_epoch, ave_size=aeta_ave_size)

            num_updates = ep_len

            ck_list = get_ck_list_exp(replay_size, num_updates, eta_current, update_order)

            ## now we use exploratory policy's episode return to
            recent_total_ep_return += ep_ret
            recent_n_episode += 1

            for k in range(num_updates):
                c_k = ck_list[k]
                if c_k < c_min:
                    c_k = c_min

                # get data from replay buffer
                batch = replay_buffer.sample_priority_only_batch(c_k, batch_size)

                obs_tensor = Tensor(batch['obs1']).to(device)
                obs_next_tensor = Tensor(batch['obs2']).to(device)
                acts_tensor = Tensor(batch['acts']).to(device)
                # unsqueeze is to make sure rewards and done tensors are of the shape nx1, instead of n
                # to prevent problems later
                rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(device)
                done_tensor = Tensor(batch['done']).unsqueeze(1).to(device)

                """get q loss"""
                with torch.no_grad():
                    a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = \
                        policy_net.forward(obs_next_tensor,deterministic=update_no_random, fixed_sigma=True,
                                           SOP=True, mod1=True, beta=beta, fixed_sigma_value=current_sigma)

                    q1_next = q1_target_net(torch.cat([obs_next_tensor,a_tilda_next], 1))
                    q2_next = q2_target_net(torch.cat([obs_next_tensor,a_tilda_next], 1))

                    min_next_q = torch.min(q1_next,q2_next)
                    y_q = rews_tensor + gamma*(1-done_tensor)*min_next_q

                # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                q1_prediction = q1_net(torch.cat([obs_tensor, acts_tensor], 1))
                q1_loss = mse_criterion(q1_prediction, y_q)
                q2_prediction = q2_net(torch.cat([obs_tensor, acts_tensor], 1))
                q2_loss = mse_criterion(q2_prediction, y_q)

                """
                get policy loss
                """
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, _ = \
                    policy_net.forward(obs_tensor, deterministic=True, fixed_sigma=True,
                                       SOP=True, mod1=True, beta=beta, fixed_sigma_value=current_sigma)

                # see line 12: second equation
                q1_a_tilda = q1_net(torch.cat([obs_tensor,a_tilda],1))
                q2_a_tilda = q2_net(torch.cat([obs_tensor,a_tilda],1))
                min_q1_q2_a_tilda = torch.min(q1_a_tilda,q2_a_tilda)

                # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
                policy_loss = (- min_q1_q2_a_tilda).mean()

                """update networks"""
                q1_optimizer.zero_grad()
                q1_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(q1_net.parameters(), grad_clip)
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                q2_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(q2_net.parameters(), grad_clip)
                q2_optimizer.step()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
                policy_optimizer.step()

                # see line 16: update target value network with value network
                soft_update_model1_with_model2(q1_target_net, q1_net, polyak)
                soft_update_model1_with_model2(q2_target_net, q2_net, polyak)

                current_update_index += 1
                if current_update_index % logger_store_freq == 0:
                    # store diagnostic info to logger
                    logger.store(LossPi=policy_loss.item(), LossQ1=q1_loss.item(), LossQ2=q2_loss.item(),
                                 Q1Vals=q1_prediction.detach().cpu().numpy(),
                                 Q2Vals=q2_prediction.detach().cpu().numpy(), Eta=eta_current,
                                 )

            ## store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ## reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            """
            Save pytorch model, very different from tensorflow version
            We need to save the environment, the state_dict of each network
            and also the state_dict of each optimizer
            """
            # if not dont_save: TODO save is disabled for now
            #     sac_state_dict = {'env':env,'policy_net':policy_net.state_dict(),
            #                     'target_value_net':target_value_net.state_dict(),
            #                       'q1_net':q1_net.state_dict(), 'q2_net':q2_net.state_dict(),
            #                       'policy_opt':policy_optimizer, 'value_opt':value_optimizer,
            #                       'q1_opt':q1_optimizer, 'q2_opt':q2_optimizer}
            #     if (epoch % save_freq == 0) or (epoch == epochs-1):
            #         logger.save_state(sac_state_dict, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            # we cannot use test performance since that assumes we get more data
            last_epoch_ave_performance = recent_total_ep_return / recent_n_episode
            replay_buffer.store_epoch_performance(last_epoch_ave_performance)
            recent_total_ep_return = 0
            recent_n_episode = 0

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('Eta', average_only=True)
            logger.log_tabular('LastER', last_epoch_ave_performance)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sop_aere(lambda: gym.make(args.env), hidden_sizes=[args.hid] * args.l,
             gamma=args.gamma, seed=args.seed, epochs=args.epochs,
             steps_per_epoch=args.steps_per_epoch,
             logger_kwargs=logger_kwargs)