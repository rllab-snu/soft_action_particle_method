import numpy as np
import gym
import random
import collections
import matplotlib.pyplot as plt
import time

from spinup.algos.sapm import Defs
from spinup.algos.sapm.Q_2layers import Q2_Network
from spinup.algos.sapm.Q_3layers import Q3_Network
from spinup.utils.logx import EpochLogger

"""

SAPM (Soft Action Particle Method)

"""
def sapm(env_fn, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=200, replay_size=int(1e6),
        
        #######################################options########################################
        # N_layers : 2(2layers Q-Network), 3(3layers Q-Network)
        # DPP      : 0(Random Sampling), 1(DPP Sampling)
        # Ablation : 0(Fixed Particles), 1(Grad + No Resampling), 2(Grad + Resampling), 3(SAPM)
        N_layers = 2, DPP = 1, Ablation = 3,
        ######################################################################################
         
        layer_size1=300, layer_size2=400, lr_Q=0.0001, lr_M=0.01, lr_S=0.01,
        batch_size=100, start_steps=10000, act_train_step=100, policy_delay=1,
        
        act_size=32, min_d=0.5, scale=0.2, std=10,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    np.random.seed(seed)    
    rng = np.random.RandomState(seed)
    random.seed(seed)
    
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    env.seed(seed)
    test_env.seed(seed)
    
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    
    replay_buffer = collections.deque()
    if N_layers == 2:
        Critic_Network = Q2_Network(seed, obs_dim, act_dim, act_size, act_limit, batch_size, scale, layer_size1, layer_size2, lr_Q, lr_M, lr_S, std, Ablation)
    elif N_layers == 3:
        Critic_Network = Q3_Network(seed, obs_dim, act_dim, act_size, act_limit, batch_size, scale, layer_size1, layer_size2, lr_Q, lr_M, lr_S, std, Ablation)
    
    M_batch, S_batch = Critic_Network.get_action_batch()
    t_M_batch, t_S_batch = Critic_Network.get_target_action_batch()
    
    """
    # Count variables
    scopes = ['actions/action_batch', 'q_critic', 'H1']
    var_counts = tuple(Critic_Network.count_vars(scope) for scope in scopes)
    print(var_counts)
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)
    
    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})
    """
    
    def get_action(o):
        actions, action_soft = Critic_Network.get_softmax(o, scale)
        action0 = np.random.choice(len(action_soft), size=1, p=action_soft)[0]
        return actions[action0]
        
    def test_get_action(o, actions):
        action_test = Critic_Network.get_test_q_batch(np.reshape(o,[1,-1]), actions)
        action_test = np.reshape(action_test, [1,-1])[0]
        return actions[np.argmax(action_test)]
      
    def test_agent(actions, flag=True, n=10):
        result = 0
        for j in range(n):
            o_t, r_t, d_t, ep_ret_t, ep_len_t = test_env.reset(), 0, False, 0, 0
            while not(d_t or (ep_len_t == max_ep_len)):
                o_t, r_t, d_t, _ = test_env.step(test_get_action(o_t, actions))
                ep_ret_t += r_t
                ep_len_t += 1
            result += ep_ret_t
            
            if flag:
                logger.store(TestEpRet2=ep_ret_t, TestEpLen=ep_len_t)
            else:
                logger.store(TestEpRet=ep_ret_t, TestEpLen=ep_len_t)
        
        return result/n

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    global_steps, max_ret = 0, -10000
    best_actions = M_batch

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        if t > start_steps:
            a = get_action(o)
        else:
            a0 = np.random.randint(act_size)
            a = np.random.normal(M_batch[a0], S_batch[a0])

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        global_steps += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.append((o, o2, a, r, d))
        if len(replay_buffer)>replay_size:
            replay_buffer.popleft()
        
        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if (d or (ep_len == max_ep_len)) and global_steps > batch_size:
            for tr in range(global_steps-ep_len+1, global_steps+1):
                minibatch = random.sample(replay_buffer, batch_size)
                cost, q_t_1, state_t_batch = Defs.train(Critic_Network, minibatch, scale, obs_dim, act_dim, act_size, act_limit)
                logger.store(LossQ=cost, QVals=q_t_1)
                    
                if tr % policy_delay == 0:
                    Critic_Network.update_target_critic()
                
                if (Ablation!=0) and (tr % act_train_step == 0):
                    for k in range(1):
                        actor_loss, _, _ = Critic_Network.train_weighted_actor(state_t_batch)
                        logger.store(LossPi=np.sum(actor_loss))
                        
                        M_batch, S_batch = Critic_Network.get_action_batch()
                        t_M_batch, t_S_batch = Critic_Network.get_target_action_batch()
                        
                        M_batch, S_batch, t_M_batch, t_S_batch = Defs.clipping_action(M_batch, S_batch, t_M_batch, t_S_batch, act_limit, std)
                        
                        if Ablation!=1:
                            if DPP == 0:
                                M_batch, S_batch, t_M_batch, t_S_batch = Defs.resampling_random(M_batch, S_batch, t_M_batch, t_S_batch, act_limit, std, min_d, np)
                            elif DPP == 1:
                                M_batch, S_batch, t_M_batch, t_S_batch = Defs.resampling_dpp(M_batch, S_batch, t_M_batch, t_S_batch, act_limit, std, min_d, np)
                            else:
                                print("   It is invalid RESAMPLING option.   ")
                        
                        Critic_Network.realign_action_batch(M_batch, S_batch, t_M_batch, t_S_batch)
                    
                    Critic_Network.update_action_target_critic()
                    
                M_batch, S_batch = Critic_Network.get_action_batch()
                t_M_batch, t_S_batch = Critic_Network.get_target_action_batch()
                    
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            Defs.get_action_variance(M_batch, min_d, act_limit)

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_result1 = test_agent(M_batch, flag=False)
            test_result2 = test_agent(best_actions, flag=True)
            if max_ret < test_result1 and test_result2 < test_result1:
                best_actions = M_batch
                
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet2', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    
    parser.add_argument('--DPP', type=int, default=0)
    parser.add_argument('--N_layers', type=int, default=2)
    parser.add_argument('--Ablation', type=int, default=3)
    
    parser.add_argument('--act_size', type=int, default=32)
    parser.add_argument('--min_d', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sapm')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    sapm(lambda : gym.make(args.env), #actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)