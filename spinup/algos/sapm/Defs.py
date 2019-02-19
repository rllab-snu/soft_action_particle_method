# -*- coding: utf-8 -*-

import numpy as np
from spinup.algos.sapm import DPP

GAMMA = 0.99

def train(Q_Network, train_batch, s_scale, input_size, num_actions, size_action_batch, action_scale):
    
    state_t_batch, state_t_1_batch, action_batch, reward_batch, done_batch = zip(*train_batch)
    
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)
    
    q_t_1 = Q_Network.get_softV(state_t_1_batch, s_scale)
    q_t_1_batch = reward_batch + GAMMA*q_t_1*(1-done_batch)
    q_t_1_batch = np.reshape(q_t_1_batch,[-1,1])
    
    errors, cost, _ = Q_Network.train_critic(state_t_batch, action_batch, q_t_1_batch)
    cost = np.sum(cost, axis=1)
    #errors = np.sum(errors, axis=1)
    
    return cost, q_t_1, state_t_batch

def clipping_action(M_batch, S_batch, t_M_batch, t_S_batch, act_limit, std):
        M_batch, S_batch, t_M_batch, t_S_batch = np.array(M_batch), np.array(S_batch), np.array(t_M_batch), np.array(t_S_batch)
        
        max_M = np.max(np.abs(M_batch), axis=1, keepdims=True)
        B = max_M > act_limit
        B = ~B + B/max_M*act_limit
        M_batch = M_batch*B
        
        S_batch = np.clip(S_batch, 0, act_limit/std)
        t_S_batch = np.clip(t_S_batch, 0, act_limit/std)
        
        return M_batch, S_batch, t_M_batch, t_S_batch

def get_action_variance(A_batch, md, act_limit):
    A_batch = np.array(A_batch)
    num_actions = np.shape(A_batch)[0]
    dim_actions = np.shape(A_batch)[1]

    distance, num, num2, oor = 0., 0., 0., 0.
    for i in range(num_actions):
        if max(np.max(A_batch[i]), np.max(-A_batch[i])) > act_limit :
            oor += 1

        A = np.square(A_batch - A_batch[i])
        A = np.sum(A, axis = 1)

        num = num + (A < md).sum()
        num2 = num2 + (A < md*3).sum()

        A = np.sqrt(A)
        distance = distance + np.sum(A)

    num = (num-num_actions)*100/(num_actions*(num_actions-1))
    num = round(num, 3)

    num2 = (num2-num_actions)*100/(num_actions*(num_actions-1))
    num2 = round(num2, 3)

    distance = distance / (num_actions*(num_actions-1))
    distance = round(distance, 4)

    oor = round(oor / num_actions * 100, 3)
    print
    print ("      ( Action Batch  ::::  Distance : {},  Percent : {}%({}%),  OoR : {}% )".format(distance, num, num2, oor))
    print

def resampling_random(M_batch, S_batch, t_M_batch, t_S_batch, act_limit, std, md, npy):
    M_batch, S_batch, t_M_batch, t_S_batch = np.array(M_batch), np.array(S_batch), np.array(t_M_batch), np.array(t_S_batch)
    num_actions = np.shape(M_batch)[0]
    dim_actions = np.shape(M_batch)[1]

    for i in range(num_actions):
        M = np.square(M_batch - M_batch[i])
        M = np.sum(M, axis = 1)

        if (M < md).sum() > 1:
            M_batch[i] = npy.random.random(dim_actions)*2*act_limit - act_limit
            S_batch[i] = npy.random.random(dim_actions)*act_limit/std
            t_M_batch[i], t_S_batch[i] = M_batch[i], S_batch[i]

    return M_batch, S_batch, t_M_batch, t_S_batch

def resampling_dpp(M_batch, S_batch, t_M_batch, t_S_batch, act_limit, std, md, npy):
        M_batch, S_batch, t_M_batch, t_S_batch = np.array(M_batch), np.array(S_batch), np.array(t_M_batch), np.array(t_S_batch)
        num_actions = np.shape(M_batch)[0]
        dim_actions = np.shape(M_batch)[1]
        
        idx, k = [True]*num_actions, 0
        for i in range(num_actions-1):
            A = np.square(M_batch[i+1:,:] - M_batch[i])
            A = np.sum(A, axis = 1)
            if (A < md).sum() > 1:
                idx[i], k, S_batch[i] = False, k+1, npy.random.random(dim_actions)*act_limit/std
                t_S_batch[i] = S_batch[i]
                
        if k > 0:
            M_batch = np.array(M_batch)[idx]
            t_M_batch = np.array(t_M_batch)[idx]
            
            c_M_batch = npy.random.random([k*10, dim_actions])*2*act_limit - act_limit
            M_batch = DPP.sample_k(M_batch, c_M_batch, 0.1, k)
            t_M_batch = np.concatenate((t_M_batch, M_batch[(num_actions-k):,:]))
            
        return M_batch, S_batch, t_M_batch, t_S_batch

