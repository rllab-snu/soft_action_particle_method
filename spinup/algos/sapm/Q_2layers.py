# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math

TAU = 0.001

class Q2_Network:
    def __init__(self, seed, dim_obs, dim_act, act_size, act_scale, batch_size, scale, ls_1, ls_2, lr_Q, lr_M, lr_S, std, SAPM):
        
        self.seed = seed
        self.dim_state, self.dim_action = dim_obs, dim_act
        self.batch_size = batch_size
        self.action_size, self.action_scale = act_size, act_scale
        self.sc, self.std = scale, std
        self.SAPM = True if (SAPM == 3) else False
        
        self.lr_Q, self.lr_M, self.lr_S = lr_Q, lr_M, lr_S
        self.N_HIDDEN_1, self.N_HIDDEN_2 = ls_1, ls_2
        
        tf.set_random_seed(seed)
        self.g = tf.Graph()
        with self.g.as_default():
            self.create_placeholder()
            self.create_action_batch()
            self.create_net()
            self.create_optimizer()
            self.create_softmax()
            self.create_update_operation()
            self.init_session()
            
    def create_placeholder(self):
        self.A_batch = tf.placeholder("float32", [None, self.dim_action])
        self.t_A_batch = tf.placeholder("float32", [None, self.dim_action])
        
        self.M_batch = tf.placeholder("float32", [None, self.dim_action])
        self.t_M_batch = tf.placeholder("float32", [None, self.dim_action])
        self.S_batch = tf.placeholder("float32", [None, self.dim_action])
        self.t_S_batch = tf.placeholder("float32", [None, self.dim_action])
        
        self.state_in = tf.placeholder("float",[None, self.dim_state])
        self.action_in = tf.placeholder("float",[None, self.dim_action])
        self.means = tf.placeholder("float",[None, self.dim_action])
        self.stds = tf.placeholder("float",[None, self.dim_action])
        self.weighted_action_in = tf.placeholder("float", [self.batch_size, self.action_size])
        
        self.q_value_in = tf.placeholder("float",[None,1])
        self.scale = tf.placeholder("float")
    
    # create the random action set
    def create_action_batch(self):
        self.action_mean = tf.Variable(tf.random_uniform([self.action_size, self.dim_action], seed=self.seed, minval = -self.action_scale, maxval = self.action_scale), name = "action_mean")
        self.action_std = tf.Variable(tf.random_uniform([self.action_size, self.dim_action], seed=self.seed, minval = 0., maxval = self.action_scale/self.std), name = "action_std")
        #self.action_std = tf.Variable(tf.ones([self.action_size, self.dim_action]), name = "action_std")
        
        self.t_action_mean = tf.Variable(tf.random_uniform([self.action_size, self.dim_action], seed=self.seed, minval = -self.action_scale, maxval = self.action_scale), name = "t_action_mean")
        self.t_action_std = tf.Variable(tf.random_uniform([self.action_size, self.dim_action], seed=self.seed, minval = 0., maxval = self.action_scale/self.std), name = "t_action_std")
        #self.t_action_std = tf.Variable(tf.ones([self.action_size, self.dim_action]), name = "t_action_std")
        
    def create_net(self):
        tf.set_random_seed(self.seed)
        N_HIDDEN_1 = self.N_HIDDEN_1
        N_HIDDEN_2 = self.N_HIDDEN_2
        
        with tf.variable_scope('q_critic'):
            normals = self.means + self.stds*tf.random_normal([self.action_size, self.dim_action], 0.0, 1.0)
            self.actions = tf.clip_by_value(normals, -self.action_scale, self.action_scale)
            
            h1 = tf.layers.dense(self.state_in, N_HIDDEN_1, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1')
            h2 = tf.layers.dense(h1, N_HIDDEN_2,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2')
            h3 = tf.layers.dense(self.actions, N_HIDDEN_2,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H3')
            self.q_predict = tf.layers.dense(tf.nn.relu(h2+h3), 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='critic')
            
            h4 = tf.layers.dense(self.action_in, N_HIDDEN_2, reuse=True, name='H3')
            self.q_train = tf.layers.dense(tf.nn.relu(h2+h4), 1, reuse=True, name='critic')
            
            state = tf.tile(h2, [1, self.action_size])
            state = tf.reshape(state, [-1, N_HIDDEN_2])
            action = tf.tile(h3, [self.batch_size, 1])
            stateXaction = state + action
            self.stateXaction = tf.layers.dense(tf.nn.relu(stateXaction), 1, reuse=True, name='critic')
            
            q_batch = tf.reshape(self.stateXaction, [self.batch_size, -1])/self.scale
            max_q = tf.reshape(tf.reduce_max(q_batch, axis=1), [-1,1])
            e_q = tf.exp(q_batch - max_q)
            action_weight = e_q/tf.reshape(tf.reduce_sum(e_q, axis=1), [-1,1])
            action_weight = tf.stop_gradient(action_weight)
            
            Q = tf.reshape(self.stateXaction, [self.batch_size, -1])
            Q = tf.reduce_sum(Q * action_weight, axis = 0)
            self.w_predict = tf.reshape(Q, [-1, 1])
            
            Q2 = tf.reshape(self.stateXaction, [self.batch_size, -1])
            Q2 = tf.reduce_sum(Q2, axis = 0)
            self.w2_predict = tf.reshape(Q2, [-1, 1])
            
        with tf.variable_scope('q_target'):
            normals_t = self.means + self.stds*tf.random_normal([self.action_size, self.dim_action], 0.0, 1.0)
            self.actions_t = tf.clip_by_value(normals_t, -self.action_scale, self.action_scale)
            
            h1_t = tf.layers.dense(self.state_in, N_HIDDEN_1, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1_t')
            h2_t = tf.layers.dense(h1_t, N_HIDDEN_2,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2_t')
            h3_t = tf.layers.dense(self.actions_t, N_HIDDEN_2,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H3_t')
            self.q_target = tf.layers.dense(tf.nn.relu(h2_t+h3_t), 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='target')
            
            state_t = tf.tile(h2_t, [1, self.action_size])
            state_t = tf.reshape(state_t, [-1, N_HIDDEN_2])
            action_t = tf.tile(h3_t, [self.batch_size, 1])
            stateXaction_t = state_t + action_t
            self.stateXaction_t = tf.layers.dense(tf.nn.relu(stateXaction_t), 1, reuse=True, name='target')
            
        self.weights_c = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_critic')
        self.weights_t = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')
        
    def create_optimizer(self):
        self.error = self.q_train-self.q_value_in
        
        self.cost = tf.pow(self.error,2)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_Q).minimize(self.cost)
        
        self.m_actor = [self.means]
        self.m_gradients = tf.gradients(self.w_predict, self.m_actor, tf.fill((self.action_size, 1), -1.))
        self.m_optimizer = tf.train.AdamOptimizer(self.lr_M).apply_gradients(zip(self.m_gradients,[self.action_mean]))
        
        self.s_actor = [self.stds]
        self.s_gradients = tf.gradients(self.w_predict, self.s_actor, tf.fill((self.action_size, 1), -1.))
        self.s_optimizer = tf.train.AdamOptimizer(self.lr_S).apply_gradients(zip(self.s_gradients,[self.action_std]))
        
        self.m2_actor = [self.means]
        self.m2_gradients = tf.gradients(self.w2_predict, self.m2_actor, tf.fill((self.action_size, 1), -1.))
        self.m2_optimizer = tf.train.AdamOptimizer(self.lr_M).apply_gradients(zip(self.m2_gradients,[self.action_mean]))
        
        self.s2_actor = [self.stds]
        self.s2_gradients = tf.gradients(self.w2_predict, self.s2_actor, tf.fill((self.action_size, 1), -1.))
        self.s2_optimizer = tf.train.AdamOptimizer(self.lr_S).apply_gradients(zip(self.s2_gradients,[self.action_std]))
        
    def create_softmax(self):
        action_Q = tf.reshape(self.q_predict, [-1])/self.scale
        max_x = tf.reduce_max(action_Q, axis=0)
        e_x = tf.exp(action_Q - max_x)
        p = e_x/tf.reduce_sum(e_x, axis=0)
        self.softmax = p/tf.reduce_sum(p, axis=0)
        
        z = tf.reshape(self.stateXaction_t, [self.batch_size, -1])/self.scale
        max_z = tf.reduce_max(z, axis=1)
        e_z = tf.exp(z - max_z[:, tf.newaxis])
        e_sum = tf.reduce_sum(e_z, axis=1)
        self.softV = self.scale * (tf.log(e_sum) + max_z)
    
    def create_update_operation(self):
        copy_net_ops = []
        for var, var_old in zip(self.weights_c, self.weights_t):
            copy_net_ops.append(var_old.assign(var))
        self.copy_net_ops = copy_net_ops
        
        update_net_ops = []
        for var, var_old in zip(self.weights_c, self.weights_t):
            update_net_ops.append(var_old.assign(TAU*var+(1-TAU)*var_old))
        self.update_net_ops = update_net_ops
        
        self.copy_action_ops = [self.t_action_mean.assign(self.action_mean), self.t_action_std.assign(self.action_std)]
        self.update_action_ops = [self.t_action_mean.assign(TAU*self.action_mean+(1-TAU)*self.t_action_mean),
                                  self.t_action_std.assign(TAU*self.action_std+(1-TAU)*self.t_action_std)]
        
        self.update_action_batch_op = [
            self.action_mean.assign(self.M_batch),
            self.action_std.assign(self.S_batch),
            self.t_action_mean.assign(self.t_M_batch),
            self.t_action_std.assign(self.t_S_batch)
        ]
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run([self.copy_net_ops, self.copy_action_ops])
        
    def realign_action_batch(self, M_batch, S_batch, t_M_batch, t_S_batch):
        self.sess.run(self.update_action_batch_op, feed_dict={self.M_batch:M_batch, self.S_batch:S_batch, self.t_M_batch:t_M_batch, self.t_S_batch:t_S_batch})
        
    def update_target_critic(self):
        self.sess.run(self.update_net_ops)
    
    def update_action_target_critic(self):
        self.sess.run(self.update_action_ops)
        
    def train_critic(self, state_t_batch, action_t_batch, y_batch):
        return self.sess.run([self.error, self.cost, self.optimizer],
                     feed_dict={self.state_in:state_t_batch, self.action_in:action_t_batch, self.q_value_in:y_batch})
    
    def get_q_batch(self, state_t):
        means, stds= self.sess.run([self.action_mean, self.action_std])
        q_batch = self.sess.run(self.q_predict,\
                     feed_dict={self.state_in: state_t, self.means:means, self.stds:stds})
        return q_batch
    
    def get_test_q_batch(self, state_t, test_actions):
        q_batch = self.sess.run(self.q_train,\
                     feed_dict={self.state_in: state_t, self.action_in:test_actions})
        return q_batch
    
    def get_stateXq_batch(self, state_t):
        means, stds= self.sess.run([self.action_mean, self.action_std])
        q_batch = self.sess.run(self.stateXaction, feed_dict={self.state_in: state_t, self.means:means, self.stds:stds})
        return q_batch
    
    def get_target_q_batch(self, state_t):
        means, stds= self.sess.run([self.t_action_mean, self.t_action_std])
        t_q_batch = self.sess.run(self.stateXaction_t, feed_dict={self.state_in: state_t, self.means:means, self.stds:stds})
        return t_q_batch
    
    def get_action_batch(self):
        return self.sess.run([self.action_mean, self.action_std])
    
    def get_target_action_batch(self):
        return self.sess.run([self.t_action_mean, self.t_action_std])
    
    def train_weighted_actor(self, state_t):
        means, stds = self.sess.run([self.action_mean, self.action_std])
        if self.SAPM :
            return self.sess.run([self.w_predict, self.m_optimizer, self.s_optimizer],\
                            feed_dict={self.state_in:state_t, self.means:means, self.stds:stds, self.scale:self.sc})
        else :
            return self.sess.run([self.w2_predict, self.m2_optimizer, self.s2_optimizer],\
                            feed_dict={self.state_in:state_t, self.means:means, self.stds:stds, self.scale:self.sc})
    
    def get_softmax(self, state, scale=1.):
        means, stds= self.sess.run([self.action_mean, self.action_std])
        return self.sess.run([self.actions, self.softmax], feed_dict={self.state_in:np.reshape(state, [1,-1]), self.means: means, self.stds: stds, self.scale:scale})
    
    def get_softV(self, state, scale=1.):
        means, stds= self.sess.run([self.t_action_mean, self.t_action_std])
        return self.sess.run(self.softV, feed_dict={self.state_in:state, self.means:means, self.stds:stds, self.scale:scale})
    
    