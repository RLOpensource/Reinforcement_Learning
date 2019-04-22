import tensorflow as tf
import model
import collections
import numpy as np
import random
import gym
import buffer
from tensorboardX import SummaryWriter

class DQN:
    def __init__(self, max_length, state_size, output_size, hidden, n_step, batch_size,
                    gamma, lr, train_size, update_size, activation):
        self.epsilon = 1.0
        self.sess = tf.Session()
        self.max_length = max_length
        self.state_size = state_size
        self.output_size = output_size
        self.hidden = hidden
        self.n_step = n_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.train_size = train_size
        self.update_size = update_size
        self.tau = 0.995
        self.activation = activation
        
        self.n_step_buffer = buffer.n_step_buffer(n_step=self.n_step)
        self.memory = buffer.Memory(capacity=int(self.max_length))

        self.state_shape = [None]

        if type(self.state_size) == int:
            self.state_shape.append(self.state_size)
        elif type(self.state_size) == list:
            self.state_shape.extend(self.state_size)

        self.G_ph = tf.placeholder(tf.float32, shape=[None])
        self.x_ph = tf.placeholder(tf.float32, shape=self.state_shape)
        self.a_ph = tf.placeholder(tf.int32, shape=[None])
        self.w    = tf.placeholder(tf.float32, shape=[None])

        if type(self.state_size) == int:
            with tf.variable_scope('main'):
                self.main = model.dueling(self.x_ph, self.hidden, self.activation, self.output_size)

            with tf.variable_scope('target'):
                self.target = model.dueling(self.x_ph, self.hidden, self.activation, self.output_size)

        elif type(self.state_size) == list:
            with tf.variable_scope('main'):
                self.main = model.cnn_dueling(self.x_ph, self.hidden, self.activation, self.output_size)
            
            with tf.variable_scope('target'):
                self.target = model.cnn_dueling(self.x_ph, self.hidden, self.activation, self.output_size)

        self.one_hot_a_ph = tf.one_hot(self.a_ph, depth=self.output_size)
        self.main_q_value = tf.reduce_sum(self.main * self.one_hot_a_ph, axis=1)

        self.unweighted_loss = ((self.G_ph - self.main_q_value) ** 2) * 0.5
        self.per_loss = tf.reduce_mean(self.unweighted_loss * self.w)

        self.per_optimizer = tf.train.AdamOptimizer(self.lr)
        self.per_train = self.per_optimizer.minimize(self.per_loss)

        self.loss = tf.reduce_mean((self.G_ph - self.main_q_value) ** 2) * 0.5
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train = self.optimizer.minimize(self.loss)

        self.main_params = model.get_vars('main')
        self.target_params = model.get_vars('target')

        self.op_holder = tf.group([tf.assign(v_targ, v_main)
                                    for v_main, v_targ in zip(model.get_vars('main'), model.get_vars('target'))])

        self.target_update = tf.group([tf.assign(v_targ, self.tau * v_targ + (1 - self.tau) * v_main)
                                    for v_main, v_targ in zip(model.get_vars('main'), model.get_vars('target'))])

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.from_list_main = [tf.placeholder(tf.float32, i.get_shape()) for i in self.main_params]
        self.from_list_target = [tf.placeholder(tf.float32, i.get_shape()) for i in self.target_params]

        self.write_main_parameter = tf.group([tf.assign(v_targ, v_main)
                                    for v_main, v_targ in zip(self.from_list_main, self.main_params)])

        self.write_target_parameter = tf.group([tf.assign(v_targ, v_main)
                                    for v_main, v_targ in zip(self.from_list_target, self.target_params)])

    def load_model(self, path):
        self.saver.restore(self.sess, path)

    def save_model(self, path):
        self.saver.save(self.sess, path)

    def write_parameter(self, main_p, target_p):
        feed_dict={i:j for i, j in zip(self.from_list_main, main_p)}
        self.sess.run(self.write_main_parameter, feed_dict=feed_dict)
        feed_dict={i:j for i, j in zip(self.from_list_target, target_p)}
        self.sess.run(self.write_target_parameter, feed_dict=feed_dict)

    def get_parameter(self):
        main_p, target_p = self.sess.run([self.main_params, self.target_params])
        return main_p, target_p

    def update_parameter(self):
        self.sess.run(self.op_holder)
        #self.sess.run(self.target_update)

    def update(self):
        minibatch, idxs, IS_weight = self.memory.sample(self.batch_size)
        minibatch = np.array(minibatch)
        state_batch = np.stack(minibatch[:, 0])
        next_state_batch = np.stack(minibatch[:, 1])
        discounted_reward_batch = np.stack(minibatch[:, 2])
        done_batch = np.stack(minibatch[:, 3])
        action_batch = np.stack(minibatch[:, 4])

        q_value_for_select_action = self.sess.run(self.main, feed_dict={self.x_ph: next_state_batch})
        selected_action = np.argmax(q_value_for_select_action, axis=1)
        target_q_value = self.sess.run(self.target, feed_dict={self.x_ph: next_state_batch})
        target_value = [np.power(self.gamma, self.n_step) * q[a] * (1-int(d)) for a, q, d in zip(selected_action, target_q_value, done_batch)]

        target = np.stack(discounted_reward_batch) + np.stack(target_value)

        _, loss = self.sess.run([self.per_train, self.per_loss], feed_dict={self.x_ph: state_batch, self.w: IS_weight, self.a_ph: action_batch, self.G_ph: target})

        state_value = self.sess.run(self.main, feed_dict={self.x_ph: state_batch})
        state_value = [sv[a] for a, sv in zip(action_batch, state_value)]
        
        td_error = np.abs(target - np.stack(state_value))

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, td_error[i])
        return loss

    def append_to_memory(self, state, next_state, reward, done, action):
        self.n_step_buffer.append(state, next_state, reward, done, action)
        n_step_state, n_step_next_state, n_step_reward, n_step_done, n_step_action = \
                        self.n_step_buffer.get_sample()
        if len(n_step_state) == self.n_step:
            discounted_n_step_reward = [np.power(self.gamma, i) * r for i, r in enumerate(n_step_reward)]
            discounted_n_step_reward_sum = np.sum(discounted_n_step_reward)
            q_value_for_select_action = self.sess.run(self.main, feed_dict={self.x_ph: [n_step_next_state[-1]]})
            selected_action = np.argmax(q_value_for_select_action, axis=1)[0]
            target_q_value = self.sess.run(self.target, feed_dict={self.x_ph: [n_step_next_state[-1]]})[0]
            target_value = (1-int(n_step_done[-1])) * np.power(self.gamma, self.n_step) * target_q_value[selected_action]
            target = discounted_n_step_reward_sum + target_value
            state_value = self.sess.run(self.main, feed_dict={self.x_ph: [n_step_state[0]]})[0]
            state_value = state_value[n_step_action[0]]
            td_error = np.abs(target - state_value)
            self.memory.add(td_error, (n_step_state[0], n_step_next_state[-1], discounted_n_step_reward_sum, n_step_done[-1], n_step_action[0]))

    def get_action(self, state, epsilon):
        if np.random.rand() > epsilon:
            q_value = self.sess.run(self.main, feed_dict={self.x_ph: [state]})
            q_value = q_value[0]
            action = np.argmax(q_value, axis=0)
            return action, q_value[action]
        else:
            action = np.random.randint(self.output_size)
            return action, None