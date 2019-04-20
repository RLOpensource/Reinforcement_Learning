import tensorflow as tf
import model
import collections
import numpy as np
import random
import gym
import buffer
from tensorboardX import SummaryWriter

class DQN:
    def __init__(self, env, max_length, state_size, output_size, hidden, n_step, batch_size,
                    gamma, lr, train_size, activation):
        self.epsilon = 1.0
        self.sess = tf.Session()
        self.env = gym.make(env)
        self.max_length = max_length
        self.state_size = state_size
        self.output_size = output_size
        self.hidden = hidden
        self.n_step = n_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.train_size = train_size
        self.tau = 0.995
        self.activation = activation
        self.memory = buffer.replay_buffer(n_step_size=self.n_step, gamma=self.gamma, max_length=self.max_length)
        self.x_ph, self.G_ph = \
            model.placeholders(self.state_size, None)
        self.a_ph = tf.placeholder(tf.int32, shape=[None])

        with tf.variable_scope('main'):
            self.main = model.dueling(self.x_ph, self.hidden, self.activation, self.output_size)

        with tf.variable_scope('target'):
            self.target = model.dueling(self.x_ph, self.hidden, self.activation, self.output_size)

        self.one_hot_a_ph = tf.one_hot(self.a_ph, depth=self.output_size)
        self.main_q_value = tf.reduce_sum(self.main * self.one_hot_a_ph, axis=1)

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

    def write_parameter(self, main_p, target_p):
        feed_dict={i:j for i, j in zip(self.from_list_main, main_p)}
        self.sess.run(self.write_main_parameter, feed_dict=feed_dict)
        feed_dict={i:j for i, j in zip(self.from_list_target, target_p)}
        self.sess.run(self.write_target_parameter, feed_dict=feed_dict)

    def get_parameter(self):
        main_p, target_p = self.sess.run([self.main_params, self.target_params])
        return main_p, target_p

    def update_parameter(self):
        #self.sess.run(self.op_holder)
        self.sess.run(self.target_update)

    def update(self):
        sample = self.memory.get_sample(self.batch_size)
        state = sample['state'][:, 0]
        action = sample['action'][:, 0]
        next_state = sample['next_state'][:, -1]
        reward = sample['reward']
        done = sample['done'][:, -1]

        for r in reward:
            for i, y in enumerate(r):
                r[i] = np.power(self.gamma, i) * y
        
        reward = np.sum(reward, axis=1)
        q_value_for_select_action = self.sess.run(self.main, feed_dict={self.x_ph: next_state})
        selected_action = np.argmax(q_value_for_select_action, axis=1)
        target_q_value = self.sess.run(self.target, feed_dict={self.x_ph: next_state})
        target_value = [np.power(self.gamma, self.n_step) * q[a] * (1-int(d)) for a, q, d in zip(selected_action, target_q_value, done)]

        target_value, reward = np.stack(target_value), np.stack(reward)
        G_t = target_value + reward

        _, loss = self.sess.run(
            [self.train, self.loss],
            feed_dict={self.x_ph: state, self.a_ph: action, self.G_ph: G_t}
            )
        return loss

    def get_action(self, state, epsilon):
        if np.random.rand() > epsilon:
            q_value = self.sess.run(self.main, feed_dict={self.x_ph: [state]})
            q_value = q_value[0]
            action = np.argmax(q_value, axis=0)
        else:
            action = np.random.randint(self.output_size)
        return action

    def run(self):
        writer = SummaryWriter()
        for i in range(99999999):
            self.epsilon = 1 / (i+1)
            done = False
            state = self.env.reset()
            score = 0
            step = 0
            step_per_loss = 0
            self.memory.n_step.reset()
            while not done:
                if i % 10 == 0:
                    self.env.render()
                step += 1
                action = self.get_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.memory.append(state, next_state, reward, done, action)
                state = next_state
                if len(self.memory.memory) > self.batch_size:
                    if step % self.train_size == 0:
                        step_per_loss += 1
                        loss = self.update()
                        self.update_parameter()
            writer.add_scalar('data/score', score, i)
            if len(self.memory.memory) > self.batch_size:
                writer.add_scalar('data/loss', loss/step_per_loss, i)
            print(score, i) 

if __name__ == '__main__':
    agent = DQN(
        env='MountainCar-v0',
        max_length=1e3,
        state_size=2,
        output_size=3,
        hidden=[256, 256, 256],
        n_step=10,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        train_size=1,
        activation=tf.nn.relu
    )

    agent.run()