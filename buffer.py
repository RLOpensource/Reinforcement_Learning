import collections
import numpy as np
import random

class n_step_buffer:
    def __init__(self, n_step):
        self.n_step = n_step
        self.state = collections.deque(maxlen=int(self.n_step))
        self.next_state = collections.deque(maxlen=int(self.n_step))
        self.reward = collections.deque(maxlen=int(self.n_step))
        self.done = collections.deque(maxlen=int(self.n_step))
        self.action = collections.deque(maxlen=int(self.n_step))

    def reset(self):
        self.state = collections.deque(maxlen=int(self.n_step))
        self.next_state = collections.deque(maxlen=int(self.n_step))
        self.reward = collections.deque(maxlen=int(self.n_step))
        self.done = collections.deque(maxlen=int(self.n_step))
        self.action = collections.deque(maxlen=int(self.n_step))

    def append(self, state, next_state, reward, done, action):
        self.state.append(state)
        self.next_state.append(next_state)
        self.reward.append(reward)
        self.done.append(done)
        self.action.append(action)

    def get_sample(self):
        return np.stack(self.state), np.stack(self.next_state), np.stack(self.reward), np.stack(self.done), np.stack(self.action)

class replay_buffer:
    def __init__(self, gamma, n_step_size=3, max_length=1e6):
        self.gamma = gamma
        self.max_length = max_length
        self.key = ['state', 'next_state', 'reward', 'done', 'action']
        self.n_step_size = n_step_size
        self.n_step = n_step_buffer(n_step=self.n_step_size)
        self.memory = collections.deque(maxlen=int(self.max_length))

    def reset(self):
        self.memory = collections.deque(maxlen=int(self.max_length))

    def append_actor(self, state, next_state, reward, done, action):
        self.memory.append((state, next_state, reward, done, action))

    def append(self, state, next_state, reward, done, action):
        self.n_step.append(state, next_state, reward, done, action)
        n_step_state = self.n_step.get_sample()[0]
        n_step_next_state = self.n_step.get_sample()[1]
        n_step_reward = self.n_step.get_sample()[2]
        n_step_done = self.n_step.get_sample()[3]
        n_step_action = self.n_step.get_sample()[4]

        n_step_size = len(n_step_state)
        if n_step_size == self.n_step_size:
            self.memory.append((n_step_state, n_step_next_state,
                    n_step_reward, n_step_done, n_step_action))

    def get_sample(self, sample_size):
        batch = random.sample(self.memory, sample_size)
        state = np.stack([e[0] for e in batch])
        next_state = np.stack([e[1] for e in batch])
        reward = np.stack([e[2] for e in batch])
        done = np.stack([e[3] for e in batch])
        action = np.stack([e[4] for e in batch])
        batch_memory = [state, next_state, reward, done, action]

        return {k:v for k, v in zip(self.key, batch_memory)} 