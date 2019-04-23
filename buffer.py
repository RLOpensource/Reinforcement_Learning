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

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)