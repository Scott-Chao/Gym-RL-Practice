import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DuelingQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNet, self).__init__()

        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        # 状态价值流 (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # 动作优势流 (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        features = self.feature(x)

        v = self.value_stream(features)  # Shape: [batch, 1]
        a = self.advantage_stream(features)  # Shape: [batch, action_dim]

        # Q = V + (A - A.mean)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class SumTree:
    """底层 SumTree 结构，用于高效存储优先级和采样"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_idx = 0

    def update(self, tree_idx, p):
        """更新某个叶子节点的优先级，并递归更新父节点"""
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p):
        """在新数据进入时，初始化其优先级并放入叶子节点"""
        tree_idx = self.data_idx + self.capacity - 1
        self.update(tree_idx, p)
        self.data_idx = (self.data_idx + 1) % self.capacity

    def get_leaf(self, v):
        """根据采样值 v，检索对应的叶子节点、优先级和索引"""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                break

            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        leaf_idx = parent_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_dim, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.tree = SumTree(capacity)

        self.max_priority = 1.0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.tree.add(self.max_priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices, tree_indices, priorities = [], [], []

        segment = self.tree.total_priority / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            v = np.random.uniform(segment * i, segment * (i + 1))
            t_idx, p, d_idx = self.tree.get_leaf(v)
            tree_indices.append(t_idx)
            priorities.append(p)
            indices.append(d_idx)

        # 计算 IS Weights
        probs = np.array(priorities) / self.tree.total_priority
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()

        return (
            torch.FloatTensor(self.states[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]),
            tree_indices,
            torch.FloatTensor(weights).view(-1, 1),
        )

    def update_priorities(self, tree_indices, td_errors):
        """训练完一个 Batch 后，根据最新的 TD-Error 更新优先级"""
        priorities = (np.abs(td_errors) + 1e-5) ** self.alpha
        for t_idx, p in zip(tree_indices, priorities):
            self.tree.update(t_idx, p)

        self.max_priority = max(self.max_priority, np.max(priorities))


class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
        if len(self.buffer) < self.n:
            return None

        # 计算 n-step return
        # G = r0 + g*r1 + g^2*r2...
        reward = 0
        for i, exp in enumerate(self.buffer):
            reward += (self.gamma**i) * exp[2]

        # 取第 0 步的状态动作，和第 n 步的下一个状态
        state, action, _, _, _ = self.buffer[0]
        _, _, _, next_state, done = self.buffer[-1]

        return state, action, reward, next_state, done


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = DuelingQNet(state_dim, action_dim)
        self.target_net = DuelingQNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.memory = PrioritizedReplayBuffer(100000, state_dim)
        self.gamma = 0.99
        self.batch_size = 128
        self.action_dim = action_dim

        self.n_step = 3

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state_t).argmax().item()

    def store(self, s, a, r, s_next, done):
        self.memory.add(s, a, r, s_next, done)

    def train(self):
        if self.memory.size < self.batch_size:
            return 0.0

        s, a, r, s_next, done, tree_indices, weights = self.memory.sample(self.batch_size)

        # 预测当前 Q 值
        q_values = self.q_net(s).gather(1, a)
        avg_q = q_values.mean().item()

        # 计算 Target Q 值
        with torch.no_grad():
            best_actions = self.q_net(s_next).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(s_next).gather(1, best_actions)
            target_q = r + (self.gamma**self.n_step) * max_next_q * (1 - done)

        diff = q_values - target_q

        td_errors = diff.detach().abs().squeeze().cpu().numpy()
        self.memory.update_priorities(tree_indices, td_errors)

        loss = (weights * diff.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target()
        return avg_q

    def update_target(self, tau=0.005):
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
