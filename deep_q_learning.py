import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from visualizer import RLVisualizer

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.action_dim = action_dim

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state).argmax().item()

    def store(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train(self):
        if len(self.memory) < self.batch_size: return 0.0

        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next, done = zip(*batch)

        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a).view(-1, 1)
        r = torch.FloatTensor(r).view(-1, 1)
        s_next = torch.FloatTensor(np.array(s_next))
        done = torch.FloatTensor(done).view(-1, 1)

        # 当前 Q 值：通过主网络计算当前状态下采取动作 a 的价值
        q_values = self.q_net(s).gather(1, a)
        avg_q = q_values.mean().item()

        # 计算 Target Q 值
        with torch.no_grad():
            max_next_q = self.target_net(s_next).max(1)[0].view(-1, 1)
            target_q = r + self.gamma * max_next_q * (1 - done)

        criterion = nn.MSELoss()
        loss = criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target()
        return avg_q

    def update_target(self, tau=0.005):
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# 训练主循环
env = gym.make("CartPole-v1")
agent = DQNAgent(4, 2)
epsilon = 1.0

viz = RLVisualizer(title="DQN Training Performance")

for ep in range(1000):
    s, _ = env.reset()
    total_r = 0
    q_sum = 0.0
    q_count = 0
    while True:
        a = agent.choose_action(s, epsilon)
        with torch.no_grad():
            state_t = torch.FloatTensor(s).unsqueeze(0)
            q_sum += agent.q_net(state_t).max(1)[0].item()
            q_count += 1
        s_next, r, terminated, truncated, _ = env.step(a)

        modified_r = r if not terminated else -10

        agent.store(s, a, modified_r, s_next, terminated)

        q_val = agent.train()
        if q_val > 0:
            q_sum += q_val
            q_count += 1

        s = s_next
        total_r += r
        if terminated or truncated:
            break

    avg_q = (q_sum / q_count) if q_count else float("nan")
    viz.add_data(total_r, avg_q=avg_q)
    if ep % 20 == 0:
        viz.draw()

    epsilon = max(0.01, epsilon * 0.99)
    if ep % 50 == 0:
        print(f"Ep: {ep}, Reward: {total_r}, AvgQ: {avg_q:.2f}, Epsilon: {epsilon:.2f}")

viz.save("dqn_results.png")
