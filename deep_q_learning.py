import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from pathlib import Path
from visualizer import RLVisualizer

torch.set_num_threads(4)

MODEL_DIR = "models"
MODEL_PATH = Path(MODEL_DIR) / "best_lunar_lander.pth"
if not MODEL_PATH.parent.exists():
    MODEL_PATH.parent.mkdir(parents=True)

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

class DuelingQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNet, self).__init__()

        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        # 状态价值流 (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 动作优势流 (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)

        v = self.value_stream(features)      # Shape: [batch, 1]
        a = self.advantage_stream(features)  # Shape: [batch, action_dim]

        # Q = V + (A - A.mean)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

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

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[idx]),
            torch.LongTensor(self.actions[idx]),
            torch.FloatTensor(self.rewards[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(self.dones[idx])
        )

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = DuelingQNet(state_dim, action_dim)
        self.target_net = DuelingQNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(100000, state_dim)
        self.gamma = 0.99
        self.batch_size = 128
        self.action_dim = action_dim

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

        s, a, r, s_next, done = self.memory.sample(self.batch_size)

        # 预测当前 Q 值
        q_values = self.q_net(s).gather(1, a)
        avg_q = q_values.mean().item()

        # 计算 Target Q 值
        with torch.no_grad():
            best_actions = self.q_net(s_next).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(s_next).gather(1, best_actions)
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

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--test", action="store_true", help="Test the saved model")
    args = parser.parse_args()

    if args.train:
        run_train()
    else:
        run_test()

def run_train():
    env = gym.make("LunarLander-v3")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.99

    viz = RLVisualizer(title="DQN Training Performance")

    best_reward = -np.inf

    for ep in range(1000):
        s, _ = env.reset()
        total_r = 0
        q_sum = 0.0
        q_count = 0

        while True:
            a = agent.choose_action(s, epsilon)
            s_next, r, terminated, truncated, _ = env.step(a)

            agent.store(s, a, r, s_next, terminated)

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
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if total_r >= best_reward:
            best_reward = total_r
            agent.save(MODEL_PATH)

        if ep % 50 == 0:
            viz.draw()
            print(f"Ep: {ep}, Reward: {total_r:.2f}, AvgQ: {avg_q:.2f}, Epsilon: {epsilon:.2f}")

    viz.save("dqn_results.png")
    env.close()

def run_test():
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH}...")
        agent.load(MODEL_PATH)
    else:
        print("No saved model found! Please train first.")
        return

    for ep in range(10):
        s, _ = env.reset()
        total_r = 0

        while True:
            a = agent.choose_action(s, epsilon=0)
            s_next, r, terminated, truncated, _ = env.step(a)
            s = s_next
            total_r += r
            if terminated or truncated:
                break

        print(f"Test Episode: {ep}, Reward: {total_r:.2f}")

    env.close()

if __name__ == "__main__":
    main()
