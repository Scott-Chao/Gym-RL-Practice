import gymnasium as gym
import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, bins=(1, 1, 6, 12)):
        self.env = env
        self.bins = bins
        self.action_size = env.action_space.n
        
        # 定义每个维度的观测边界
        self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        self.state_bounds[1] = [-3.0, 3.0]  # 修正速度边界
        self.state_bounds[3] = [-3.5, 3.5]  # 修正角速度边界

        # 初始化 Q-Table
        self.q_table = np.zeros(self.bins + (self.action_size,))
        
        # 超参数
        self.alpha = 0.1    # 学习率
        self.gamma = 0.95   # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995

    def discretize_state(self, obs):
        """将连续观测值映射为离散索引"""
        discretized = []
        for i in range(len(obs)):
            # 线性插值：将 obs[i] 映射到 [0, bins[i]-1] 之间
            low, high = self.state_bounds[i]
            # 剪裁值以防越界
            val = np.clip(obs[i], low, high)
            # 计算索引：(val - low) / (high - low) * (bins - 1)
            index = int(np.round((self.bins[i] - 1) * (val - low) / (high - low)))
            discretized.append(index)
        return tuple(discretized)

    def choose_action(self, state):
        """epsilon-greedy 策略"""
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Q-Learning 更新公式"""
        old_value = self.q_table[state][action]
        # TD Target: r + gamma * max(Q(s'))
        next_max = np.max(self.q_table[next_state])
        
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

env = gym.make("CartPole-v1")
agent = QLearningAgent(env)

for episode in range(2000):
    obs, _ = env.reset()
    state = agent.discretize_state(obs)
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = agent.discretize_state(next_obs)
        
        done = terminated or truncated
        
        # 如果杆子倒了，给予惩罚以加速收敛
        actual_reward = reward if not terminated else -100
        
        agent.learn(state, action, actual_reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
            
    agent.epsilon = max(0.01, agent.epsilon * agent.epsilon_decay)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

env.close()
