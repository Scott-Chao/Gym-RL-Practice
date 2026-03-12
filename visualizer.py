import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class RLVisualizer:
    def __init__(self, title):
        self.rewards = []
        self.avg_rewards = []
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.title = title

        plt.ion()

    def add_data(self, reward):
        """添加新数据并计算滑动平均"""
        self.rewards.append(reward)
        window = min(len(self.rewards), 50)
        avg = np.mean(self.rewards[-window:])
        self.avg_rewards.append(avg)

    def draw(self):
        """刷新画布"""
        self.ax.clear()
        self.ax.plot(self.rewards, color='blue', alpha=0.3, label='Episode Reward')
        self.ax.plot(self.avg_rewards, color='red', linewidth=2, label='Moving Average (50)')

        self.ax.set_title(self.title)
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)

        plt.pause(0.01)

    def save(self, filename):
        """保存最终图像"""
        plt.ioff()
        filepath = Path("outputs") / filename
        self.fig.savefig(filepath)
        print(f"图表已保存至: {filepath}")
