import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class RLVisualizer:
    def __init__(self, title):
        self.rewards = []
        self.avg_rewards = []
        self.q_values = []
        self.avg_q_values = []
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax_q = None
        self.title = title

        plt.ion()

    def add_data(self, reward, avg_q=None):
        """添加新数据并计算滑动平均"""
        self.rewards.append(reward)
        window = min(len(self.rewards), 50)
        self.avg_rewards.append(float(np.mean(self.rewards[-window:])))

        self.q_values.append(np.nan if avg_q is None else float(avg_q))

        recent_q = np.array(self.q_values[-window:], dtype=float)
        self.avg_q_values.append(float(np.nanmean(recent_q)) if np.isfinite(recent_q).any() else np.nan)

    def draw(self):
        """刷新画布"""
        self.ax.clear()
        self.ax.scatter(
            range(len(self.rewards)),
            self.rewards,
            color="blue",
            alpha=0.3,
            s=12,
            label="Episode Reward",
        )
        self.ax.plot(self.avg_rewards, color="red", linewidth=2, label="Reward MA (50)")

        self.ax.set_title(self.title)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, linestyle="--", alpha=0.6)

        handles, labels = self.ax.get_legend_handles_labels()

        if np.isfinite(np.array(self.q_values, dtype=float)).any():
            if self.ax_q is None:
                self.ax_q = self.ax.twinx()
            self.ax_q.clear()

            self.ax_q.plot(self.avg_q_values, color="green", linewidth=2, label="Avg Q MA (50)")
            self.ax_q.set_ylabel("Q")

            h2, l2 = self.ax_q.get_legend_handles_labels()
            handles += h2
            labels += l2

        self.ax.legend(handles, labels, loc="best")
        plt.pause(0.01)

    def save(self, filename):
        """保存最终图像"""
        plt.ioff()
        filepath = Path("outputs") / filename
        self.fig.savefig(filepath)
        print(f"图表已保存至: {filepath}")
