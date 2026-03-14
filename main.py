import gymnasium as gym
import torch
import numpy as np
import argparse
from pathlib import Path
from visualizer import RLVisualizer

from dqn_agent import DQNAgent, NStepBuffer

torch.set_num_threads(4)

MODEL_DIR = "models"
MODEL_PATH = Path(MODEL_DIR) / "best_lunar_lander.pth"

if not MODEL_PATH.parent.exists():
    MODEL_PATH.parent.mkdir(parents=True)

n_step_n = 3
n_step_gamma = 0.99
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.99

def run_train():
    env = gym.make("LunarLander-v3")

    n_buffer = NStepBuffer(n_step_n, n_step_gamma)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, n_step_n)

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

            n_step_data = n_buffer.add(s, a, r, s_next, terminated)
            if n_step_data:
                agent.store(*n_step_data)

            q_val = agent.train()

            if q_val is not None:
                q_sum += q_val
                q_count += 1

            s = s_next
            total_r += r

            if terminated:
                if len(n_buffer.buffer) == n_step_n:
                    n_buffer.buffer.popleft()
                while len(n_buffer.buffer) > 0:
                    n_step_data = n_buffer.get_n_step_info()
                    agent.store(*n_step_data)
                    n_buffer.buffer.popleft()
                break
            if truncated:
                n_buffer.buffer.clear()
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
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, n_step_n)

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


if __name__ == "__main__":
    main()
