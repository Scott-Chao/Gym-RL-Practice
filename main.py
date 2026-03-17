import gymnasium as gym
import torch
import numpy as np
import argparse
from pathlib import Path
from visualizer import RLVisualizer

from dqn_agent import DQNAgent, NStepBuffer
from ppo_agent import PPOAgent

torch.set_num_threads(4)

MODEL_DIR = "models"
MODEL_PATH_DQN = Path(MODEL_DIR) / "best_lunar_lander_dqn.pth"
MODEL_PATH_PPO = Path(MODEL_DIR) / "best_lunar_lander_ppo.pth"

if not Path(MODEL_DIR).exists():
    Path(MODEL_DIR).mkdir(parents=True)

n_step_n = 3
n_step_gamma = 0.99
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.99


def run_train_dqn(env):
    global epsilon
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
            agent.save(MODEL_PATH_DQN)

        if ep % 50 == 0:
            viz.draw()
            print(f"Ep: {ep}, Reward: {total_r:.2f}, AvgQ: {avg_q:.2f}, Epsilon: {epsilon:.2f}")

    viz.save("dqn_results.png")


def run_test_dqn(env):
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, n_step_n)

    if MODEL_PATH_DQN.exists():
        print(f"Loading DQN model from {MODEL_PATH_DQN}...")
        agent.load(MODEL_PATH_DQN)
    else:
        print("No saved DQN model found! Please train first.")
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


def run_train_ppo(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)

    viz = RLVisualizer(title="PPO Training Performance")

    best_reward = -np.inf
    update_timestep = 2000
    timestep = 0

    for ep in range(2000):
        s, _ = env.reset()
        total_r = 0

        while True:
            a, a_logp = agent.choose_action(s)

            a_clipped = np.clip(a, -1.0, 1.0)

            s_next, r, terminated, truncated, _ = env.step(a_clipped)

            agent.store((s, a, a_logp, r, terminated))

            s = s_next
            total_r += r
            timestep += 1

            if terminated or truncated:
                break

        viz.add_data(total_r)

        if timestep >= update_timestep:
            agent.update()
            timestep = 0

        if total_r >= best_reward:
            best_reward = total_r
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                },
                MODEL_PATH_PPO,
            )

        if ep % 50 == 0:
            viz.draw()
            print(f"Ep: {ep}, Reward: {total_r:.2f}")

    viz.save("ppo_results.png")


def run_test_ppo(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)

    if MODEL_PATH_PPO.exists():
        print(f"Loading PPO model from {MODEL_PATH_PPO}...")
        checkpoint = torch.load(MODEL_PATH_PPO)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
    else:
        print("No saved PPO model found! Please train first.")
        return

    for ep in range(10):
        s, _ = env.reset()
        total_r = 0

        while True:
            with torch.no_grad():
                s_tensor = torch.FloatTensor(s).unsqueeze(0)
                mu, _ = agent.actor(s_tensor)
                a = mu.numpy()[0]

            a_clipped = np.clip(a, -1.0, 1.0)
            s_next, r, terminated, truncated, _ = env.step(a_clipped)
            s = s_next
            total_r += r
            if terminated or truncated:
                break

        print(f"Test Episode: {ep}, Reward: {total_r:.2f}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the model")
    group.add_argument("--test", action="store_true", help="Test the saved model")

    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ppo"],
        default="dqn",
        help="Algorithm to use (dqn or ppo)",
    )
    args = parser.parse_args()

    if args.algo == "dqn":
        env_kwargs = {"id": "LunarLander-v3"}
    else:
        env_kwargs = {"id": "LunarLander-v3", "continuous": True}

    if args.test:
        env_kwargs["render_mode"] = "human"

    env = gym.make(**env_kwargs)

    if args.train:
        if args.algo == "dqn":
            run_train_dqn(env)
        else:
            run_train_ppo(env)
    else:
        if args.algo == "dqn":
            run_test_dqn(env)
        else:
            run_test_ppo(env)

    env.close()


if __name__ == "__main__":
    main()
