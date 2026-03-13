import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

print(f"初始观测状态: {observation}")
print(f"观测空间范围: {env.observation_space}")
print(f"动作空间大小: {env.action_space}")

for _ in range(100):
    # 随机采取动作
    action = env.action_space.sample()

    # 与环境交互
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
