import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        x = self.fc(x)
        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": lr},
                {"params": self.critic.parameters(), "lr": lr},
            ]
        )

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.memory = []

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)

        return action.numpy()[0], action_log_prob.item()

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        states = torch.FloatTensor(np.array([t[0] for t in self.memory]))
        actions = torch.FloatTensor(np.array([t[1] for t in self.memory]))
        old_log_probs = torch.FloatTensor(
            np.array([t[2] for t in self.memory])
        ).view(-1, 1)
        rewards = [t[3] for t in self.memory]
        is_terminals = [t[4] for t in self.memory]

        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns).view(-1, 1)

        with torch.no_grad():
            old_state_values = self.critic(states)
            advantages = returns - old_state_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            mu, std = self.actor(states)
            dist = Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            state_values = self.critic(states)

            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()

            loss_critic = nn.MSELoss()(state_values, returns)

            loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []
