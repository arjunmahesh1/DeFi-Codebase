# Implement RL agent

import numpy as np

class Agent:
    def __init__(self, env, eta=0.1):
        self.env = env
        self.eta = eta  # EWA learning rate parameter

        self.N_actions = env.N_price_intervals * env.M_liquidity_levels

        self.cumulative_rewards = np.zeros(self.N_actions)
        self.probabilities = np.ones(self.N_actions) / self.N_actions

        self.action_space = []
        for i in range(env.N_price_intervals):
            for j in range(env.M_liquidity_levels):
                self.action_space.append((i,j))

    def select_action(self):
        action_index = np.random.choice(self.N_actions, p=self.probabilities)
        action = self.action_space[action_index]

        return action, action_index
    
    def update_probabilities(self, action_index, reward):
        self.cumulative_rewards[action_index] += reward
        exponentiated_rewards = np.exp(self.eta * self.cumulative_rewards)
        self.probabilities = exponentiated_rewards / np.sum(exponentiated_rewards)

    def reset(self):
        self.cumulative_rewards = np.zeros(self.N_actions)
        self.probabilities = np.ones(self.N_actions) / self.N_actions