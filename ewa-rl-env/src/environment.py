# RL environment

import gym
from gym import spaces
import numpy as np
import pandas as pd

class UniswapEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(UniswapEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)
        self.current_step = 0

        # Action and observation space
        self.N_price_intervals = 10
        self.M_liquidity_levels = 5

        # Define price intervals and liquidity levels
        self.price_intervals = self._define_price_intervals()
        self.liquidity_levels = [0.2, 0.4, 0.6, 0.8, 1.0]   # Fractions of initial balance

        # Action space: MultiDiscrete space for price interval and liquidity level
        self.action_space = spaces.MultiDiscrete([self.N_price_intervals, self.M_liquidity_levels])

        self.feature_columns = ['price', 'volume', 'volatility', 'liquidity', 'feesUSD', 'hour', 'day_of_week']
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_columns),), dtype=np.float32)

        # Financial States Initialization
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position_value = 0.0
        self.total_rewards = 0.0

    def _define_price_intervals(self):
        '''
        Price intervals based on historical price data
        '''
        price_min = self.data['price'].min()
        price_max = self.data['price'].max()

        price_intervals = np.linspace(price_min, price_max, self.N_price_intervals + 1)
        intervals = []
        for i in range(self.N_price_intervals):
            intervals.append((price_intervals[i], price_intervals[i + 1]))

        return intervals
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_value = 0.0
        self.total_rewards = 0.0

        return self._get_observation()
    
    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        obs = row[self.feature_columns].values.astype(np.float32)

        return obs
    
    def step(self, action):
        price_interval_idx, liquidity_level_idx = action

        price_interval = self.price_intervals[price_interval_idx]
        liquidity_fraction = self.liquidity_levels[liquidity_level_idx]

        liquidity_amount = self.balance * liquidity_fraction

        reward = self._calculate_reward(price_interval, liquidity_amount)

        self.balance -= liquidity_amount
        self.balance += reward
        self.total_rewards += reward

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        if not done:
            obs = self._get_observation()
        else:
            obs = None

        info = {
            'balance': self.balance,
            'position_value': self.position_value,
            'total_rewards': self.total_rewards,
        }

        return obs, reward, done, info
    
    def _calculate_reward(self, price_interval, liquidity_amount):
        current_row = self.data.iloc[self.current_step]
        next_row = self.data.iloc[min(self.current_step + 1, self.n_steps - 1)]

        current_price = current_row['price']
        next_price = next_row['price']

        price_in_interval = (current_price >= price_interval[0]) and (current_price <= price_interval[1])

        if price_in_interval:
            volume_fraction = self._calculate_volume_fraction(price_interval)
            fees_earned = volume_fraction * current_row['volume'] * 0.003 * (liquidity_amount / self.initial_balance)
        else:
            fees_earned = 0.0

        if price_in_interval:
            price_change = next_price - current_price
            position_value_change = liquidity_amount * (price_change / current_price)
        else:
            position_value_change = 0.0

        reward = fees_earned + position_value_change

        self.position_value = liquidity_amount + position_value_change

        return reward

    def _calculate_volume_fraction(self, price_interval):
        total_price_range = self.data['price'].max() - self.data['price'].min()
        interval_range = price_interval[1] - price_interval[0]
        volume_fraction = interval_range / total_price_range
        
        return volume_fraction
    
    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
            f"Position Value: {self.position_value:.2f}, Total Rewards: {self.total_rewards:.2f}"
        )

