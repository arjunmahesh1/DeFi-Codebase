# RL environment

import gym
from gym import spaces
import numpy as np
import pandas as pd


class UniswapEnv(gym.Env):
    def __init__(
        self, 
        data, 
        initial_balance=100000,
        gas_fee=10.0,
        slippage_rate=0.0025
    ):
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
        self.liquidity_in_pool = 0.0 
        self.position_value = 0.0
        self.total_rewards = 0.0

        # Slippage & Gas Parameters
        self.gas_fee = gas_fee
        self.slippage_rate = slippage_rate

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
        self.liquidity_in_pool = 0.0
        self.position_value = 0.0
        self.total_rewards = 0.0

        return self._get_observation()
    
    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        obs = row[self.feature_columns].values.astype(np.float32)

        return obs
    
    def step(self, action):
        """
        action = (price_interval_idx, liquidity_move_idx)

        Example meaning of liquidity_move_idx:
          0 -> no change
          1 -> deposit 25% of wallet
          2 -> deposit 50% of wallet
          3 -> withdraw 25% of pool
          4 -> withdraw 50% of pool
        """
        price_interval_idx, liquidity_move_idx = action

        # 1. Realistic deposit/withdraw: adjusts self.balance and self.liquidity_in_pool with slippage/gas cost
        self._apply_liquidity_move(liquidity_move_idx)

        # 2. Calculate how the pool’s value changes from current_step -> next_step
        reward = self._calculate_reward(price_interval_idx)

        self.total_rewards += reward

        # 3. Go to the next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        obs = None if done else self._get_observation()

        # Info dictionary
        info = {
            'balance': self.balance,
            'liquidity_in_pool': self.liquidity_in_pool,
            'position_value': self.position_value,
            'total_rewards': self.total_rewards
        }

        return obs, reward, done, info

    def _apply_liquidity_move(self, liquidity_move_idx):
        """Deposit or withdraw a fraction of available wallet or pool liquidity."""
        # Example mapping:
        moves = {
            0: 0.0,   # no move
            1: 0.25,  # deposit 25% of wallet
            2: 0.50,  # deposit 50% of wallet
            3: -0.25, # withdraw 25% of pool
            4: -0.50, # withdraw 50% of pool
        }
        fraction = moves.get(liquidity_move_idx, 0.0)

        if fraction > 0:  
            deposit_amount = self.balance * fraction
            if deposit_amount > 0:
                # SLIPPAGE AND GAS COSTS
                # TODO : Enhanced slippage modeling, approximate gas costs in Gwei
                slippage_cost = self.slippage_rate * deposit_amount
                total_cost = slippage_cost + self.gas_fee

                if deposit_amount <= total_cost:
                    return

                # Move net deposit into pool
                net_deposit = deposit_amount - total_cost
                self.balance -= deposit_amount
                self.liquidity_in_pool += net_deposit

        elif fraction < 0:
            # withdraw from the pool
            portion = abs(fraction)
            withdraw_amount = self.liquidity_in_pool * portion
            if withdraw_amount > 0:
                slippage_cost = self.slippage_rate * withdraw_amount
                total_cost = slippage_cost + self.gas_fee
                if withdraw_amount <= total_cost:
                    return

                net_withdraw = withdraw_amount - total_cost
                self.liquidity_in_pool -= withdraw_amount
                self.balance += net_withdraw
        else:
            pass
    
    def _calculate_reward(self, price_interval_idx):
        current_row = self.data.iloc[self.current_step]
        next_row = self.data.iloc[min(self.current_step + 1, self.n_steps - 1)]

        current_price = current_row['price']
        next_price = next_row['price']

        price_interval = self.price_intervals[price_interval_idx]
        if self.liquidity_in_pool <= 0:
            self.position_value = 0.0
            return 0.0

        # TRADING FEES: if the price is in the chose interval, earn fees
        # TODO: simplification compared to Uniswap V3
        price_in_interval = (current_price >= price_interval[0]) and (current_price <= price_interval[1])
        if price_in_interval:
            volume_fraction = self._calculate_volume_fraction(price_interval)
            # fees_earned scaled by how big the pool is relative to initial capital (bigger self.liquidity_in_pool = more fees)
            fees_earned = volume_fraction * current_row['volume'] * 0.003 * (self.liquidity_in_pool / self.initial_balance)
        else:
            fees_earned = 0.0

        # PRICE CHANGE
        # TODO: simplification compared to Uniswap V3
        if price_in_interval:
            price_change = (next_price - current_price) / current_price
            position_value_change = self.liquidity_in_pool * price_change
        else:
            position_value_change = 0.0

        
        delta_pool_value = fees_earned + position_value_change
        self.liquidity_in_pool += delta_pool_value

        self.position_value = self.liquidity_in_pool

        return delta_pool_value

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

