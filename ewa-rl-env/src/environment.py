# RL environment

import gym
from gym import spaces
import numpy as np
import pandas as pd
import math

def tick_to_price(tick):
    return 1.0001 ** tick

def price_to_tick(price):
    return int(math.log(price, 1.0001))

def simulate_swap(delta_x, sqrt_price, L, fee_tier=0.0005):
    """
    Bonding Curve: p' = (sqrt(p) + delta_x / L)^2
    return final price p', average execution price, tokens_in, tokens_out, etc.
    Exact-input token0 swap.
    Returns: sqrt_p_new, delta_y, fee_x
    """
    # fees skimmed off upfront
    fee_x = delta_x * fee_tier
    delta_x_post = delta_x - fee_x

    # Calculate the new price
    numerator = L * sqrt_price
    denominator = L + delta_x_post * sqrt_price
    sqrt_price_new = numerator / denominator

    # token1 out (delta_y > 0 means pool pays out token1)
    delta_y = L * (sqrt_price - sqrt_price_new)

    return sqrt_price_new, delta_y, fee_x

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
        self.M_liquidity_levels = 5

        # Define liquidity levels
        self.liquidity_levels = [0.2, 0.4, 0.6, 0.8, 1.0]   # Fractions of initial balance

        # Min/Max Ticks from price data
        price_min = self.data['price'].min()
        price_max = self.data['price'].max()

        self.min_tick = price_to_tick(price_min)
        self.max_tick = price_to_tick(price_max)

        # Discrete Tick Choices (adjustable increments)
        self.tick_step = 60
        self.num_tick_choices = max(1,(self.max_tick - self.min_tick) // self.tick_step)

        # Action space: MultiDiscrete space for price interval and liquidity level
        self.action_space = spaces.MultiDiscrete([
            self.num_tick_choices,
            self.num_tick_choices,
            self.M_liquidity_levels
        ])

        # Observation space
        self.feature_columns = ['price', 'volume', 'volatility', 'liquidity', 'feesUSD', 'hour', 'day_of_week']
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.feature_columns),), 
            dtype=np.float32
        )

        # Financial States Initialization
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.liquidity_in_pool = 0.0 
        self.position_value = 0.0
        self.total_rewards = 0.0

        self.sqrt_price_pool = math.sqrt(self.data['price'].iloc[0])

        # Slippage & Gas Parameters
        self.gas_fee = gas_fee
        self.slippage_rate = slippage_rate
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.liquidity_in_pool = 0.0
        self.position_value = 0.0
        self.total_rewards = 0.0
        self.sqrt_price_pool = math.sqrt(self.data['price'].iloc[0])

        return self._get_observation()
    
    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        obs = row[self.feature_columns].values.astype(np.float32)

        return obs
    
    def step(self, action):
        """
        action = (lower_idx, upper_idx, liquidity_move_idx)
            - lower_idx, upper_idx are discrete indices > become ticks

        Example meaning of liquidity_move_idx:
          0 -> no change
          1 -> deposit 25% of wallet
          2 -> deposit 50% of wallet
          3 -> withdraw 25% of pool
          4 -> withdraw 50% of pool
        """

        # 1. Convert lower/upper tick indices into ticks
        lower_idx, upper_idx, liquidity_move_idx = action

        lower_tick = self.min_tick + (lower_idx * self.tick_step)
        upper_tick = self.min_tick + (upper_idx * self.tick_step)

        # 2. Convert ticks into prices
        self.lower_price = tick_to_price(lower_tick)
        self.upper_price = tick_to_price(upper_tick)

        # 3. Realistic deposit/withdraw: adjusts self.balance and self.liquidity_in_pool with slippage/gas cost
        self._apply_liquidity_move(liquidity_move_idx)

        # 4. Calculate how the pool’s value changes from last step -> current step
        reward = self._calculate_reward()
        self.total_rewards += reward

        # 5. Go to the next step/advance environment
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
                self.liquidity_in_pool = min(self.liquidity_in_pool, 10 * self.initial_balance)  # Cap pool size to 10x initial balance

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
    
    def _calculate_reward(self):
        """Compare the current price to the chosen (self.lower_price, self.upper_price).
        If in range, earn fees + price changes. Otherwise, 0 fees or no price gain."""

        if self.liquidity_in_pool <= 0:
            self.position_value = 0.0
            
            return 0.0

        current_row = self.data.iloc[self.current_step]
        next_row = self.data.iloc[min(self.current_step + 1, self.n_steps - 1)]

        current_price_market = current_row['price']
        next_price_market = next_row['price']

        in_range = (current_price_market >= self.lower_price) and (current_price_market <= self.upper_price) 
        if not in_range:
            self.position_value = self.liquidity_in_pool

            return 0.0
        
        # Simulation of pool reaction to swap volume: assumption that a fraction of volume is trading with pool
        total_liq  = current_row["liquidity"] or 1e18 
        pool_share = np.clip(self.liquidity_in_pool / total_liq, 0, 1)
        volume_traded = current_row['volume'] * pool_share      # Approximation
        token0_in = volume_traded / current_price_market        # Approximation

        p_before = self.sqrt_price_pool
        L = 1e6         # Liquidity Param
        p_final, avg_exec = simulate_swap(token0_in, p_before, L)
        self.sqrt_price_pool = p_final

        # TRADING FEES: if the price is in the chose interval, earn fees
        # TODO: simplification compared to Uniswap V3
        fee_tier = 0.0005          # 0.05 % ≈ popular ETH/USDC pool
        fees_earned = fee_tier * volume_traded

        # IMPERMANENT LOSS FROM PRICE MOVEMENT
        # TODO: simplification compared to Uniswap V3
        price_change_fraction = (next_price_market - current_price_market) / current_price_market
        position_value_change = self.liquidity_in_pool * price_change_fraction

        
        delta_pool_value = fees_earned + position_value_change
        self.liquidity_in_pool = np.clip(
            self.liquidity_in_pool + delta_pool_value,
            0,
            10 * self.initial_balance
        )
        self.position_value = self.liquidity_in_pool

        return delta_pool_value
    
    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
            f"Pool: {self.liquidity_in_pool:.2f}, Position: {self.position_value:.2f}, "
            f"Total Rewards: {self.total_rewards:.2f}"
        )

