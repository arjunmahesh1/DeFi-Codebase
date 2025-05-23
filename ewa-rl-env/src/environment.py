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

        self.token0_in_pool = 0.0
        self.token1_in_pool = 0.0
    
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

        if fraction == 0.0:
            return
        
        # DEPOSIT
        if fraction > 0:  
            usd_deposit = self.balance * fraction
            slippage = self.slippage_rate * usd_deposit
            gas_usd = self.gas_fee                    
            if usd_deposit <= slippage + gas_usd:
                return                                    # nothing left after costs

            net_usd = usd_deposit - (slippage + gas_usd)

            # split 50 / 50 in value into token0 & token1
            p = self.data.loc[self.current_step, "price"]       # token1/token0
            token0_amt = net_usd / (2 * p)                      # value half in token0
            token1_amt = net_usd / 2                            # value half in token1

            self.token0_in_pool += token0_amt
            self.token1_in_pool += token1_amt

            self.balance -= usd_deposit
            self.position_value = self.token0_in_pool * p + self.token1_in_pool

        # WITHDRAW
        else:
            portion = abs(frac)

            # withdraw proportionally from BOTH token balances
            token0_out = self.token0_in_pool * portion
            token1_out = self.token1_in_pool * portion

            p = self.data.loc[self.current_step, "price"]
            usd_withdraw = token0_out * p + token1_out
            slippage = self.slippage_rate * usd_withdraw
            gas_usd = self.gas_fee
            if usd_withdraw <= slippage + gas_usd:
                return

            # burn LP position
            self.token0_in_pool -= token0_out
            self.token1_in_pool -= token1_out

            net_usd = usd_withdraw - (slippage + gas_usd)
            self.balance += net_usd
            self.position_value = self.token0_in_pool * p + self.token1_in_pool
    
    def _calculate_reward(self):
        """
        Reward_t =   (Δ position value)        # mark-to-market P&L
                + fees_earned_usd              # swap fees we earned
                – gas_cost_usd                 # execution cost
        Position value is token0*P + token1.
        """

        # Guard: no liquidity => no reward
        if self.token0_in_pool == 0 and self.token1_in_pool == 0:
            self.position_value = 0.0

            return 0.0

        # Grab current / next market prices and pool liquidity (L)
        cur_row = self.data.iloc[self.current_step]
        next_row = self.data.iloc[min(self.current_step + 1, self.n_steps - 1)]

        P_t = float(cur_row["price"])          # token1 per token0
        P_t1 = float(next_row["price"])
        sqrtP_t = math.sqrt(P_t)

        L_pool   = float(cur_row["liquidity"])      # on-chain total liquidity
        if L_pool <= 0:                                     # fall-back guard
            L_pool = 1e6

        # Only earn when current price is inside our chosen band
        if not (self.lower_price <= P_t <= self.upper_price):
            # still hold inventory but no fees / price impact
            self.position_value = self.token0_in_pool * P_t + self.token1_in_pool
            return 0.0

        # Simulate swap flow hitting *our* share of the pool
        pool_share = np.clip(self.position_value / (P_t * L_pool * 2), 0, 1)
        volume_usd = cur_row["volume"] * pool_share            # crude share
        dx_token0 = volume_usd / P_t                          # token0 in

        fee_tier = 0.0005
        sqrtP_new, dy_token1, fee_token0 = simulate_swap(dx_token0, sqrtP_t, L=L_pool, fee_tier=fee_tier)

        # update internal “pool price” reference 
        self.sqrt_price_pool = sqrtP_new

        # Update our token balances
        self.token0_in_pool += dx_token0 + fee_token0   # got token0 from trader + fees
        self.token1_in_pool -= dy_token1               # paid out token1 to trader

        # Value position *after* swap at next-step market price
        new_value = self.token0_in_pool * P_t1 + self.token1_in_pool    # USD
        fees_usd = fee_token0 * P_t                                    # fees at trade price

        # Gas cost (USD) – keep constant for now, convert Gwei→USD later
        gas_cost_usd = self.gas_fee              # placeholder (already USD)

        # Reward
        reward = (new_value - self.position_value) + fees_usd - gas_cost_usd

        self.position_value = new_value
        self.total_rewards += reward

        return reward

    
    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
            f"Pool: {self.liquidity_in_pool:.2f}, Position: {self.position_value:.2f}, "
            f"Total Rewards: {self.total_rewards:.2f}"
        )

