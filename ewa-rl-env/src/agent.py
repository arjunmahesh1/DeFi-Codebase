# Implement RL agent

import numpy as np
import math
from math import log
try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    PPO = None
    DQN = None
    DummyVecEnv = None
    print("Install Stable Baselines with 'pip install stable-baselines3' to use agent.")

# EWA Agent
class EWAAgent:
    def __init__(self, env, eta=0.1):
        self.env = env
        self.eta = eta  # EWA learning rate parameter

        self.N_actions = env.num_tick_choices * env.num_tick_choices * env.M_liquidity_levels

        self.cumulative_rewards = np.zeros(self.N_actions)
        self.probabilities = np.ones(self.N_actions) / self.N_actions

        self.action_space = []
        for lower_idx in range(env.num_tick_choices):
            for upper_idx in range(env.num_tick_choices):
                for liq_idx in range(env.M_liquidity_levels):
                    self.action_space.append((lower_idx, upper_idx, liq_idx))

        self.name = "EWAAgent_eta{}".format(eta)

    def select_action(self):
        action_index = np.random.choice(self.N_actions, p=self.probabilities)
        action = self.action_space[action_index]

        return action, action_index
    
    def update_probabilities(self, action_index, reward):
        self.cumulative_rewards[action_index] += reward

        scaled_rewards = self.eta * self.cumulative_rewards
        max_scaled = np.max(scaled_rewards)
        shifted = scaled_rewards - max_scaled

        exponentiated = np.exp(shifted)
        denom = np.sum(exponentiated)
        if denom == 0:
            self.probabilities = np.ones_like(exponentiated) / len(exponentiated)
        else:
            self.probabilities = exponentiated / denom

    def reset(self):
        self.cumulative_rewards = np.zeros(self.N_actions)
        self.probabilities = np.ones(self.N_actions) / self.N_actions


# Passive Agent
class NoRebalanceStrategy:
    def __init__(self, env):
        self.env = env
        self.name = "NoRebalance"

        mid_idx = env.num_tick_choices // 2
        liq_move_idx = 0
        
        self.fixed_action = (mid_idx, mid_idx, liq_move_idx)
        self.fixed_action_index = (
            mid_idx * env.num_tick_choices * env.M_liquidity_levels 
            + mid_idx * env.M_liquidity_levels 
            + liq_move_idx
        )

    def select_action(self):
        return self.fixed_action, self.fixed_action_index

    def update_probabilities(self, action_index, reward):
        pass  

    def reset(self):
        pass


# Fixed Interval Strategy Agent
class FixedIntervalStrategy:
    """
    Always place liquidity in a certain fraction of the price around the current price.
    """
    def __init__(self, env, width_pct=0.05):
        self.env = env
        self.width_pct = width_pct
        self.name = "FixedInterval_width{}".format(width_pct)

        self.liquidity_idx = env.M_liquidity_levels - 1  # always full

    def select_action(self):
        current_price = self.env.data.iloc[self.env.current_step]['price']
        current_tick_float = log(current_price, 1.0001)

        lower_price = current_price * (1.0 - self.width_pct)
        upper_price = current_price * (1.0 + self.width_pct)

        lower_tick_float = log(lower_price, 1.0001)
        upper_tick_float = log(upper_price, 1.0001)

        # Approximate index
        def map_tick_to_index(tick_value):
            offset = (tick_value - self.env.min_tick) / self.env.tick_step
            idx = int(max(0, min(self.env.num_tick_choices - 1, offset)))

            return idx

        lower_idx = map_tick_to_index(lower_tick_float)
        upper_idx = map_tick_to_index(upper_tick_float)

        liq_move_idx = 2

        action = (lower_idx, upper_idx, liq_move_idx)

        action_index = (
            lower_idx * self.env.num_tick_choices * self.env.M_liquidity_levels
            + upper_idx * self.env.M_liquidity_levels
            + liq_move_idx
        )

        return action, action_index

    def update_probabilities(self, action_index, reward):
        pass  # No learning (static)

    def reset(self):
        pass


# Reset Agent (Variation of 'Concentrated Liquidity in AMMs' paper)
class ResetIntervalStrategy:
    def __init__(self, env, width_pct=0.05):
        self.env = env
        self.width_pct = width_pct
        self.name = "ResetInterval_width{}".format(width_pct)

        self.current_lower = None
        self.current_upper = None
        self.action_idx = None

    def select_action(self):
        current_price = self.env.data.iloc[self.env.current_step]['price']
        if (self.current_lower is None) or (current_price < self.current_lower) or (current_price > self.current_upper):
            self.current_lower = current_price * (1.0 - self.width_pct)
            self.current_upper = current_price * (1.0 + self.width_pct)

            lower_tick_float = math.log(self.current_lower, 1.0001)
            upper_tick_float = math.log(self.current_upper, 1.0001)

            def map_tick_to_index(tick_val):
                offset = (tick_val - self.env.min_tick) / self.env.tick_step
                idx = int(max(0, min(self.env.num_tick_choices - 1, offset)))
                return idx

            lower_idx = map_tick_to_index(lower_tick_float)
            upper_idx = map_tick_to_index(upper_tick_float)

            liquidity_idx = self.env.M_liquidity_levels - 1

            self.action_idx = (lower_idx, upper_idx, liquidity_idx)

        action = self.action_idx
        (l_i, u_i, liq_i) = action
        a_idx = (
            l_i * self.env.num_tick_choices * self.env.M_liquidity_levels
            + u_i * self.env.M_liquidity_levels
            + liq_i
        )
        return action, a_idx

    def update_probabilities(self, action_index, reward):
        pass

    def reset(self):
        self.current_lower = None
        self.current_upper = None
        self.action_idx = None


# TODO: Stable Baselines Agent
# class StableBaselinesAgent:
#     def __init__(self, env, algo="PPO", timesteps=5000):
#         self.env = env
#         self.name = f"StableBaselines_{algo}"
#         self.algo = algo
#         self.timesteps = timesteps
#         self.model = None

#     def train(self):
#         # Convert to a Vectorized environment for SB3
#         if DummyVecEnv is None:
#             raise ImportError("Please install stable-baselines3 to use this agent.")

#         def make_env():
#             # Re-init a fresh environment for stable-baselines (careful with data usage)
#             return self.env
        
#         vec_env = DummyVecEnv([make_env])
        
#         if self.algo == "PPO":
#             self.model = PPO("MlpPolicy", vec_env, verbose=0)
#         elif self.algo == "DQN":
#             self.model = DQN("MlpPolicy", vec_env, verbose=0)
#         else:
#             raise ValueError("Unsupported algo for demonstration.")
        
#         # Train
#         self.model.learn(total_timesteps=self.timesteps)

#     def select_action(self):
#         obs = self.env._get_observation()  # or however we retrieve the latest obs
#         # The env step method expects (discrete) action in the form (i, j). 
#         # But SB3 typically returns a single integer if MultiDiscrete, or a float if Box.
#         # We'll get the raw action from the model:
#         action, _states = self.model.predict(obs, deterministic=True)

#         # action might be an array of shape (2, ) if the env has MultiDiscrete(10,5).
#         # So let's parse it properly:
#         if isinstance(action, np.ndarray) and len(action) == 2:
#             price_idx, liq_idx = action
#         else:
#             # Fallback in case DQN returns an integer, or something else. 
#             # (SB3 does handle MultiDiscrete, but be sure to set it up properly.)
#             price_idx = int(action // self.env.M_liquidity_levels)
#             liq_idx = int(action % self.env.M_liquidity_levels)

#         action_idx = price_idx * self.env.M_liquidity_levels + liq_idx
#         return (price_idx, liq_idx), action_idx
    
#     def update_probabilities(self, action_index, reward):
#         # Not applicable for SB3. We rely on internal RL updates.
#         pass

#     def reset(self):
#         # Possibly re-train or just do nothing. 
#         # Usually you call train() once before evaluating.
#         pass