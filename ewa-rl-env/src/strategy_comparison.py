import pandas as pd
import numpy as np
from environment import UniswapEnv
from agent import (
    EWAAgent,
    NoRebalanceStrategy,
    FixedIntervalStrategy,
    ResetIntervalStrategy,
    #StableBaselinesAgent
)

def evaluate_strategy(env, agent, num_episodes=1):
    rewards_all = []
    for eps in range(num_episodes):
        obs = env.reset()
        agent.reset()
        ep_reward = 0
        done = False

        while not done:
            action, action_idx = agent.select_action()
            obs, reward, done, info = env.step(action)

            agent.update_probabilities(action_idx, reward)
            ep_reward += reward

        rewards_all.append(ep_reward)

    return rewards_all

def main():
    data = pd.read_csv("../data/train_data.csv")
    
    env_config = {
        "data": data,
        "initial_balance": 100000,
        "gas_fee": 10.0,
        "slippage_rate": 0.0025,
    }
    env = UniswapEnv(**env_config)

    ewa_agent = EWAAgent(env, eta=0.1)
    no_rebalance = NoRebalanceStrategy(env)
    fixed_interval = FixedIntervalStrategy(env, width_pct=0.05)
    reset_interval = ResetIntervalStrategy(env, width_pct=0.05)

    #sb3_agent = StableBaselinesAgent(env, algo="PPO", timesteps=5000)
    #sb3_agent.train()

    strategies = [ewa_agent, no_rebalance, fixed_interval, reset_interval]
    results = {}

    for strat in strategies:
        strat_name = strat.name
        rew = evaluate_strategy(env, strat, num_episodes=1)
        results[strat_name] = rew

    for strat_name, rew in results.items():
        print(f"{strat_name}: Reward = {np.mean(rew):.2f} +/- {np.std(rew):.2f}")

if __name__ == "__main__":
    main()