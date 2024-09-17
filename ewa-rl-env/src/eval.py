# Evaluation script

import pandas as pd
import numpy as np
from environment import UniswapEnv
from agent import Agent

def evaluate_agent(agent, data):
    state = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    results = []

    while not done:
        # Agent selects action based on learned policy
        action, action_index = agent.select_action()

        # Environment returns next state and reward
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        step += 1

        results.append({
            'step': step,
            'action': action,
            'balance': env.balance,
            'position_value': env.position_value,
            'total_rewards': total_reward
        })

        state = next_state

    results_df = pd.DataFrame(results)
    results_df.to_csv('../results/evaluation_results.csv', index=False)
    print(f"Evaluation completed. Total reward: {total_reward:.2f}")
    print("Evaluation results saved to '../results/evaluation_results.csv'")

if __name__ == "__main__":
    data = pd.read_csv('../data/test_data.csv')

    numeric_columns = ['price', 'volume', 'volatility', 'liquidity', 'feesUSD']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=numeric_columns + ['hour', 'day_of_week'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    env = UniswapEnv(data)
    agent = Agent(env)
    agent.probabilities = np.load('../models/agent_probabilities.npy')

    def no_update_probabilities(self, action_index, reward):
        pass  

    agent.update_probabilities = no_update_probabilities.__get__(agent, Agent)

    evaluate_agent(agent, data)
