# Training loop for RL Agent

import numpy as np
import pandas as pd
from environment import UniswapEnv
from agent import Agent

def train_agent(data, num_episodes=100, eta=0.1):
    env = UniswapEnv(data)
    agent = Agent(env, eta=eta)
    
    results = []

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}/{num_episodes}")
        state = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, action_index = agent.select_action()

            next_state, reward, done, info = env.step(action)

            agent.update_probabilities(action_index, reward)

            total_reward += reward
            step += 1

            results.append({
                'episode': episode + 1,
                'step': step,
                'action': action,
                'balance': env.balance,
                'position_value': env.position_value,
                'total_rewards': env.total_rewards
            })

            env.render()

            state = next_state

        print(f"Episode {episode + 1} finished. Total reward: {total_reward:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('../results/training_results.csv', index=False)
    print("Training results saved to 'training_results.csv'")
    
    return agent

if __name__ == "__main__":
    data = pd.read_csv('../data/train_data.csv')

    numeric_columns = ['price', 'volume', 'volatility', 'liquidity', 'feesUSD']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(subset=numeric_columns + ['hour', 'day_of_week'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    print("Data columns:", data.columns.tolist())
    print("First few rows:")
    print(data.head())

    trained_agent = train_agent(data, num_episodes=100, eta=0.1)
    np.save('../models/agent_probabilities.npy', trained_agent.probabilities)
    print("Agent's action probabilities saved to '../models/agent_probabilities.npy'")


