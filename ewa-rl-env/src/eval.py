import pandas as pd, numpy as np, json, pickle
from environment import UniswapEnv
from agent import EWAAgent
from pathlib import Path

DATA_DIR  = Path("../data")
MODEL_DIR = Path("../models")

# load test data
data = pd.read_csv(DATA_DIR/"test_data.csv")

# recreate training action space & probs
action_space = np.load(MODEL_DIR/"agent_action_space.npy", allow_pickle=True)
prob         = np.load(MODEL_DIR/"agent_probabilities.npy")

# build env – feed test_data
env = UniswapEnv(data)

# stitch agent
agent = EWAAgent(env, eta=0.1)
agent.action_space = list(action_space)
agent.N_actions    = len(agent.action_space)
agent.probabilities = prob
assert len(prob) == agent.N_actions

# roll evaluation
state, done, total = env.reset(), False, 0
while not done:
    act, idx = agent.select_action()
    state, rew, done, info = env.step(act)
    total += rew
print(f"Total reward (test set): {total:.2f}")
