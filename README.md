# RL Project Documentation

## Data Preparation
- Merged sensor, system, action, and reward files.
- Encoded all categorical input features for robust neural modeling.

## RL Environment and Policy
- Used transitions (state, action, reward, next_state) for policy learning.
- Implemented simple neural network for demonstration; easily extensible to DQN/A2C/BCQ.

## Evaluation and Analysis
- Compared agent's actions to ground-truth logs in 'action_logs_test.csv'.
- Visualized action distributions, reward curves.