# Implementation of SARSA Control Algorithm using FrozenLake Environment

## Aim

To implement the SARSA (State-Action-Reward-State-Action) Control algorithm for learning the optimal policy in the FrozenLake environment using Reinforcement Learning.

---

## Algorithm

### SARSA Control Algorithm

1. Import the required libraries.
2. Create the FrozenLake environment using OpenAI Gym.
3. Initialize:
   - Learning rate \( \alpha \)
   - Discount factor \( \gamma \)
   - Exploration rate \( \epsilon \)
   - Q-table \( Q(s,a) \)
4. Define the epsilon-greedy policy:
   - Choose a random action with probability \( \epsilon \).
   - Otherwise choose the action with the highest Q-value.
5. For each episode:
   - Reset the environment.
   - Choose an initial action using the epsilon-greedy policy.
   - Repeat until the episode ends:
     - Take the selected action.
     - Observe:
       - Next state
       - Reward
       - Terminal condition
     - Select the next action using the epsilon-greedy policy.
     - Update the Q-value using SARSA update rule:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[
r + \gamma Q(s',a') - Q(s,a)
\right]
\]

     - Update the current state and action.
6. Store rewards for each episode.
7. Reduce epsilon gradually for better exploitation.
8. Display the learned Q-table and reward performance graph.

---

## Program

```
# ============================================================
# SARSA CONTROL ALGORITHM
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 15000

# Q Table
n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))

# Reward Tracking
rewards_per_episode = []

# ============================================================
# Epsilon Greedy Policy
# ============================================================

def choose_action(state):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])

# ============================================================
# SARSA Training
# ============================================================

for episode in range(episodes):

    state, _ = env.reset()

    action = choose_action(state)

    done = False

    total_reward = 0

    while not done:

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        if not done:

            next_action = choose_action(next_state)

            # SARSA Update
            Q[state, action] += alpha * (
                reward
                + gamma * Q[next_state, next_action]
                - Q[state, action]
            )

            state = next_state
            action = next_action

        else:

            # Terminal State Update
            Q[state, action] += alpha * (
                reward - Q[state, action]
            )

        total_reward += reward

    # Store Reward
    rewards_per_episode.append(total_reward)

    # Decay Exploration
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

# ============================================================
# LEARNED POLICY
# ============================================================

print("\nLearned Optimal Policy:\n")

actions = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

for state in range(n_states):

    best_action = np.argmax(Q[state])

    print(f"State {state:2d} --> {actions[best_action]}")

# ============================================================
# STATE ACTION VALUES
# ============================================================

print("\nQ-Table:\n")
print(np.round(Q, 3))

# ============================================================
# REWARD GRAPH
# ============================================================

window = 100

moving_avg = np.convolve(
    rewards_per_episode,
    np.ones(window)/window,
    mode="valid"
)

plt.figure(figsize=(10,5))

plt.plot(moving_avg)

plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Progress")

plt.grid(True)

plt.show()      
```

## Output
<img width="714" height="718" alt="{B9FD4D95-BE29-4824-A4A4-BBAD33B3E1AC}" src="https://github.com/user-attachments/assets/111d0215-783a-4f02-ab2d-433c4a5ec61a" />


## Output Graph

The graph shows the reward obtained per episode during SARSA training. As training progresses, the rewards increase, indicating that the agent learns the optimal policy.
<img width="1118" height="535" alt="{DC44DB26-BBA4-4C98-A45D-43EBA9EA0BB9}" src="https://github.com/user-attachments/assets/a1578d83-9e6f-4146-bb59-6a5db54666bb" />


## Result

Thus, the SARSA Control algorithm was successfully implemented in the FrozenLake environment. The agent learned the optimal action-value function and improved its performance through continuous interaction with the environment using on-policy Temporal Difference learning.
