# Snake RL --- Reinforcement Learning Project 🐍

## 📘 Theoretical Description

Reinforcement learning is a type of machine learning in which an agent
learns to make decisions by interacting with an environment, receiving
rewards or penalties based on the actions it takes, with the goal of
maximizing the total reward over time. Unlike supervised learning, no
correct answers are provided to imitate — only a feedback signal
(reward) indicating how good or bad an action was. Through trial and
error, exploration, and progressive updates to its strategy (policy),
the agent builds increasingly effective behavior. Formally, the problem
is often modeled as a Markov Decision Process: at each state the agent
chooses an action, the environment returns a new state and a reward, and
the cycle repeats until the agent learns a policy that maximizes the
expected cumulative reward.

------------------------------------------------------------------------

## 🎯 Project Goal

Train a PPO (Proximal Policy Optimization) agent capable of playing
Snake using:

-   A custom Gymnasium-compatible environment
-   Stable-Baselines3
-   Configurable reward shaping
-   Cumulative training (resume training)
-   Rendering with pygame

------------------------------------------------------------------------

## 📂 Project Structure

    SnakeAI/
    ├─ snake_env.py        # Custom Gymnasium environment
    ├─ train_snake.py      # PPO training script
    ├─ play_snake.py       # AI gameplay visualization
    └─ ppo_snake.zip       # Saved model after training

------------------------------------------------------------------------

## ⚙️ Requirements

-   Python 3.11 recommended
-   Dependencies:

``` bash
pip install gymnasium stable-baselines3 torch pygame numpy
```

------------------------------------------------------------------------

## 🚀 How to Use

### 1️⃣ Training

``` bash
py -3.11 train_snake.py
```

-   If `ppo_snake.zip` exists → continues training
-   If it does not exist → creates a new model

### 2️⃣ Watch the AI play

``` bash
py -3.11 play_snake.py
```

### 3️⃣ Numerical evaluation (optional)

``` bash
py -3.11 eval_snake.py
```

------------------------------------------------------------------------

## 🧠 Reward Shaping

Configurable in `snake_env.py`:

``` python
REWARD_EAT = 5.0
REWARD_APPROACH = 0.20
PENALTY_AWAY = -0.5
STEP_PENALTY = -0.03
REPEAT_PENALTY = -0.50
MAX_RECENT_HEADS = 8
```

Reward components:

-   + High reward for eating food

-   + Reward proportional to approaching food (Manhattan distance)

-   − Penalty for moving away from food

-   − Penalty for each step (anti-loop)

-   − Penalty for recent oscillations

-   − Collision penalty (-1)

Reward shaping guides the agent toward intelligent behavior while
avoiding exploits or infinite loops.

------------------------------------------------------------------------

## 🏗 Architecture

-   Algorithm: PPO
-   Policy: MLP (2 layers of 256 neurons)
-   Environment: 10x10 grid
-   Observation: normalized 10x10 matrix
-   Cumulative training supported

------------------------------------------------------------------------

## 📌 Final Note

This project is designed as a complete educational exercise in
Reinforcement Learning: from building the environment to visualizing the
trained agent.
