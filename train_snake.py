import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from snake_env import SnakeEnv

# -----------------------------
# Vectorized environment required by SB3
# -----------------------------
env = DummyVecEnv([lambda: SnakeEnv()])

# -----------------------------
# Parameters and model path
# -----------------------------
MODEL_PATH = "ppo_snake"       # will be saved as ppo_snake.zip
total_steps = 3_000_000        # timesteps to perform in this run (add if you restart)

# -----------------------------
# We define the largest policy network (used only if we create a new model)
# -----------------------------
policy_kwargs = dict(net_arch=[256, 256])

# -----------------------------
# Load existing template if present, otherwise create a new one
# -----------------------------
model_file_exists = os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH + ".zip")

if model_file_exists:
    try:
        print(f"Found existing model '{MODEL_PATH}'. Loading and continuing training...")
        model = PPO.load(MODEL_PATH, env=env)
    except Exception as e:
        print("Error loading existing template, I'll create a new template. Error:", e)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=0.0003,
            clip_range=0.2,
            n_steps=2048,
            batch_size=64,
        )
else:
    print("No templates found. I'm creating a new template from scratch.")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        clip_range=0.2,
        n_steps=2048,
        batch_size=64,
    )

# -----------------------------
# Training (with Ctrl+C management to save progress)
# -----------------------------
try:
    print(f"Start training for {total_steps:,} timesteps...")
    model.learn(total_timesteps=total_steps)
except KeyboardInterrupt:
    print("\nTraining interrupted by user (KeyboardInterrupt). Saving current model...")
    model.save(MODEL_PATH)
    print(f"Model saved as '{MODEL_PATH}'. Repeat the training later to continue.")
    raise

# -----------------------------
# Save trained model
# -----------------------------
model.save(MODEL_PATH)

print(f"Model saved as '{MODEL_PATH}' after {total_steps:,} timesteps!")
