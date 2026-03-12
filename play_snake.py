from stable_baselines3 import PPO
from snake_env import SnakeEnv
import pygame
import time

env = SnakeEnv(render_mode="human")
model = PPO.load("ppo_snake")

obs, _ = env.reset()
done = False

while True:
    for event in pygame.event.get():  # manages window closing
        if event.type == pygame.QUIT:
            env.close()
            exit()

    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    env.render()
    time.sleep(0.1)

    done = terminated or truncated
    if done:

        obs, _ = env.reset()
