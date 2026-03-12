import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from collections import deque

GRID_SIZE = 10
CELL_SIZE = 30

# --- Reward tuning (tweak these values ​​to experiment) ---
REWARD_EAT = 5.0            # reward for eating food
REWARD_APPROACH = 0.20      # distance reduction multiplier (manhattan)
PENALTY_AWAY = -0.5        # penalty for walking away from food
STEP_PENALTY = -0.03        # penalty for each step (avoids infinite looping)
REPEAT_PENALTY = -0.50      # penalty for repeating recent head position
MAX_RECENT_HEADS = 8        # how many recent heads to keep in memory

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, max_steps=None):
        super(SnakeEnv, self).__init__()

        self.render_mode = render_mode

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation = 10x10 grid, values ​​in [0,1]
        # 0.0 = empty, 0.5 = body, 0.75 = head, 1.0 = food
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32
        )

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
            )
            pygame.display.set_caption("Snake RL")

        # max_steps per episode (avoids infinite episodes)
        self.max_steps = max_steps if max_steps is not None else GRID_SIZE * GRID_SIZE * 6

        # recent head history to detect oscillations
        self._recent_heads = deque(maxlen=MAX_RECENT_HEADS)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        # initial downward direction (dx,dy)
        self.direction = (0, 1)
        self.food = self._spawn_food()
        self.done = False
        self.steps = 0
        self._recent_heads.clear()
        self._recent_heads.append(self.snake[0])
        self.score = 0  # number of foods eaten

        return self._get_obs(), {}

    def _spawn_food(self):
        # generates valid position not occupied by the snake
        # if the grid is full, return None (win)
        free_cells = (GRID_SIZE * GRID_SIZE) - len(self.snake)
        if free_cells <= 0:
            return None

        while True:
            pos = (random.randint(0, GRID_SIZE - 1),
                   random.randint(0, GRID_SIZE - 1))
            if pos not in self.snake:
                return pos

    def _get_obs(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # body
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = 0.75  # head
            else:
                grid[y][x] = 0.5   # body

        # food (if it exists)
        if self.food is not None:
            fx, fy = self.food
            grid[fy][fx] = 1.0

        return grid

    def step(self, action):
        # prevent immediate reversal (avoid 180°)
        opposites = {
            (0, -1): (0, 1),
            (0, 1): (0, -1),
            (-1, 0): (1, 0),
            (1, 0): (-1, 0),
        }

        # action map -> proposed direction
        proposed = self.direction
        if action == 0:
            proposed = (0, -1)
        elif action == 1:
            proposed = (0, 1)
        elif action == 2:
            proposed = (-1, 0)
        elif action == 3:
            proposed = (1, 0)

        # ignore reversal if it is exactly the opposite of the current direction
        if opposites.get(self.direction) != proposed:
            self.direction = proposed
        # otherwise keep going (no immediate suicide)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0
        terminated = False
        truncated = False

        self.steps += 1

        # collision control (wall or self)
        if (
            new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in self.snake
        ):
            reward = -1.0
            terminated = True
            info = {"score": self.score, "steps": self.steps}
            return self._get_obs(), reward, terminated, truncated, info

        # Win situation: if there is no more room for food (snake fills the grid)
        if self.food is None:
            # game won
            reward = REWARD_EAT
            terminated = True
            info = {"score": self.score, "steps": self.steps}
            return self._get_obs(), reward, terminated, truncated, info

        # Manhattan distance before and after (reward shaping)
        old_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        proposed_new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        # update snake
        self.snake.insert(0, new_head)

        ate = False
        if new_head == self.food:
            # ate
            reward += REWARD_EAT
            self.score += 1
            ate = True
            # spawn new food
            self.food = self._spawn_food()
        else:
            # uneaten: remove tail
            self.snake.pop()

        # reward proportional to the approach (positive if the distance decreases)
        dist_delta = old_distance - proposed_new_distance
        if dist_delta > 0:
            reward += dist_delta * REWARD_APPROACH
        elif dist_delta < 0:
            reward += dist_delta * (-PENALTY_AWAY)  # dist_delta is negative; applies small penalty

        # Passive step penalty to discourage infinite loops or unnecessary turns
        reward += STEP_PENALTY

        # stronger penalty if recent head position (oscillations) is repeated
        if new_head in list(self._recent_heads):
            # if he hasn't just eaten (if he has eaten it is justified to reposition himself)
            if not ate:
                reward += REPEAT_PENALTY

        # updates test history
        self._recent_heads.appendleft(new_head)

        # timeout / truncated if too many steps
        if self.steps >= self.max_steps:
            truncated = True
            info = {"score": self.score, "steps": self.steps}
            return self._get_obs(), reward, terminated, truncated, info

        info = {"score": self.score, "steps": self.steps}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        # draw background
        self.screen.fill((30, 30, 30))

        # draw cells (snake and food)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                val = 0.0
                # get value from observation for consistency
                if (x, y) in self.snake:
                    if (x, y) == self.snake[0]:
                        val = 0.75
                    else:
                        val = 0.5
                elif (x, y) == self.food:
                    val = 1.0

                if val > 0.0:
                    # Colors: light green head, dark green body, red food
                    if val == 0.75:
                        color = (0, 220, 0)
                    elif val == 0.5:
                        color = (0, 150, 0)
                    else:
                        color = (200, 30, 30)

                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)

        # draw thin grid for clarity
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, (50, 50, 50), (0, i * CELL_SIZE), (GRID_SIZE * CELL_SIZE, i * CELL_SIZE))
            pygame.draw.line(self.screen, (50, 50, 50), (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE * CELL_SIZE))

        pygame.display.flip()

    def close(self):
        # closes pygame cleanly
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:

            pass
