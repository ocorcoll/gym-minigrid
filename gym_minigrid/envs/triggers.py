# replicated from https://arxiv.org/abs/1907.08027

from gym import spaces
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from enum import IntEnum

import numpy as np
import random
import time


# Enumeration of possible actions
class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


class TriggersEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """    

    def __init__(self, size=8, agent_view_size=3, n_triggers=2, n_prizes=2):
        self.n_triggers = n_triggers
        self.n_prizes = n_prizes
        self.used_positions = set()
        self.prizes = np.zeros((size, size))
        self.triggers = np.zeros((size, size))
        self.agent_color = 'yellow'
        
        super().__init__(
            grid_size=size,
            max_steps=50,
            agent_view_size=agent_view_size,
        )
        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def _get_free_random_position(self, width, height):
        seed = time.time() % 2**32-1
        random.seed(seed)
        position = tuple(random.sample(range(1, width-1), 2))
        while self.triggers[position] == 1 \
            or self.prizes[position] == 1 \
            or self.agent_pos == position:
            random.seed(seed)
            position = tuple(random.sample(range(1, width-1), 2,)) # assumes width === height
            seed += 1
        self.used_positions.add(position)
        return position

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height, self.agent_color)
        self.prizes = np.zeros((width,height))
        self.triggers = np.zeros((width,height))

        # Generate prizes
        for i in range(self.n_prizes):
            x, y = self._get_free_random_position(width, height)
            self.prizes[x,y] = 1
            prize = Prize('pink')
            self.put_obj(prize, x, y)

        # Generate triggers
        for i in range(self.n_triggers):
            x, y = self._get_free_random_position(width, height)
            self.triggers[x,y] = 1
            self.put_obj(Switch(), x, y)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent at a random position and orientation
        self.place_agent()

        self.mission = "step on all triggers and then collect all rewards"

    def _touched_prize(self, prize):
        some_triggers_left = np.sum(self.triggers) > 0
        no_prizes_left = np.sum(self.prizes) < 2 # if no prizes after this one, done

        if some_triggers_left:
            done = no_prizes_left
            reward = -1
        else:
            done = no_prizes_left 
            reward = 1

        self.prizes[prize.cur_pos] = 0
        self.grid.set(*prize.cur_pos, None)
        return done, reward

    def _touched_switch(self, switch):
        self.triggers[switch.cur_pos] = 0
        self.grid.set(*switch.cur_pos, None)
        return False, 0

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'switch':
                done, reward = self._touched_switch(fwd_cell)
            if fwd_cell != None and fwd_cell.type == 'prize':
                done, reward = self._touched_prize(fwd_cell)
        else:
            pass # ignoring unknown actions

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, dict(step=self.step_count)

class TriggersEnv3x3Dumb(TriggersEnv):
    def __init__(self):
        super().__init__(agent_view_size=3, n_triggers=0, n_prizes=2)

class TriggersEnv3x3T1P1(TriggersEnv):
    def __init__(self):
        super().__init__(agent_view_size=3, n_triggers=1, n_prizes=1)

class TriggersEnv3x3(TriggersEnv):
    def __init__(self):
        super().__init__(agent_view_size=3, n_triggers=2, n_prizes=2)

class TriggersEnv5x5(TriggersEnv):
    def __init__(self):
        super().__init__(agent_view_size=5, n_triggers=2, n_prizes=2)

class TriggersEnv5x5T1P1(TriggersEnv):
    def __init__(self):
        super().__init__(agent_view_size=5, n_triggers=1, n_prizes=1)

class TriggersEnv7x7(TriggersEnv):
    def __init__(self):
        super().__init__(agent_view_size=7, n_triggers=2, n_prizes=2)

class TriggersEnv7x7T1P1(TriggersEnv):
    def __init__(self):
        super().__init__(agent_view_size=7, n_triggers=1, n_prizes=1)

register(
    id='MiniGrid-Triggers-3x3-Dumb-v0',
    entry_point='gym_minigrid.envs:TriggersEnv3x3Dumb'
)

register(
    id='MiniGrid-Triggers-3x3-T1P1-v0',
    entry_point='gym_minigrid.envs:TriggersEnv3x3T1P1'
)

register(
    id='MiniGrid-Triggers-3x3-v0',
    entry_point='gym_minigrid.envs:TriggersEnv3x3'
)

register(
    id='MiniGrid-Triggers-5x5-v0',
    entry_point='gym_minigrid.envs:TriggersEnv5x5'
)

register(
    id='MiniGrid-Triggers-5x5-T1P1-v0',
    entry_point='gym_minigrid.envs:TriggersEnv5x5T1P1'
)

register(
    id='MiniGrid-Triggers-7x7-v0',
    entry_point='gym_minigrid.envs:TriggersEnv7x7'
)

register(
    id='MiniGrid-Triggers-7x7-T1P1-v0',
    entry_point='gym_minigrid.envs:TriggersEnv7x7T1P1'
)
