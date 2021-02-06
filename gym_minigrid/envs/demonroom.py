from collections import deque

import numpy as np

from operator import add

from gym_minigrid.roomgrid import RoomGrid, WorldObj, MiniGridEnv, fill_coords, point_in_circle, COLORS
from gym_minigrid.register import register


class Demon(WorldObj):

    def __init__(self, env: MiniGridEnv, parent, colors):
        super().__init__('demon', colors[0])

        self.colors = colors
        self.parent = parent
        self.current_dir = 0
        self.pos_history = deque(maxlen=3)
        self.color_history = deque(maxlen=3)
        self.env = env

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

    def can_contain(self):
        return False

    def toggle(self, env, pos):
        return False

    def can_pickup(self):
        return False

    def can_pickup_content(self):
        return False

    def _move(self, next_position, direction):
        self.env.grid.set(*self.cur_pos, None)
        self.init_pos = next_position
        self.cur_pos = next_position
        self.current_dir = direction
        self.env.grid.set(*next_position, self)

    def can_move(self, position):
        if self.env.grid.get(*position) is not None:
            return False

        # Don't place the object where the agent is
        if np.array_equal(position, self.env.agent_pos):
            return False

        if position[0] >= self.env.width or position[1] >= self.env.height or position[0] < 0 or position[1] < 0:
            return False

        return True

    def move(self):
        self.pos_history.append(self.cur_pos)

        new_x = 0
        new_y = 0
        if np.random.rand() >= 0.5:
            new_x = np.random.choice([+1, -1])
        else:
            new_y = np.random.choice([+1, -1])

        top = tuple(map(add, self.pos_history[-1], (new_x, new_y)))

        try:
            self.env.place_obj(self, top=top, size=(1, 1), max_tries=100)
            self.env.grid.set(*self.pos_history[-1], None)
        except:
            pass

    def change_color(self):
        if self.parent is not None:
            self.color_history.append(0 if self.parent.color == self.parent.colors[0] else 1)
        else:
            self.color_history.append(int(self.env.agent_dir % 2 == 0))

        if len(self.color_history) <= 2:
            return

        if self.color_history[-3] != self.color_history[-2]:
            self.color = self.colors[0] if self.color_history[-2] == 0 else self.colors[1]


class DemonRoom(RoomGrid):

    def __init__(self, num_demons, seed=None):
        self.demons = None
        self.num_demons = num_demons

        self.room_size = 8
        self.num_cols = 1
        self.num_rows = 1
        super().__init__(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            room_size=self.room_size,
            max_steps=16*self.room_size**2,
            seed=seed,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.demons = list()
        self.num_elements = 1

        for i, colors in enumerate([('blue', 'brown'), ('green', 'pink')][:self.num_demons]):
            parent = self.demons[-1] if len(self.demons) > 0 else None
            demon = Demon(self, parent, colors)
            self.place_obj(demon, (i * (self.room_size - 3) + 1, 1), (1, 1))
            self.demons.append(demon)

        self.place_agent(0, 0)
        self.mission = 'grab ball'

    def step(self, action):
        # Don't allow to move
        if action == 2:
            action = 3

        obs, reward, done, info = super().step(action)
        for demon in self.demons:
            demon.move()
            demon.change_color()

        obs = self.gen_obs()
        return obs, reward, done, info

    def _reached_goal(self):
        done = False
        reward = 0
        return done, reward


class DemonRoomEnv(DemonRoom):

    def __init__(self, seed=None):
        super().__init__(num_demons=2, seed=seed)


register(id='MiniGrid-DemonRoom-v0', entry_point='gym_minigrid.envs:DemonRoomEnv')
