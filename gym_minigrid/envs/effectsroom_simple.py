import re
import numpy as np

from operator import add

from gym_minigrid.envs import Demon
from gym_minigrid.minigrid import Ball, Goal
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register


class EffectsRoomSimple(RoomGrid):

    def __init__(self, num_demons=1, seed=None, demon_mode='horizontal'):
        self.demon_mode = demon_mode
        self.colors = ['yellow', 'purple', 'blue', 'red']
        self.num_elements = 0
        self.num_demons = num_demons
        self.balls = None
        self.demons = None

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

    def reject_next_to(self, env, pos):
        sx, sy = self.door.cur_pos
        x, y = pos
        d = abs(sx - x) + abs(sy - y)
        return d < 2

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.balls = list()
        self.demons = list()
        self.num_elements = 1

        num_balls = 1
        for i in range(num_balls):
            color = self.colors[i]
            ball, _ = self.add_object(0, 0, 'ball', color)
            setattr(ball, 'index', self.num_elements)
            self.balls.append(ball)
            self.num_elements += 1

        for i in range(self.num_demons):
            demon = Demon()
            self.place_in_room(0, 0, demon)
            setattr(demon, 'index', self.num_elements)
            self.demons.append(demon)
            self.num_elements += 1

        self.place_agent(0, 0)
        self.mission = 'grab ball'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        def move(demon, next_position, direction):
            self.grid.set(*demon.cur_pos, None)
            demon.init_pos = next_position
            demon.cur_pos = next_position
            demon.current_dir = direction
            self.grid.set(*next_position, demon)

        def can_move(position):
            if self.grid.get(*position) is not None:
                return False

            # Don't place the object where the agent is
            if np.array_equal(position, self.agent_pos):
                return False

            if position[0] >= self.width or position[1] >= self.height or position[0] < 0 or position[1] < 0:
                return False

            return True

        if self.demons is not None and self.demon_mode != 'static':
            for i_obst in range(len(self.demons)):
                demon = self.demons[i_obst]
                if self.demon_mode == 'random':
                    old_pos = demon.cur_pos
                    top = tuple(map(add, old_pos, (-1, -1)))

                    try:
                        self.place_obj(demon, top=top, size=(3, 3), max_tries=100)
                        self.grid.set(*old_pos, None)
                    except:
                        pass
                else:
                    right_position = (demon.cur_pos[0] + 1, demon.cur_pos[1])
                    down_position = (demon.cur_pos[0], demon.cur_pos[1] - 1)
                    left_position = (demon.cur_pos[0] - 1, demon.cur_pos[1])
                    up_position = (demon.cur_pos[0], demon.cur_pos[1] + 1)

                    afford = {k: can_move(v) for k, v in [('right', right_position), ('down', down_position), ('left', left_position), ('up', up_position)]}
                    has_moved = False
                    while not has_moved:
                        empty_forward = (demon.current_dir == 0 and afford['right']) or (demon.current_dir == 1 and afford['down']) or (demon.current_dir == 2 and afford['left']) or (demon.current_dir == 3 and afford['up'])
                        empty_backward = (demon.current_dir == 0 and afford['left']) or (demon.current_dir == 1 and afford['up']) or (demon.current_dir == 2 and afford['right']) or (demon.current_dir == 3 and afford['down'])
                        empty_right = (demon.current_dir == 0 and afford['down']) or (demon.current_dir == 1 and afford['left']) or (demon.current_dir == 2 and afford['up']) or (demon.current_dir == 3 and afford['right'])

                        if self.demon_mode == 'circle':
                            cannot_move = not np.any(list(afford.values()))
                            if empty_forward and not empty_right:
                                # Go forwards
                                demon.current_dir = demon.current_dir
                            elif not empty_forward and empty_right:
                                # Turn right
                                demon.current_dir = (demon.current_dir + 1) % 4
                            elif not empty_forward and not empty_right:
                                # Turn left
                                demon.current_dir = (demon.current_dir - 1) % 4
                        else:
                            if not empty_forward:
                                demon.current_dir = 0 if demon.current_dir == 2 else 2

                            cannot_move = not empty_forward and not empty_backward

                        if cannot_move:
                            has_moved = True
                        elif demon.current_dir == 0 and afford['right']:
                            move(demon, right_position, direction=0)
                            has_moved = True
                        elif demon.current_dir == 1 and afford['down']:
                            move(demon, down_position, direction=1)
                            has_moved = True
                        elif demon.current_dir == 2 and afford['left']:
                            move(demon, left_position, direction=2)
                            has_moved = True
                        elif demon.current_dir == 3 and afford['up']:
                            move(demon, up_position, direction=3)
                            has_moved = True

        return obs, reward, done, info

    def _reached_goal(self):
        done = False
        reward = 0
        return done, reward


class EffectsRoomSimpleE1(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=1, seed=seed, demon_mode='horizontal')


class EffectsRoomSimpleE2(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=2, seed=seed, demon_mode='horizontal')


class EffectsRoomSimpleE3(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=3, seed=seed, demon_mode='horizontal')


class EffectsRoomSimpleE1Circle(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=1, seed=seed, demon_mode='circle')


class EffectsRoomSimpleE2Circle(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=2, seed=seed, demon_mode='circle')


class EffectsRoomSimpleE3Circle(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=3, seed=seed, demon_mode='circle')


class EffectsRoomSimpleE1Static(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=1, seed=seed, demon_mode='static')


class EffectsRoomSimpleE2Static(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=2, seed=seed, demon_mode='static')


class EffectsRoomSimpleE3Static(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=3, seed=seed, demon_mode='static')


class EffectsRoomSimpleE1Random(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=1, seed=seed, demon_mode='random')


class EffectsRoomSimpleE2Random(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=2, seed=seed, demon_mode='random')


class EffectsRoomSimpleE3Random(EffectsRoomSimple):

    def __init__(self, seed=None):
        super().__init__(num_demons=3, seed=seed, demon_mode='random')


register(id='MiniGrid-EffectsRoomSimple-horizontal-e1-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE1')
register(id='MiniGrid-EffectsRoomSimple-horizontal-e2-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE2')
register(id='MiniGrid-EffectsRoomSimple-horizontal-e3-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE3')
register(id='MiniGrid-EffectsRoomSimple-circle-e1-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE1Circle')
register(id='MiniGrid-EffectsRoomSimple-circle-e2-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE2Circle')
register(id='MiniGrid-EffectsRoomSimple-circle-e3-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE3Circle')
register(id='MiniGrid-EffectsRoomSimple-static-e1-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE1Static')
register(id='MiniGrid-EffectsRoomSimple-static-e2-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE2Static')
register(id='MiniGrid-EffectsRoomSimple-static-e3-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE3Static')
register(id='MiniGrid-EffectsRoomSimple-random-e1-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE1Random')
register(id='MiniGrid-EffectsRoomSimple-random-e2-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE2Random')
register(id='MiniGrid-EffectsRoomSimple-random-e3-v0', entry_point='gym_minigrid.envs:EffectsRoomSimpleE3Random')
