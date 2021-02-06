import numpy as np

from gym_minigrid.minigrid import Ball, Goal, Key
from gym_minigrid.envs.effectsroom import Demon, FixedBox
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register


class ComplexEffectsRoom(RoomGrid):

    def __init__(self, effect, seed=None):
        """
        This environment test the agent's ability to reach a target location when the complexity of reaching this location increases.
        Complexity increases in the number of effects an agent has to perform on the environment to reach a target.

        effect keywords represent
            * k = key
            * t = target
            * d = door
            * b = ball
            * c = chest
            * e = demon
        """
        self.effect = effect
        self.num_elements = 0
        self.ball = None
        self.box = None
        self.key = None
        self.door = None
        self.demon = None

        self.room_size = 6
        self.num_rows = 1
        self.num_cols = 2
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

    def set_object(self, obj, reject_next_to=None):
        room_y = 0
        if isinstance(obj, Key) or isinstance(obj, Demon) or ('bd' in self.effect and isinstance(obj, Ball)) or ('cbd' in self.effect and isinstance(obj, FixedBox))\
                or ('cd' in self.effect and isinstance(obj, FixedBox)):
            room_x = 0
        else:
            room_x = 1

        self.place_in_room(room_x, room_y, obj, reject_next_to)
        setattr(obj, 'index', self.num_elements)
        self.num_elements += 1
        return obj

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.num_elements = 1
        self.goal = Goal()
        self.set_object(self.goal)
        setattr(self.goal, 'consumed', -1)

        self.key = Key('blue')
        self.set_object(self.key)

        self.door, _ = self.add_door(0, 0, 0, 'blue', locked=True)
        setattr(self.door, 'index', self.num_elements)
        self.num_elements += 1

        self.ball = Ball('yellow')
        self.set_object(self.ball, self.reject_next_to)

        self.box = FixedBox('yellow')
        self.set_object(self.box)

        self.demon = Demon()
        self.set_object(self.demon)

        self.place_agent(0, 0)
        self.mission = 'go to target location'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.goal.consumed >= 0:
            reward = self._reward()
            done = True

        def move(next_position, direction):
            self.grid.set(*self.demon.cur_pos, None)
            self.demon.init_pos = next_position
            self.demon.cur_pos = next_position
            self.demon.current_dir = direction
            self.grid.set(*next_position, self.demon)

        def can_move(position):
            if self.grid.get(*position) is not None:
                return False

            # Don't place the object where the agent is
            if np.array_equal(position, self.agent_pos):
                return False

            if position[0] >= self.width or position[1] >= self.height or position[0] < 0 or position[1] < 0:
                return False

            return True

        right_position = (self.demon.cur_pos[0] + 1, self.demon.cur_pos[1])
        down_position = (self.demon.cur_pos[0], self.demon.cur_pos[1] - 1)
        left_position = (self.demon.cur_pos[0] - 1, self.demon.cur_pos[1])
        up_position = (self.demon.cur_pos[0], self.demon.cur_pos[1] + 1)

        if self.demon.current_dir == 0 and can_move(right_position):
            move(right_position, direction=0)
        elif self.demon.current_dir == 1 and can_move(down_position):
            move(down_position, direction=1)
        elif self.demon.current_dir == 2 and can_move(left_position):
            move(left_position, direction=2)
        elif self.demon.current_dir == 3 and can_move(up_position):
            move(up_position, direction=3)
        else:
            self.demon.current_dir = 0 if self.demon.current_dir == 2 else 2
            # self.demon.current_dir = (self.demon.current_dir + 1) % 4

        # top = tuple(map(add, old_pos, (-1, -1)))
        #
        # try:
        #     self.place_obj(self.demon, top=top, size=(3, 3), max_tries=100)
        #     self.grid.set(*old_pos, None)
        # except:
        #     pass

        return obs, reward, done, info

    def _reached_goal(self):
        done = False
        reward = 0
        if 'b' in self.effect and 'c' in self.effect:
            if type(self.box.contains) is Ball:
                self.goal.consumed = 1
        elif 'b' in self.effect:
            if type(self.carrying) is Ball:
                self.goal.consumed = 1
        else:
            self.goal.consumed = 1

        return done, reward


class ComplexEffectsRoomDT(ComplexEffectsRoom):

    def __init__(self, seed=None):
        super().__init__('dt', seed)


class ComplexEffectsRoomDBT(ComplexEffectsRoom):

    def __init__(self, seed=None):
        super().__init__('dbt', seed)


class ComplexEffectsRoomBDT(ComplexEffectsRoom):

    def __init__(self, seed=None):
        super().__init__('bdt', seed)


class ComplexEffectsRoomDCBT(ComplexEffectsRoom):

    def __init__(self, seed=None):
        super().__init__('dcbt', seed)


class ComplexEffectsRoomCBDT(ComplexEffectsRoom):

    def __init__(self, seed=None):
        super().__init__('cbdt', seed)


class ComplexEffectsRoomCDBT(ComplexEffectsRoom):

    def __init__(self, seed=None):
        super().__init__('cdbt', seed)


register(id='MiniGrid-ComplexEffectsRoom-dt-v0', entry_point='gym_minigrid.envs:ComplexEffectsRoomDT')
register(id='MiniGrid-ComplexEffectsRoom-dbt-v0', entry_point='gym_minigrid.envs:ComplexEffectsRoomDBT')
register(id='MiniGrid-ComplexEffectsRoom-bdt-v0', entry_point='gym_minigrid.envs:ComplexEffectsRoomBDT')
register(id='MiniGrid-ComplexEffectsRoom-dcbt-v0', entry_point='gym_minigrid.envs:ComplexEffectsRoomDCBT')
register(id='MiniGrid-ComplexEffectsRoom-cbdt-v0', entry_point='gym_minigrid.envs:ComplexEffectsRoomCBDT')
register(id='MiniGrid-ComplexEffectsRoom-cdbt-v0', entry_point='gym_minigrid.envs:ComplexEffectsRoomCDBT')
