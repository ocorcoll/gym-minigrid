import re
import numpy as np

from operator import add
from gym_minigrid.minigrid import Ball, Goal, Box, WorldObj, COLORS, fill_coords, point_in_circle, MiniGridEnv, OBJECT_TO_IDX, COLOR_TO_IDX, point_in_rect
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register


class FixedBox(Box):

    def can_contain(self):
        return True

    def toggle(self, env, pos):
        return False

    def can_pickup(self):
        return False

    def can_pickup_content(self):
        return True

    def render(self, img):
        if self.contains is None:
            c = COLORS[self.color]
        else:
            c = COLORS[self.contains.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)


class Demon(WorldObj):

    def __init__(self, env: MiniGridEnv, movement_mode: str, color: str ='grey'):
        super().__init__('demon', color)
        self.current_dir = 0
        self.env = env
        self.movement_mode = movement_mode

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
        if self.movement_mode == 'random':
            old_pos = self.cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.env.place_obj(self, top=top, size=(3, 3), max_tries=100)
                self.env.grid.set(*old_pos, None)
            except:
                pass
        else:
            right_position = (self.cur_pos[0] + 1, self.cur_pos[1])
            down_position = (self.cur_pos[0], self.cur_pos[1] - 1)
            left_position = (self.cur_pos[0] - 1, self.cur_pos[1])
            up_position = (self.cur_pos[0], self.cur_pos[1] + 1)

            afford = {k: self.can_move(v) for k, v in [('right', right_position), ('down', down_position), ('left', left_position), ('up', up_position)]}
            has_moved = False
            while not has_moved:
                empty_forward = (self.current_dir == 0 and afford['right']) or (self.current_dir == 1 and afford['down']) or (self.current_dir == 2 and afford['left'])\
                                or (self.current_dir == 3 and afford['up'])
                empty_backward = (self.current_dir == 0 and afford['left']) or (self.current_dir == 1 and afford['up']) or (self.current_dir == 2 and afford['right'])\
                                 or (self.current_dir == 3 and afford['down'])
                empty_right = (self.current_dir == 0 and afford['down']) or (self.current_dir == 1 and afford['left']) or (self.current_dir == 2 and afford['up'])\
                              or (self.current_dir == 3 and afford['right'])

                if self.movement_mode == 'circle':
                    cannot_move = not np.any(list(afford.values()))
                    if empty_forward and not empty_right:
                        # Go forwards
                        self.current_dir = self.current_dir
                    elif not empty_forward and empty_right:
                        # Turn right
                        self.current_dir = (self.current_dir + 1) % 4
                    elif not empty_forward and not empty_right:
                        # Turn left
                        self.current_dir = (self.current_dir - 1) % 4
                else:
                    if not empty_forward:
                        self.current_dir = 0 if self.current_dir == 2 else 2

                    cannot_move = not empty_forward and not empty_backward

                if cannot_move:
                    has_moved = True
                elif self.current_dir == 0 and afford['right']:
                    self._move(right_position, direction=0)
                    has_moved = True
                elif self.current_dir == 1 and afford['down']:
                    self._move(down_position, direction=1)
                    has_moved = True
                elif self.current_dir == 2 and afford['left']:
                    self._move(left_position, direction=2)
                    has_moved = True
                elif self.current_dir == 3 and afford['up']:
                    self._move(up_position, direction=3)
                    has_moved = True


class EffectsRoom(RoomGrid):

    def __init__(self, effect, seed=None, use_large=False, use_single=False, demon_mode='horizontal'):
        """
        This environment test the agent's ability to reach a target location when the complexity of reaching this location increases.
        Complexity increases in the number of effects an agent has to perform on the environment to reach a target.

        effect keywords represent
            * k = key
            * t = target
            * d = door
            * b = ball
            * c = chest
            * e = env dynamic obj
        """
        self.effect = effect
        self.demon_mode = demon_mode
        self.use_single = use_single
        self.colors = ['pink', 'yellow', 'purple', 'blue', 'red']
        self.num_elements = 0
        self.balls = None
        self.boxes = None
        self.demons = None

        self.room_size = 8  # 8 if use_large else 6
        self.num_cols = 1 if use_single else 2
        self.num_rows = 1
        super().__init__(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            room_size=self.room_size,
            max_steps=500, #16*self.room_size**2,
            seed=seed,
        )

    def reject_next_to(self, env, pos):
        sx, sy = self.door.cur_pos
        x, y = pos
        d = abs(sx - x) + abs(sy - y)
        return d < 2

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.goal = Goal()
        setattr(self.goal, 'index', 1)
        self.balls = list()
        self.boxes = list()
        self.demons = list()

        if self.use_single:
            self.place_in_room(0, 0, self.goal)
            self.num_elements = 2

            # num_balls = 0 if 'b' not in self.effect else int(re.search('.*b([0-9]).*', self.effect)[1])
            num_balls = 1
            for i in range(num_balls):
                # color = self.colors[i]
                color = self.colors[0]
                ball, _ = self.add_object(0, 0, 'ball', color)
                setattr(ball, 'index', 2 + i)
                self.balls.append(ball)
                self.num_elements += 1

            # num_boxes = 0 if 'c' not in self.effect else int(re.search('.*c([0-9]).*', self.effect)[1])
            num_boxes = 1
            for i in range(num_boxes):
                # color = self.colors[i]
                color = self.colors[3]
                box = FixedBox(color)
                self.place_in_room(0, 0, box)
                setattr(box, 'index', 2 + num_balls + i)
                self.boxes.append(box)
                self.num_elements += 1

            num_demons = 1 if 'e' not in self.effect else int(re.search('.*e([0-9]).*', self.effect)[1])
            for i in range(num_demons):
                demon = Demon(self, movement_mode=self.demon_mode)
                self.place_in_room(0, 0, demon)
                setattr(demon, 'index', 2 + num_balls + num_boxes + i)
                self.demons.append(demon)
                self.num_elements += 1
        else:
            self.ball, _ = self.add_object(0, 0, 'ball', 'yellow')
            setattr(self.ball, 'index', 2)

            self.box = FixedBox('purple')
            setattr(self.box, 'index', 3)

            self.door, _ = self.add_door(0, 0, 0, 'blue', locked=True)
            setattr(self.door, 'index', 5)

            self.key, _ = self.add_object(0, 0, 'key', 'blue')
            setattr(self.key, 'index', 4)

            if 'd' in self.effect:
                self.place_in_room(1, 0, self.goal)
                self.place_in_room(1, 0, self.box, reject_fn=self.reject_next_to)
            else:
                self.place_in_room(0, 0, self.goal)
                self.place_in_room(0, 0, self.box)

            self.num_elements = 6

        self.place_agent(0, 0)
        setattr(self.goal, 'consumed', -1)
        self.mission = 'go to target location'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.goal.consumed >= 0:
            reward = self._reward()
            done = True

        if self.demons is not None and 'e' in self.effect:
            for i_obst in range(len(self.demons)):
                self.demons[i_obst].move()

        return obs, reward, done, info

    def _reached_goal(self):
        done = False
        reward = 0
        if 'b' in self.effect and 'c' in self.effect:
            requirements = [type(box.contains) is Ball and box.contains.color == box.color for box in self.boxes]
            if np.all(requirements):
                self.goal.consumed = 1
        elif 'b' in self.effect:
            if type(self.carrying) is Ball:
                self.goal.consumed = 1
        else:
            self.goal.consumed = 1

        return done, reward

    def get_state(self):
        env_state = np.zeros((self.num_elements, 5))
        env_state[0, 0] = OBJECT_TO_IDX['agent']
        env_state[0, 1] = self.agent_pos[0] + 1
        env_state[0, 2] = self.agent_pos[1] + 1
        env_state[0, 3] = self.agent_dir + 1
        env_state[0, 4] = OBJECT_TO_IDX[self.carrying.type] if self.carrying else 1

        for x in range(self.width):
            for y in range(self.height):
                element = self.grid.get(x, y)
                if element is not None and OBJECT_TO_IDX[element.type] >= 4:
                    env_state[element.index, 0] = OBJECT_TO_IDX[element.type]
                    env_state[element.index, 1] = x + 1
                    env_state[element.index, 2] = y + 1
                    env_state[element.index, 3] = COLOR_TO_IDX[element.color]

                    element_state = 0
                    if hasattr(element, 'is_open'):
                        if element.is_open:
                            element_state = 1
                        else:
                            element_state = 2

                    if hasattr(element, 'is_locked') and element.is_locked:
                        element_state = 3

                    if element.can_contain():
                        if element.contains:
                            element_state = OBJECT_TO_IDX[element.contains.type]
                        else:
                            element_state = 1

                    if OBJECT_TO_IDX[element.type] == OBJECT_TO_IDX['goal']:
                        element_state = 10 if x == self.agent_pos[0] and y == self.agent_pos[1] else 1

                    env_state[element.index, 4] = element_state

        return env_state


class EffectsRoomT(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('t', seed)


class EffectsRoomBT(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('bt', seed)


class EffectsRoomKDT(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('kdt', seed)


class EffectsRoomKDBT(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('kdbt', seed)


class EffectsRoomBCT(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('bct', seed)


class EffectsRoomKDBCT(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('kdbct', seed)


class EffectsRoomSingleE1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1t', seed, use_single=True)


class EffectsRoomSingleE1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1b1t', seed, use_single=True)


class EffectsRoomSingleE1C1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1c1b1t', seed, use_single=True)


class EffectsRoomSingleCircleE1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1t', seed, use_single=True, demon_mode='circle')


class EffectsRoomSingleCircleE1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1b1t', seed, use_single=True, demon_mode='circle')


class EffectsRoomSingleCircleE1C1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1c1b1t', seed, use_single=True, demon_mode='circle')


class EffectsRoomSingleRandomE1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1t', seed, use_single=True, demon_mode='random')


class EffectsRoomSingleRandomE1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1b1t', seed, use_single=True, demon_mode='random')


class EffectsRoomSingleRandomE1C1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1c1b1t', seed, use_single=True, demon_mode='random')


class EffectsRoomSingleXLE1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1t', seed, use_large=True, use_single=True)


class EffectsRoomSingleXLE1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1b1t', seed, use_large=True, use_single=True)


class EffectsRoomSingleXLE1C1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e1c1b1t', seed, use_large=True, use_single=True)


class EffectsRoomSingleXLE2C1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e2c1b1t', seed, use_large=True, use_single=True)


class EffectsRoomSingleXLE3C1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e3c1b1t', seed, use_large=True, use_single=True)


class EffectsRoomSingleXLCircleE3C1B1T(EffectsRoom):

    def __init__(self, seed=None):
        super().__init__('e3c1b1t', seed, use_large=True, use_single=True, demon_mode='circle')


register(id='MiniGrid-EffectsRoom-t-v0', entry_point='gym_minigrid.envs:EffectsRoomT')
register(id='MiniGrid-EffectsRoom-bt-v0', entry_point='gym_minigrid.envs:EffectsRoomBT')
register(id='MiniGrid-EffectsRoom-kdt-v0', entry_point='gym_minigrid.envs:EffectsRoomKDT')
register(id='MiniGrid-EffectsRoom-kdbt-v0', entry_point='gym_minigrid.envs:EffectsRoomKDBT')
register(id='MiniGrid-EffectsRoom-bct-v0', entry_point='gym_minigrid.envs:EffectsRoomBCT')
register(id='MiniGrid-EffectsRoom-kdbct-v0', entry_point='gym_minigrid.envs:EffectsRoomKDBCT')

register(id='MiniGrid-EffectsRoomSingle-e1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleE1T')
register(id='MiniGrid-EffectsRoomSingle-e1b1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleE1B1T')
register(id='MiniGrid-EffectsRoomSingle-e1c1b1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleE1C1B1T')

register(id='MiniGrid-EffectsRoomSingle-e1t-circle-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleCircleE1T')
register(id='MiniGrid-EffectsRoomSingle-e1b1t-circle-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleCircleE1B1T')
register(id='MiniGrid-EffectsRoomSingle-e1c1b1t-circle-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleCircleE1C1B1T')

register(id='MiniGrid-EffectsRoomSingle-e1t-random-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleRandomE1T')
register(id='MiniGrid-EffectsRoomSingle-e1b1t-random-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleRandomE1B1T')
register(id='MiniGrid-EffectsRoomSingle-e1c1b1t-random-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleRandomE1C1B1T')

register(id='MiniGrid-EffectsRoomSingleXL-t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLT')
register(id='MiniGrid-EffectsRoomSingleXL-e1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLE1T')
register(id='MiniGrid-EffectsRoomSingleXL-b1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLB1T')
register(id='MiniGrid-EffectsRoomSingleXL-e1b1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLE1B1T')
register(id='MiniGrid-EffectsRoomSingleXL-c1b1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLC1B1T')
register(id='MiniGrid-EffectsRoomSingleXL-e1c1b1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLE1C1B1T')
register(id='MiniGrid-EffectsRoomSingleXL-e2c1b1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLE2C1B1T')
register(id='MiniGrid-EffectsRoomSingleXL-e3c1b1t-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLE3C1B1T')
register(id='MiniGrid-EffectsRoomSingleXL-e3c1b1t-circle-v0', entry_point='gym_minigrid.envs:EffectsRoomSingleXLCircleE3C1B1T')
