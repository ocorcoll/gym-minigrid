from gym_minigrid.minigrid import Ball, Goal, Box
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


class ToggleBox(Box):

    def can_contain(self):
        return True

    def can_pickup(self):
        return False


class EffectsRoomBoxes(RoomGrid):

    def __init__(self, num_boxes, seed=None):
        room_size = 9
        self.num_boxes = num_boxes
        self.colors = ['yellow', 'blue', 'red', 'purple']

        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=300,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.boxes = list()
        for i, color in zip(range(self.num_boxes), self.colors):
            ball, _ = self.add_object(0, 0, 'ball', color)
            box = FixedBox(color)
            self.place_in_room(0, 0, box)
            self.boxes.append((box, ball))

        goal, _ = self.add_object(0, 0, 'goal')

        self.place_agent(0, 0)
        setattr(goal, 'consumed', -1)
        self.goal = goal
        self.mission = 'go to target location'

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Allow an extra step so the agent sees the effect of "consuming" a goal
        if self.goal.consumed > 0:
            self.goal.consumed -= 1
            reward = self._reward()
        elif self.goal.consumed == 0:
            done = True

        return obs, reward, done, info

    def _reached_goal(self):
        done = False
        reward = 0

        complete = True
        for box, ball in self.boxes:
            if box.contains is None or box.contains != ball:
                complete = False

        if complete:
            self.goal.consumed = 1

        return done, reward


class EffectsRoom(RoomGrid):

    def __init__(self, effect, seed=None):
        """
        This environment test the agent's ability to reach a target location when the complexity of reaching this location increases.
        Complexity increases in the number of effects an agent has to have on the environment to reach a target.

        effect keywords represent
            * k = key
            * t = target
            * d = door
            * b = ball
            * c = chest

        # Tasks:
        1. Go to target (t)
        2. Go to target with ball (bt)
        3. Open door and go to target (kdt)
        4. Open door and go to target with ball (kdbt)
        5. Put ball in chest and go to target (bct)
        6. Open door, put ball in chest and go to target (kdbct)
        """
        room_size = 6
        self.effect = effect
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
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

        self.ball, _ = self.add_object(0, 0, 'ball', 'yellow')
        setattr(self.ball, 'index', 2)

        self.door, _ = self.add_door(0, 0, 0, 'blue', locked=True)
        setattr(self.door, 'index', 3)

        self.key, _ = self.add_object(0, 0, 'key', 'blue')
        setattr(self.key, 'index', 4)

        self.box = FixedBox('purple')
        setattr(self.box, 'index', 5)

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

        # Allow an extra step so the agent sees the effect of "consuming" a goal
        if self.goal.consumed > 0:
            self.goal.consumed -= 1
            reward = self._reward()
        elif self.goal.consumed == 0:
            done = True

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


register(id='MiniGrid-EffectsRoom-t-v0', entry_point='gym_minigrid.envs:EffectsRoomT')
register(id='MiniGrid-EffectsRoom-bt-v0', entry_point='gym_minigrid.envs:EffectsRoomBT')
register(id='MiniGrid-EffectsRoom-kdt-v0', entry_point='gym_minigrid.envs:EffectsRoomKDT')
register(id='MiniGrid-EffectsRoom-kdbt-v0', entry_point='gym_minigrid.envs:EffectsRoomKDBT')
register(id='MiniGrid-EffectsRoom-bct-v0', entry_point='gym_minigrid.envs:EffectsRoomBCT')
register(id='MiniGrid-EffectsRoom-kdbct-v0', entry_point='gym_minigrid.envs:EffectsRoomKDBCT')
