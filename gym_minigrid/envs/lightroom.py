import numpy as np

from enum import IntEnum
from operator import add
from gym_minigrid.roomgrid import RoomGrid, WorldObj, fill_coords, point_in_circle, point_in_rect, COLORS, spaces
from gym_minigrid.register import register
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX


class Reward(WorldObj):

    def __init__(self):
        super().__init__('reward', 'green')
        self.reward = 0
        self.steps = 0
        self.item_type = 'reward'

    def update(self, reward):
        # if self.reward != reward:
        if reward != 0:
            event = [('reward', str(reward))]
        else:
            event = list()

        self.reward = reward
        return event

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.reward * COLORS[self.color])

    def encode(self):
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.reward


class Demon(WorldObj):

    def __init__(self, name, env, color: str = 'grey', movement_type: str = 'random'):
        super().__init__('demon', color)
        self.name = name
        self.env = env
        self.movement_type = movement_type
        self.dir = 0

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

    def move(self):
        old_pos = self.cur_pos
        if self.movement_type == 'random':
            self._move_random()
        elif self.movement_type == 'vertical':
            self._move_vertical()
        else:
            raise ValueError('Movement type not recognized ({})'.format(self.movement_type))

        if np.any(old_pos != self.cur_pos):
            return [(self.name, 'move')]
        else:
            return list()

    def _move_random(self):
        old_pos = self.cur_pos
        top = tuple(map(add, old_pos, (-1, -1)))

        try:
            self.env.place_obj(self, top=top, size=(3, 3), max_tries=100)
            self.env.grid.set(*old_pos, None)
        except:
            pass

    def _move_vertical(self):
        old_pos = self.cur_pos
        if self.dir == 0:
            new_pos = old_pos[0], old_pos[1] + 1
            if self.env.grid.get(*new_pos) is None and np.any(new_pos != self.env.agent_pos):
                self.env.grid.set(*new_pos, self)
                self.env.grid.set(*old_pos, None)
                self.init_pos = new_pos
                self.cur_pos = new_pos
            else:
                self.dir = 1
        elif self.dir == 1:
            new_pos = old_pos[0], old_pos[1] - 1
            if self.env.grid.get(*new_pos) is None and np.any(new_pos != self.env.agent_pos):
                self.env.grid.set(*new_pos, self)
                self.env.grid.set(*old_pos, None)
                self.init_pos = new_pos
                self.cur_pos = new_pos
            else:
                self.dir = 0


class Item(WorldObj):

    def __init__(self, name, item_type, color, ons, enables, items, enabled, on_delay, enable_delay):
        super().__init__(item_type, color)
        self.name = name
        self.item_type = item_type
        self.color = color
        self.on_delay = on_delay
        self.enable_delay = enable_delay
        self.ons = ons
        self.enables = enables
        self.items = items
        self.enabled = enabled
        self.steps = 0
        self.enable_steps = 0
        self.is_on = False

    def enable(self):
        if self.enable_steps > 0 or self.enabled:
            return

        self.enable_steps = self.enable_delay + 1

    def turn_on(self):
        if self.steps > 0 or not self.enabled or self.is_on:
            return

        self.steps = self.on_delay

    def update(self):
        events = list()
        if self.steps == 1:
            self.is_on = not self.is_on
            events.append((self.name, 'on' if self.is_on else 'off'))

            for item_name in self.enables:
                self.items[item_name].enable()

            for item_name in self.ons:
                self.items[item_name].turn_on()

        self.steps = max(0, self.steps - 1)
        if self.enable_steps == 1:
            self.enabled = not self.enabled
            events.append((self.name, 'enabled' if self.enabled else 'disabled'))

        self.enable_steps = max(0, self.enable_steps - 1)
        return events

    def __str__(self):
        return 'name: {}, color: {}, on: {}, enabled: {}'.format(self.name, self.color, self.is_on, self.enabled)


class Button(Item):

    def __init__(self, name, color, ons, enables, items, enabled, on_delay, enable_delay):
        super().__init__(name, 'button', color, ons, enables, items, enabled, on_delay, enable_delay)

    def render(self, img):
        if not self.enabled:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
            fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS['grey'])
        elif self.is_on:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
            fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS['white'])
        else:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

    def toggle(self, env, pos):
        self.turn_on()
        return True

    def encode(self):
        if self.enabled:
            code = 1 if not self.is_on else 2
        else:
            code = 0
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], code


class Light(Item):

    def __init__(self, name, color, ons, enables, items, enabled, on_delay, enable_delay):
        super().__init__(name, 'light', color, ons, enables, items, enabled, on_delay, enable_delay)

    def render(self, img):
        if self.is_on:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.51), COLORS[self.color])
            fill_coords(img, point_in_circle(0.48, 0.48, 0.31), COLORS['white'])
        else:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.51), COLORS[self.color])

    def encode(self):
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.is_on


class LightRoom(RoomGrid):

    def __init__(self, config, on_delay=1, enable_delay=1, num_demons=0, demon_movement='random', seed=None,
                 place_random=False):
        self.place_random = place_random
        self.enable_delay = enable_delay
        self.on_delay = on_delay
        self.config = config
        self.num_demons = num_demons
        self.demon_movement = demon_movement
        self.items = None
        self.demons = None

        self.room_size = 8
        self.num_cols = 1
        self.num_rows = 1

        class Actions(IntEnum):
            left = 0
            right = 1
            forward = 2
            toggle = 3

        super().__init__(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            room_size=self.room_size,
            max_steps=2*self.room_size**2,
            seed=seed,
        )

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        def allow_press(env, pos):
            positions = [(pos[0] + 1, pos[1]), (pos[0] - 1, pos[1]), (pos[0], pos[1] + 1), (pos[0], pos[1] - 1)]
            busy_spots = np.sum([int(env.grid.get(x, y) is not None) for x, y in positions])
            return busy_spots > 2

        self.items = dict()
        self.demons = list()

        reward = Reward()
        self.grid.set(0, 0, reward)
        self.items['reward'] = reward

        for i, (name, (cls, color, ons, enables)) in enumerate(self.config.items()):
            enabled = np.all([name not in item[3] for item in self.config.values()])
            item = cls(name, color, ons, enables, self.items, enabled, self.on_delay if cls is Light else 1, self.enable_delay)
            self.items[name] = item

            if self.place_random:
                self.place_obj(item, reject_fn=allow_press)
            else:
                if cls is Light:
                    self.place_obj(item, (2, i + 1), (1, 1))
                else:
                    self.place_obj(item, (5, i + 1), (1, 1))

        for i in range(self.num_demons):
            demon = Demon('demon_{}'.format(i + 1), self, movement_type=self.demon_movement)
            self.demons.append(demon)
            self.place_obj(demon)

        self.place_agent(0, 0)
        self.mission = 'do nothing'

    def step(self, action):
        prev_agent_state = (self.agent_dir, self.agent_pos[0], self.agent_pos[1])
        # Freeze agent when a change in the obs will happen
        if np.any([item.steps == 1 or getattr(item, 'enable_steps', 0) == 1 for item in self.items.values()]):
            done = self.step_count >= self.max_steps
            info = dict(step=self.step_count, events=list())
            reward = 0
        else:
            obs, reward, done, info = super().step(action)
            info['events'] = list()

        for item in self.items.values():
            if not isinstance(item, Reward):
                info['events'].extend(item.update())

        for demon in self.demons:
            info['events'].extend(demon.move())

        if self.agent_dir != prev_agent_state[0]:
            info['events'].append(('agent', 'move'))

        if self.agent_pos[0] != prev_agent_state[1] or self.agent_pos[1] != prev_agent_state[2]:
            info['events'].append(('agent', 'move'))

        if np.all([light.is_on for light in self.items.values() if light.item_type == 'light']):
            reward = 1.
            done = True

<<<<<<< HEAD
=======
        # Needs to be updated after reward is computed!
>>>>>>> 4743090229ade86ab4d2363cb89902cbc0d91d40
        for item in self.items.values():
            if isinstance(item, Reward):
                info['events'].extend(item.update(reward))

        obs = self.gen_obs()
        return obs, reward, done, info


CONFIG_CHAIN = {
    'light_1': (Light, 'yellow', [], ['button_2']),
    'light_2': (Light, 'orange', [], ['button_3']),
    'light_3': (Light, 'green', [], []),
    'button_1': (Button, 'yellow', ['light_1'], []),
    'button_2': (Button, 'orange', ['light_2'], []),
    'button_3': (Button, 'green', ['light_3'], [])
}

CONFIG_1A = {
    'light_1': (Light, 'yellow', ['light_2'], []),
    'light_2': (Light, 'orange', ['light_3'], []),
    'light_3': (Light, 'green', [], []),
    'button_1': (Button, 'yellow', ['light_1'], []),
    'button_2': (Button, 'orange', [], []),
    'button_3': (Button, 'green', [], [])
}

CONFIG_INDEP = {
    'light_1': (Light, 'yellow', [], []),
    'light_2': (Light, 'orange', [], []),
    'light_3': (Light, 'green', [], []),
    'button_1': (Button, 'yellow', ['light_1'], []),
    'button_2': (Button, 'orange', ['light_2'], []),
    'button_3': (Button, 'green', ['light_3'], [])
}

CONFIG = {'Chain': CONFIG_CHAIN, '1A': CONFIG_1A, 'Indep': CONFIG_INDEP}


class LightRoomEnv(LightRoom):

    def __init__(self, mode, **kwargs):
        super().__init__(config=CONFIG[mode], **kwargs)


class LightEnableRoomDelayChainD1VEnv(LightRoom):

    def __init__(self, seed=None):
        super().__init__(CONFIG_CHAIN, on_delay=7, num_demons=1, demon_movement='vertical', seed=seed)


class LightEnableRoomDelay1AD1VEnv(LightRoom):

    def __init__(self, seed=None):
        super().__init__(CONFIG_1A, on_delay=7, num_demons=1, demon_movement='vertical', seed=seed)


class LightEnableRoomDelayIndepD1VEnv(LightRoom):

    def __init__(self, seed=None):
        super().__init__(CONFIG_INDEP, on_delay=7, num_demons=1, demon_movement='vertical', seed=seed)


class LightEnableRoomDelayChainEnv(LightRoom):

    def __init__(self, seed=None):
        super().__init__(CONFIG_CHAIN, on_delay=7, num_demons=0, seed=seed)


class LightEnableRoomDelay1AEnv(LightRoom):

    def __init__(self, seed=None):
        super().__init__(CONFIG_1A, on_delay=7, num_demons=0, seed=seed)


class LightEnableRoomDelayIndepEnv(LightRoom):

    def __init__(self, seed=None):
        super().__init__(CONFIG_INDEP, on_delay=7, num_demons=0, seed=seed)


register(id='MiniGrid-LightRoomEnv-v0', entry_point='gym_minigrid.envs:LightRoomEnv')

register(id='MiniGrid-LightEnableDelayChainRoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelayChainEnv')
register(id='MiniGrid-LightEnableDelayChainD1VRoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelayChainD1VEnv')

register(id='MiniGrid-LightEnableDelay1ARoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelay1AEnv')
register(id='MiniGrid-LightEnableDelay1AD1VRoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelay1AD1VEnv')

register(id='MiniGrid-LightEnableDelayIndepRoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelayIndepEnv')
register(id='MiniGrid-LightEnableDelayIndepD1VRoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelayIndepD1VEnv')
