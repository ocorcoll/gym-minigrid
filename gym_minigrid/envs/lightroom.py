import numpy as np

from enum import IntEnum
from operator import add
from gym_minigrid.roomgrid import RoomGrid, WorldObj, fill_coords, point_in_circle, point_in_rect, COLORS, spaces
from gym_minigrid.register import register
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX


class Demon(WorldObj):

    def __init__(self, env, color: str ='grey'):
        super().__init__('demon', color)
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

    def move(self):
        old_pos = self.cur_pos
        top = tuple(map(add, old_pos, (-1, -1)))

        try:
            self.env.place_obj(self, top=top, size=(3, 3), max_tries=100)
            self.env.grid.set(*old_pos, None)
        except:
            pass


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

    def __init__(self, config, on_delay=1, enable_delay=1, num_demons=0, seed=None):
        self.enable_delay = enable_delay
        self.on_delay = on_delay
        self.config = config
        self.num_demons = num_demons
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
            max_steps=4*self.room_size**2,
            seed=seed,
        )

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.items = dict()
        self.demons = list()

        for i, (name, (cls, color, ons, enables)) in enumerate(self.config.items()):
            enabled = np.all([name not in item[3] for item in self.config.values()])
            item = cls(name, color, ons, enables, self.items, enabled, self.on_delay if cls is Light else 1, self.enable_delay)
            self.items[name] = item
            if cls is Light:
                self.place_obj(item, (2, i + 1), (1, 1))
            else:
                self.place_obj(item, (5, i + 1), (1, 1))

        for i in range(self.num_demons):
            demon = Demon(self)
            self.demons.append(demon)
            self.place_obj(demon, (5, i + 1), (1, 1))

        self.place_agent(0, 0)
        self.mission = 'do nothing'

    def step(self, action):
        prev_agent_state = (self.agent_dir, self.agent_pos[0], self.agent_pos[1])
        # Freeze agent when a change in the obs will happen
        if np.any([item.steps == 1 or getattr(item, 'enable_steps', 0) == 1 for item in self.items.values()]):
            done = self.step_count >= self.max_steps
            info = dict(step=self.step_count)
            reward = 0
        else:
            obs, reward, done, info = super().step(action)
            for demon in self.demons:
                demon.move()

        if np.all([light.is_on for light in self.items.values() if light.item_type == 'light']):
            reward = 1.
            done = True

        info['events'] = list()
        for item in self.items.values():
            info['events'].extend(item.update())

        if self.agent_dir != prev_agent_state[0]:
            info['events'].append(('agent', 'rotate'))

        if self.agent_pos[0] != prev_agent_state[1] or self.agent_pos[1] != prev_agent_state[2]:
            info['events'].append(('agent', 'forward'))

        obs = self.gen_obs()
        return obs, reward, done, info


class LightEnableRoomDelayChainD1Env(LightRoom):

    def __init__(self, seed=None):
        config = {'light_1': (Light, 'green', [], ['button_2']), 'light_2': (Light, 'orange', [], ['button_3']), 'light_3': (Light, 'yellow', [], []),
                  'button_1': (Button, 'pink', ['light_1'], []), 'button_2': (Button, 'blue', ['light_2'], []), 'button_3': (Button, 'purple', ['light_3'], [])}

        super().__init__(config, on_delay=7, num_demons=1, seed=seed)


class LightEnableRoomDelayChainEnv(LightRoom):

    def __init__(self, seed=None):
        config = {'light_1': (Light, 'green', [], ['button_2']), 'light_2': (Light, 'orange', [], ['button_3']), 'light_3': (Light, 'yellow', [], []),
                  'button_1': (Button, 'pink', ['light_1'], []), 'button_2': (Button, 'blue', ['light_2'], []), 'button_3': (Button, 'purple', ['light_3'], [])}

        super().__init__(config, on_delay=7, num_demons=0, seed=seed)


class LightEnableRoomDelay1AEnv(LightRoom):

    def __init__(self, seed=None):
        config = {'light_1': (Light, 'green', ['light_2'], []), 'light_2': (Light, 'orange', ['light_3'], []), 'light_3': (Light, 'yellow', [], []),
                  'button_1': (Button, 'pink', ['light_1'], []), 'button_2': (Button, 'blue', [], []), 'button_3': (Button, 'purple', [], [])}
        super().__init__(config, on_delay=7, num_demons=0, seed=seed)


class LightEnableRoomDelayIndepEnv(LightRoom):

    def __init__(self, seed=None):
        config = {'light_1': (Light, 'green', [], []), 'light_2': (Light, 'orange', [], []), 'light_3': (Light, 'yellow', [], []),
                  'button_1': (Button, 'pink', ['light_1'], []), 'button_2': (Button, 'blue', ['light_2'], []), 'button_3': (Button, 'purple', ['light_3'], [])}
        super().__init__(config, on_delay=7, num_demons=0, seed=seed)


register(id='MiniGrid-LightEnableDelayChainD1Room-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelayChainD1Env')

register(id='MiniGrid-LightEnableDelayChainRoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelayChainEnv')
register(id='MiniGrid-LightEnableDelay1ARoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelay1AEnv')
register(id='MiniGrid-LightEnableDelayIndepRoom-v0', entry_point='gym_minigrid.envs:LightEnableRoomDelayIndepEnv')
