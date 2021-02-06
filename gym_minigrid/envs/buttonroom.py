from itertools import chain

from gym_minigrid.roomgrid import RoomGrid, WorldObj, fill_coords, point_in_circle, point_in_rect, COLORS
from gym_minigrid.register import register
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX


class Button(WorldObj):

    def __init__(self, name, lights, color, delay=3):
        super().__init__('button', color)
        self.name = name
        self.color = color
        self.lights = lights
        self.delay = delay
        self.is_on = False
        self.steps = 0

    def render(self, img):
        if self.is_on:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
            fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS['white'])
        else:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

    def toggle(self, env, pos):
        if self.steps > 0:  # or max([light.steps for light in self.lights]) > 0:
            return

        self.steps = 2
        for light in self.lights:
            if light.steps == 0:
                light.steps = self.delay

        return True

    def update(self):
        self.is_on = not self.is_on if self.steps == 1 else self.is_on
        self.steps = max(0, self.steps - 1)

    def encode(self):
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.is_on


class Light(WorldObj):

    def __init__(self, name, color):
        super().__init__('light', color)
        self.name = name
        self.color = color
        self.is_on = False
        self.steps = 0

    def render(self, img):
        if self.is_on:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.51), COLORS[self.color])
            fill_coords(img, point_in_circle(0.48, 0.48, 0.31), COLORS['white'])
        else:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.51), COLORS[self.color])

    def update(self):
        self.is_on = not self.is_on if self.steps == 1 else self.is_on
        self.steps = max(0, self.steps - 1)

    def encode(self):
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.is_on


class ButtonRoom(RoomGrid):

    def __init__(self, num_buttons, delay, seed=None):
        self.num_buttons = num_buttons
        self.delay = delay
        self.colors = ['blue', 'yellow', 'pink', 'purple', 'orange', 'brown']
        assert num_buttons < len(self.colors)

        self.buttons = None
        self.lights = None

        self.room_size = 5
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

        self.buttons = list()
        self.lights = list()

        for i, color in zip(range(self.num_buttons), self.colors):
            light = Light(str(i), color)
            self.lights.append(light)
            self.place_obj(light, (1, 1), (1, 1))

            button = Button(str(i), [light], color, delay=self.delay)
            self.buttons.append(button)
            self.place_obj(button, (1, 2), (1, 1))

        self.place_agent(0, 0)
        self.mission = 'do nothing'

    def step(self, action):
        # Freeze agent until buttons and lights stabilize
        if max([item.steps for item in chain(self.buttons, self.lights)]) != 0:
            action = 6

        obs, reward, done, info = super().step(action)

        for button in self.buttons:
            button.update()

        for light in self.lights:
            light.update()

        obs = self.gen_obs()
        return obs, reward, done, info


class ButtonRoomB1D3Env(ButtonRoom):

    def __init__(self, seed=None):
        super().__init__(num_buttons=1, delay=3, seed=seed)


class ButtonRoomB1D5Env(ButtonRoom):

    def __init__(self, seed=None):
        super().__init__(num_buttons=1, delay=5, seed=seed)


class ButtonRoomB1D9Env(ButtonRoom):

    def __init__(self, seed=None):
        super().__init__(num_buttons=1, delay=9, seed=seed)


register(id='MiniGrid-ButtonRoom-b1d3-v0', entry_point='gym_minigrid.envs:ButtonRoomB1D3Env')
register(id='MiniGrid-ButtonRoom-b1d5-v0', entry_point='gym_minigrid.envs:ButtonRoomB1D5Env')
register(id='MiniGrid-ButtonRoom-b1d9-v0', entry_point='gym_minigrid.envs:ButtonRoomB1D9Env')
