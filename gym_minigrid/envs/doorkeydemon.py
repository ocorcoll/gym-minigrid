from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.envs.effectsroom import Demon


class DoorKeyDemonEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size: int = 8, num_demons: int = 1):
        self.num_demons = num_demons
        self.demons = None

        super().__init__(
            grid_size=size,
            max_steps=10*size*size,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place Demon
        self.demons = list()
        for _ in range(self.num_demons):
            demon = Demon(self, 'horizontal')
            self.demons.append(demon)
            self.place_obj(obj=demon, top=(splitIdx, 0), size=(splitIdx, height))

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        _, reward, done, info = super().step(action)

        for demon in self.demons:
            demon.move()

        obs = self.gen_obs()
        return obs, reward, done, info


class DoorKeyDemonEnv5x5(DoorKeyDemonEnv):
    def __init__(self):
        super().__init__(size=5)

class DoorKeyDemonEnv6x6(DoorKeyDemonEnv):
    def __init__(self):
        super().__init__(size=6)

class DoorKeyDemonEnv16x16(DoorKeyDemonEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-DoorKeyDemon-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyDemonEnv5x5'
)

register(
    id='MiniGrid-DoorKeyDemon-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyDemonEnv6x6'
)

register(
    id='MiniGrid-DoorKeyDemon-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyDemonEnv'
)

register(
    id='MiniGrid-DoorKeyDemon-16x16-v0',
    entry_point='gym_minigrid.envs:DoorKeyDemonEnv16x16'
)
