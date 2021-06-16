#!/usr/bin/env python3
import sys
import time
import argparse

import cv2
import numpy as np
import gym_minigrid
import gym
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def effect_to_objects(effect):
    print(effect.shape, effect.max())
    color_sum = (effect * 255.).abs().sum(dim=(1, 2))

    if color_sum.sum() == 0:
        return 'nothing'
    elif 0 == color_sum[0] and 4000 <= color_sum[1] <= 5000 and color_sum[2] <= 200:
        # Press
        return 'button_1'
    elif 2000. <= color_sum[0] <= 3000 and 2000 <= color_sum[1] <= 3000 and 2000 <= color_sum[2] <= 3000:
        # Enable
        return 'button_2'
    elif 4000 <= color_sum[0] <= 5000 and 4000 <= color_sum[1] <= 5000 and color_sum[2] == 0:
        # Press
        return 'button_2'
    elif 2000 <= color_sum[0] <= 3000 and 3000 <= color_sum[1] <= 4000 and color_sum[2] <= 1000:
        # Press
        return 'button_3'
    elif color_sum[0] <= 200 and 3000 <= color_sum[1] <= 4000 and 1000 <= color_sum[2] <= 2000:
        # Enable
        return 'button_3'
    elif 5000 <= color_sum[0] <= 6000 and color_sum[1] == 0 and 5000 <= color_sum[2] <= 6000:
        return 'light 1'
    elif color_sum[0] == 0 and 3000 <= color_sum[1] <= 4000 and 5000 <= color_sum[2] <= 6000:
        return 'light 2'
    elif color_sum[0] == 0 and color_sum[1] == 0 and 4000 <= color_sum[2] <= 6000:
        return 'light 3'
    elif color_sum[0] > 3000 and color_sum[1] == 0 and color_sum[2] == 0:
        return 'agent'
    else:
        print('unknown object:', color_sum.cpu().numpy().tolist())
        return 'unknown'


prev_obs = None
np.set_printoptions(threshold=sys.maxsize)
def step(action):
    global prev_obs
    obs, reward, done, info = env.step(action)
    print('step={}, reward={:.2f}, events={}'.format(env.step_count, reward, info['events']))
    if prev_obs is not None:
        import torch
        effect = torch.from_numpy(obs - prev_obs).float()
        print(np.abs(effect).sum(axis=(0,1)))
        print(effect.abs().sum(dim=(0, 1)))
        print(effect_to_objects(effect.permute(2, 0, 1) / 255.))

    prev_obs = obs

    if done:
        print('done!')
        redraw(obs)
        time.sleep(10)
        prev_obs = reset()
    else:
        redraw(obs)


def key_handler(event):
    print('pressed ({})'.format(event.key))

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )
    parser.add_argument(
        '--agent_view',
        default=False,
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )

    args = parser.parse_args()

    env = gym.make(args.env)
    # env = StateWrapper(env)
    env = ImgObsWrapper(RGBImgObsWrapper(env))
    print('start', args)

    if args.agent_view:
        env = ImgObsWrapper(RGBImgObsWrapper(env))

    window = Window('gym_minigrid - ' + args.env)
    window.reg_key_handler(key_handler)

    reset()

    # Blocking event loop
    window.show(block=True)
