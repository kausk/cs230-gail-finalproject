## Code is from https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/examples/roboschool-weights/enjoy_TF_HopperPyBulletEnv_v0_2017may.py

import gym
import numpy as np
import pybullet as p
import pybulletgym.envs
import time
from stable_baselines.gail import GAIL


def render(envName, modelPath):
    print("create env")
    env = gym.make(envName)
    model = GAIL.load(modelPath)

    env.reset()
    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        #disable rendering during reset, makes loading much faster
        obs = env.reset()
        print("frame")
        while 1:
            time.sleep(0.02)
            a = model.predict(obs)[0]
            
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1
            distance = 5
            yaw = 0
            ## humanPos = p.getLinkState(torsoId,4)[0]
            ## p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

            still_open = env.render("human")
            if still_open is None:
                return
            if not done:
                continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60*2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0:
                    break


