#!/usr/bin/env python3
"""Simulates pre-learned policy."""
import argparse
import sys

import joblib
import tensorflow as tf

from garage.sampler.utils import rollout
import gym
from gym import wrappers


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,default='/home/dell/garage/examples/data/local/trpo-pusher-200-dist-ctrl-touch/trpo_pusher_200_dist_ctrl_touch_2020_04_17_14_26_48_0001/params.pkl', help='path to the snapshot file')
    parser.add_argument('--max_path_length',
                        type=int,
                        default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=20, help='Speedup')
    args = parser.parse_args()
    with tf.compat.v1.Session() as sess:
        data = joblib.load(args.file)
        policy = data['algo'].policy
        #env = data['env']
        env = gym.make('Pusher3DOF-v1')
        env =  wrappers.Monitor(env, "recording/pusher3dof_diff",resume=True)
        for epi in range(500):
            #env = gym.make('Pusher3DOF-v1')
            #env =  wrappers.Monitor(env, "recording/pusher3dof_diff",resume=True)
            obs = env.reset()
            #if epi%100 == 0:
                #print("episode %i"%epi)
            step = 0
            while(1):
            #action = env.action_space.sample()
            # print("This is step %i"%i)
                action, _= policy.get_action(obs)
                obs, reward, done, info = env.step(action)
                if step == 0:
                    print(info)
                if done: 
                    #print("done at step %i"%i)
                    #print("done")
                    break
                step += 1

    
    ##env.close()
