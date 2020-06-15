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
    parser.add_argument('file', type=str,default='', help='path to the snapshot file',)
    parser.add_argument('--max_path_length',
                        type=int,
                        default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1, help='Speedup')
    args = parser.parse_args()
    with tf.compat.v1.Session() as sess:
        data = joblib.load(args.file)
        policy = data['algo'].policy
        #env = data['env']
        env = gym.make('Pusher3DOF-v1')
        env =  wrappers.Monitor(env, "recording/51",resume=True)
        obs = env.reset()
        for i in range(51):
            action, _= policy.get_action(obs)
            obs, reward, done, info = env.step(action)
    env.close()
