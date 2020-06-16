#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from garage.experiment import LocalRunner, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.envs import TfEnv
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
import numpy as np
import subprocess



def rand_push():
    vp = np.random.uniform(low=0, high=360)
    return dict(nvp=1, vp=vp, imsize=(48, 48), taskname="push", modelname='/home/dell/garage/examples/model/ctxskipstartgoalvpdistract10000',
        meanfile=None, modeldata='/home/dell/garage/examples/model/pusher24.npy')

push_params = {
    "env" : "Pusher3DOF-v1",
    "rand" : rand_push,
}    
ours_mode = dict(mode='ours', mode2='ours', scale=0.01)

randparams = push_params['rand']()
copyparams = randparams.copy()
copyparams.update(ours_mode)


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters

    """
    env = TfEnv(env_name='Pusher3DOF-v1')

    runner = LocalRunner(snapshot_config)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=49,
                discount=0.99,
                center_adv=False,
                max_kl_step=0.005,
                **copyparams
                )

    #runner.setup(algo, env)
    #runner.train(n_epochs=100, batch_size=50*250)
    runner.restore("/home/dell/garage/data/local/pusher/pusher_2020_06_01_23_45_24_0001")
    runner.resume(n_epochs=800)


run_experiment(
    run_task,
    exp_prefix="pusher",
    snapshot_mode='last',
    seed=1,
)
