#!/usr/bin/env python3
"""
This is an example to train a task with VPG algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
from garage.experiment import run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(env_name='Pusher3DOF-v1')

        policy = GaussianMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32),
                                      init_std=10)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=100,
                    discount=0.99,
                    max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=200, batch_size=50*250)


run_experiment(
    run_task,
    exp_prefix="trpo_pusher_200_0524",
    snapshot_mode='last',
    seed=1,
)
