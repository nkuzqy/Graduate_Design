import numpy as np
from garage.envs import normalize
from garage.tf.policies import GaussianMLPPolicy
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
import gym
from garage.experiment import run_experiment
from garage.tf.experiment import LocalTFRunner
from garage.tf.envs import TfEnv
def rand_push():
    vp = np.random.uniform(low=0, high=360)
    return dict(nvp=1, vp=vp, imsize=(48, 48), taskname="push", modelname='/home/dell/garage/examples/model/ctxskipstartgoalvpdistract10000',
        meanfile=None, modeldata='/home/dell/garage/examples/model/vdata_push_full_distract3244.npy')

push_params = {
    "env" : "Pusher3DOF-v1",
    "rand" : rand_push,
}    
ours_mode = dict(mode='ours', mode2='ours', scale=1.0)

randparams = push_params['rand']()
copyparams = randparams.copy()
copyparams.update(ours_mode)
def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(normalize(gym.make('Pusher3DOF-v1')))

        policy = GaussianMLPPolicy(
                                    env_spec=env.spec,
                                    hidden_sizes=(32, 32),
                                    init_std=10
                                )

        baseline = LinearFeatureBaseline(env_spec=env.spec)


        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            #whole_paths=True,
            max_path_length=50,
            max_kl_step=0.01,
            **copyparams
        )
        runner.setup(algo, env)
        runner.train(n_epochs=200, batch_size=50*250)


run_experiment(
    run_task,
    exp_prefix="imi_push",
    #n_parallel=4,
    # dry=True,
    snapshot_mode="all",
    seed=1,
    # mode="ec2_mujoco",
    # terminate_machine=False
)