import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from math import acos
class StrikerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self._striked = False
        self._min_strike_dist = np.inf
        self.strike_threshold = 0.1
        self.itr = 0
        mujoco_env.MujocoEnv.__init__(self, 'striker.xml', 5)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        self._min_strike_dist = min(self._min_strike_dist, np.linalg.norm(vec_2))
        if np.linalg.norm(vec_1) < self.strike_threshold:
            self._striked = True
            self._strike_pos = self.get_body_com("tips_arm")

        if self._striked:
            vec_3 = self.get_body_com("object") - self._strike_pos
            reward_near = - np.linalg.norm(vec_3)
        else:
            reward_near = - np.linalg.norm(vec_1)

        reward_dist = - self._min_strike_dist
        reward_ctrl = - np.square(a).sum()
        reward = 3 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        #self.do_simulation(a, self.frame_skip)
        
        ob = self._get_obs()
        done = False
        if self.itr == 0:
            pobj = self.get_body_com("object")
            pgoal = self.get_body_com("goal")
            parm1 = self.get_body_com("tips_arm")
            parm2 = self.get_body_com("r_elbow_flex_link")
            r1 =  parm1-parm2
            r2 = pobj-parm2
            r3 = pgoal-pobj
            d1 = np.linalg.norm(r1)
            d2 = np.linalg.norm(r2)
            d3 = np.linalg.norm(r3)
            r1r2 = map(lambda e,f:e*f, r1,r2)
            r2r3 = map(lambda e,f:e*f, r2,r3)
            theta1 = acos(np.linalg.norm(list(r1r2))/(d1*d2))
            theta2 = acos(np.linalg.norm(list(r2r3))/(d2*d3))
            self.difficult = theta1*d2 + theta2*d3 #+ 0.5
        if self.itr>500:
            done = True
        if reward_dist > -0.15 and self.itr > 1:
            done = True
            print("Success at %i"%self.itr)
        self.itr += 1
        return ob, reward, done, dict(difficult=self.difficult)#reward_dist=reward_dist,
                #reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        rotation_angle = np.random.uniform(low=-0, high=360)
        if hasattr(self, "_kwargs") and 'vp' in self._kwargs:
            rotation_angle = self._kwargs['vp']
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid=-1

    def reset_model(self):
        self._min_strike_dist = np.inf
        self._striked = False
        self._strike_pos = None
        self.itr = 0

        qpos = self.init_qpos

        self.ball = np.array([0.5, -0.175])
        while True:
            self.goal = np.concatenate([
                    self.np_random.uniform(low=0.15, high=0.7, size=1),
                    self.np_random.uniform(low=0.1, high=1.0, size=1)])
            if np.linalg.norm(self.ball - self.goal) > 0.17:
                break

        qpos[-9:-7] = [self.ball[1], self.ball[0]]
        qpos[-7:-5] = self.goal
        diff = self.ball - self.goal
        angle = -np.arctan(diff[0] / (diff[1] + 1e-8))
        qpos[-1] = angle / 3.14
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1,
                size=self.model.nv)
        qvel[7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
