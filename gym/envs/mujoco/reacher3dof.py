import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET
import os
# import gym.envs.mujoco.arm_shaping
import scipy.misc
from math import acos
class ReacherEnv3DOF(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        #self.randomize_xml('3link_gripper_reach_2d.xml')
        #mujoco_env.MujocoEnv.__init__(self, 'temp.xml', 5)
        mujoco_env.MujocoEnv.__init__(self, '3link_gripper_reach_2d.xml', 5)

    def step(self, a):
        parm = self.get_body_com("distal_4")
        pgoal = self.get_body_com("goal")
        reward_dist = - np.linalg.norm(parm-pgoal)
        self.do_simulation(a, self.frame_skip)
        
        parm = self.get_body_com("distal_4")
        reward_dist = - np.linalg.norm(parm-pgoal)

        ob = self._get_obs()
        done = False
        if not hasattr(self, 'itr'):
            self.itr = 0

        if not hasattr(self, 'np_random'):
            self.seed()

        reward_true = 0
        if self.itr == 0:
            #self.reward_orig = -reward_dist
            pgoal = self.get_body_com("goal")
            parm1 = self.get_body_com("distal_4")
            parm2 = self.get_body_com("distal_2")
            r1 =  parm1-parm2
            r2 = pgoal-parm2
            d1 = np.linalg.norm(r1)
            d2 = np.linalg.norm(r2)
            r1r2 = map(lambda e,f:e*f, r1,r2)
            theta1 = acos(np.linalg.norm(list(r1r2))/(d1*d2))
            self.difficult = theta1*d2 
        #if self.itr == 49:
         #   reward_true = reward_dist/self.reward_orig
        if  self.itr == 500:
            done = True
        if reward_dist > -0.2:
            done = True
            #print("Success at %i"%self.itr)
            

        img = None
        if self.itr % 2 == 1 and hasattr(self, "_kwargs") \
            and 'imsize' in self._kwargs and self._kwargs['mode'] != 'oracle' and self._kwargs['mode'] != 'tpil':
            img = self.render('rgb_array')
            idims = self._kwargs['imsize']
            img = scipy.misc.imresize(img, idims)

        self.itr += 1
        return ob, reward_dist, done, dict(difficult=self.difficult)#reward_true=reward_true, img=img)

    def viewer_setup(self):
        # self.itr = 0
        self.viewer.cam.trackbodyid=0
        self.viewer.cam.distance = 4.0
        rotation_angle = np.random.uniform(low=0, high=360)
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

    def getcolor(self):
        color = np.random.uniform(low=0, high=1, size=3)
        while np.linalg.norm(color - np.array([1.,0.,0.])) < 0.5:
            color = np.random.uniform(low=0, high=1, size=3)
        return np.concatenate((color, [1.0]))
    def randomize_xml(self, xml_name):
        fullpath = os.path.join(os.path.dirname(__file__), "assets", xml_name)
        newpath = os.path.join(os.path.dirname(__file__), "assets", "temp.xml")
        tree = ET.parse(fullpath)
        root = tree.getroot()
        worldbody = tree.find(".//worldbody")
        num_objects = int(np.random.uniform(low=0, high=6, size=1))
        print("num_objs = %f"%num_objects)
        #num_objects = 6
        for object_to_spawn in range(num_objects):
            #print("--------------------------------------")
            pos_x = np.random.uniform(low=-0.9, high=0.9, size=1)
            pos_y = np.random.uniform(low=0, high=1.0, size=1)
            rgba_colors = self.getcolor()
            ET.SubElement(
                worldbody, "geom",
                pos="%f %f -0.145"%(pos_x, pos_y),
                rgba="%f %f %f 1"%(rgba_colors[0], rgba_colors[1], rgba_colors[2]),
                name="object" + str(object_to_spawn),
                size="0.1 0.1 0.1",
                density='0.00001',
                type="cylinder",
                contype="0",
                conaffinity="0"

            )
        print("writing")
        tree.write(newpath)        

    def reset_model(self):
        
        self.itr = 0
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        self.object = np.array([0.0, 0.0])
        rgbatmp = np.copy(self.model.geom_rgba)
        bgcolor = self.getcolor()
        if hasattr(self, "_kwargs") and 'bgcolor' in self._kwargs:
            bgcolor = np.array(self._kwargs['bgcolor'])
        armcolor = self.getcolor()
        while np.linalg.norm(bgcolor - armcolor) < 0.5:
            armcolor = np.concatenate((np.random.uniform(low=0, high=1, size=3), [1.0]))
        if hasattr(self, "_kwargs") and 'armcolor' in self._kwargs:
            armcolor = np.array(self._kwargs['armcolor'])
        rgbatmp[0, :] = bgcolor
        for k in range(2, 9):
            rgbatmp[-k, :] = armcolor

        geompostemp = np.copy(self.model.geom_pos)
        for body in range(len(self.model.geom_names)):
            if 'object' in str(self.model.geom_names[body]):
                pos_x = np.random.uniform(low=-0.9, high=0.9)
                pos_y = np.random.uniform(low=0, high=1.0)
                rgba = self.getcolor()
                isinv = np.random.random()
                if isinv>0.5:
                    rgba[-1] = 0.
                rgbatmp[body, :] = rgba
                geompostemp[body, 0] = pos_x
                geompostemp[body, 1] = pos_y

        if hasattr(self, "_kwargs") and 'geoms' in self._kwargs:
            geoms = self._kwargs['geoms']
            ct = 0
            for body in range(len(geompostemp)):
                if 'object' in str(self.model.geom_names[body]):
                    rgbatmp[body, :] = geoms[ct][0]
                    geompostemp[body, 0] = geoms[ct][1]
                    geompostemp[body, 1] = geoms[ct][2]
                    ct += 1

        self.model.geom_rgba[:,:] = rgbatmp[:,:]
        self.model.geom_pos[:,:] = geompostemp[:,:]

        if hasattr(self, "_kwargs") and 'goal' in self._kwargs:
            self.goal = np.array(self._kwargs['goal'])
        else:
            self.goal = np.concatenate([np.random.uniform(low=-1.1, high=-0.5, size=1),
                 np.random.uniform(low=0.5, high=1.1, size=1)])

        # qpos[-4:-2] = self.object
        qpos[-2:] = self.goal
        qvel = self.init_qvel
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if not hasattr(self, 'np_random'):
            self.seed()
        if not hasattr(self, 'object'):
            self.reset_model()
        if hasattr(self, "_kwargs") and 'mode' in self._kwargs \
            and (self._kwargs['mode'] == 'tpil' or self._kwargs['mode'] == 'inceptionsame'):
            return np.concatenate([
                self.model.data.qpos.flat[:-4],
                self.model.data.qvel.flat[:-4],
                self.get_body_com("distal_4"),
                self.get_body_com("goal"),
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat[:-4],
                self.sim.data.qvel.flat[:-4],
            ])
