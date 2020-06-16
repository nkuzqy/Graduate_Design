import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as ET
import os
import scipy.misc
# import gym.envs.mujoco.sweep_shaping
from math import acos
class CleanerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.randomize_xml('cleaning_task.xml')
        mujoco_env.MujocoEnv.__init__(self, 'temp_sweep.xml', 5, viewersize=(72*5, 128*5))
        #mujoco_env.MujocoEnv.__init__(self, 'cleaning_task.xml', 5, viewersize=(72*5, 128*5))
        self.itr = 0

    def step(self, a):  
        self.do_simulation(a, self.frame_skip)
        parm = self.get_body_com("distal_4")
        obj0 = self.get_body_com("object0")
        obj1 = self.get_body_com("object1")
        obj2 = self.get_body_com("object2")
        obj3 = self.get_body_com("object3")
        obj4 = self.get_body_com("object4")
        pgoal = self.get_body_com("goal")
        reward_obj0 = - np.linalg.norm(obj0-pgoal)
        reward_obj1 = - np.linalg.norm(obj1-pgoal)
        reward_obj2 = - np.linalg.norm(obj2-pgoal)
        reward_obj3 = - np.linalg.norm(obj3-pgoal)
        reward_obj4 = - np.linalg.norm(obj4-pgoal)
        reward_touch0 = - np.linalg.norm(parm-obj0)
        reward_touch1 = - np.linalg.norm(parm-obj1)
        reward_touch2 = - np.linalg.norm(parm-obj2)
        reward_touch3 = - np.linalg.norm(parm-obj3)
        reward_touch4 = - np.linalg.norm(parm-obj4)
        #print("dist_array:[%f,%f,%f,%f,%f]"%(reward_obj0,reward_obj1,reward_obj2,reward_obj3,reward_obj))
        diff_xpos = -np.linalg.norm(self.data.site_xpos[0][1] - self.data.site_xpos[1][1])
        reward_ctrl = - np.square(a).sum()
        reward = reward_obj0 + reward_obj1 + reward_obj2 + reward_obj3 + reward_obj4 + reward_touch0 + reward_touch1 + reward_touch2 + reward_touch3 + reward_touch4 + \
                 0.001*reward_ctrl

        if not hasattr(self, 'itr'):
            self.itr = 0
        true_reward =  reward_obj0 + reward_obj1 + reward_obj2 + reward_obj3 + reward_obj4
        if self.itr == 0:
            self.reward_orig = -true_reward
            obj0 = self.get_body_com("object0")
            obj1 = self.get_body_com("object1")
            obj2 = self.get_body_com("object2")
            obj3 = self.get_body_com("object3")
            obj4 = self.get_body_com("object4")
            ob = [obj0,obj1,obj2,obj3,obj4]
            pgoal = self.get_body_com("goal")
            parm = self.get_body_com("distal_4")
            self.difficult = 0
            for i in range(5):
                pobj = ob[i]
                r1 = pobj-parm
                r2 = pgoal-pobj
                d1 = np.linalg.norm(r1)
                d2 = np.linalg.norm(r2)
                r1r2 = map(lambda e,f:e*f, r1,r2)
                theta1 = acos(np.linalg.norm(list(r1r2))/(d1*d2))
                self.difficult += theta1*d2
        #true_reward /= self.reward_orig
        #self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        img = None
        if self.itr % 2 == 1 and hasattr(self, "_kwargs") \
            and 'imsize' in self._kwargs and self._kwargs['mode'] != 'tpil' and self._kwargs['mode'] != 'oracle':
            img = self.render('rgb_array')
            idims = self._kwargs['imsize']
            img = scipy.misc.imresize(img, idims)
        if self.itr != 49:
            true_reward = 0
        
        if self.itr >200:
            done = True
        if reward_obj0 > -0.28 and reward_obj1 > -0.28 and reward_obj3 > -0.28 and reward_obj4 > -0.28 and self.itr >1:
            print("Success at %i"%self.itr)
            done = True
        self.itr += 1
        return ob, reward, done, dict(difficult=self.difficult)#reward_true=true_reward, img=img)

    def randomize_xml(self, xml_name):
        fullpath = os.path.join(os.path.dirname(__file__), "assets", xml_name)
        newpath = os.path.join(os.path.dirname(__file__), "assets", "temp_sweep.xml")
        tree = ET.parse(fullpath)
        root = tree.getroot()
        worldbody = tree.find(".//worldbody")
        objects = ['object0','object1','object2','object3','object4']
        for ob in objects:
            pos_x = np.random.uniform(low=-0.15, high=0.15, size=1)
            pos_y = np.random.uniform(low=-0.15, high=0.15, size=1)
            obj = ET.SubElement(worldbody,"body", attrib={"name":ob,"pos":"%f %f -0.08"%(pos_x, pos_y)})
            geom = ET.SubElement(obj,"geom",attrib={"conaffinity":"1","contype":"1","density":"0.001","rgba":"0.2 0.2 0.2 1", "size":"0.05","type":"sphere"})
            joint = ET.SubElement(obj,"joint",attrib={"name":"%s_slidey"%(ob),"type":"slide","pos":"0.025 0.025 0.025","axis":"0 1 0","range":"-10.3213 10.3","damping":"0.5"})
            joint = ET.SubElement(obj,"joint",attrib={"name":"%s_slidex"%(ob),"type":"slide","pos":"0.025 0.025 0.025","axis":"1 0 0","range":"-10.3213 10.3"})
        tree.write(newpath)

    def viewer_setup(self):
        #print("in viewer_setup")
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
        
    def reset_model(self):        
        self.itr = 0
        ####USER######
        geompostemp = np.copy(self.model.geom_pos)
        for body in range(len(self.model.geom_names)):
            if 'object' in str(self.model.geom_names[body]):
                pos_x = np.random.uniform(low=-0.15, high=0.15, size=1)
                pos_y = np.random.uniform(low=-0.15, high=0.15, size=1)
                
                geompostemp[body, 0] = pos_x
                geompostemp[body, 1] = pos_y
        self.model.geom_pos[:,:] = geompostemp[:,:]
        ##########
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])
