import numpy as np
from typing import Optional, Tuple, List
from copy import deepcopy
import gymnasium as gym
from arcle.loaders import ARCLoader, Loader, MiniARCLoader

class ArcEnv(gym.Env):
    def __init__(self, traces: List, traces_info: List, include_goal: bool = False):
        self.include_goal = include_goal
        super(ArcEnv, self).__init__()
        self.arcloader = ARCLoader()
        self.miniarcloader = MiniARCLoader()
        self.arcenv = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=None, data_loader=self.arcloader, max_grid_size=(30,30), colors=10, max_episode_steps=None)
        self.miniarcenv = gym.make('ARCLE/O2ARCv2Env-v0', render_mode=None, data_loader=self.miniarcloader, max_grid_size=(30,30), colors=10, max_episode_steps=None)
        self.env = self.arcenv
        self.traces = traces
        self.traces_info = traces_info
        self._max_episode_steps = 200
        self.idx = 1
        self._task = None
        
    def _get_obs(self):
        if self.include_goal:
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[self.idx] = 1.0 # one_hot = [0, 0, ..., 1, 0, ... 0] (one_hot[idx] = 1)
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot]) # obs += one_hot
        else:
            obs = super()._get_obs()
        return obs
    
    def get_idx(self):
        return self.idx

    def findbyname(self, name):
        for i, aa in enumerate(self.arcloader.data):
            if aa[4]['id'] == name:
                self.env = self.arcenv
                return i
        for i, aa in enumerate(self.miniarcloader.data):
            if aa[4]['id'] == name:
                self.env = self.miniarcenv
                return i
    
    def set_task(self, task):
        self._task = task
        # self._goal_dir = self._task['direction']
        state = self.env.reset(options= {'adaptation':False, 'prob_index':self.findbyname(self.traces_info[self.idx][0]), 'subprob_index': self.traces_info[self.idx][1]})

    def set_task_idx(self, idx):
        self.idx = idx
        self.findbyname(self.traces_info[self.idx][0])
        self.set_task(self.traces[self.idx])