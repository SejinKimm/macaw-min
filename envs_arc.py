import numpy as np
from typing import Optional, Tuple, List
from copy import deepcopy
import gymnasium as gym
from arcle.loaders import ARCLoader, Loader, MiniARCLoader

class ArcEnv(gym.Env):
    def __init__(self, traces: List, traces_info: List, include_goal: bool = False):
        self.include_goal = include_goal
        super(ArcEnv, self).__init__()
        # if tasks is None:
        #     tasks = [{'direction': 1}, {'direction': -1}]
        self.arcloader = ARCLoader()
        self.miniarcloader = MiniARCLoader()
        self.arcenv = gym.make('ARCLE/O2ARCv2Env-v0', render_mode='ansi',data_loader=self.arcloader, max_grid_size=(30,30), colors=10, max_episode_steps=None)
        self.miniarcenv = gym.make('ARCLE/O2ARCv2Env-v0', render_mode='ansi', data_loader=self.miniarcloader, max_grid_size=(30,30), colors=10, max_episode_steps=None)
        self.traces = traces
        self.traces_info = traces_info
        #self.set_task_idx(0)
        self._max_episode_steps = 200
        self.traces = []
        
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[idx] = 1.0 # one_hot = [0, 0, ..., 1, 0, ... 0] (one_hot[idx] = 1)
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot]) # obs += one_hot
        else:
            obs = super()._get_obs()
        return obs
    
    # def set_task(self, task):
    #     self._task = task
    #     self._goal_dir = self._task['direction']
    #     self.reset()

    # def set_task_idx(self, idx):
    #     self.set_task(self.tasks[idx])