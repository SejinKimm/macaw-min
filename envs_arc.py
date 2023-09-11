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
        

    def findbyname(self, name):
        for i, aa in enumerate(self.arcloader()):
            if aa[4]['id'] == name:
                return i
        for i, aa in enumerate(self.miniarcloader()):
            if aa[4]['id'] == name:
                return i

    def action_convert(self, action_entry):
        _, action, data, grid = action_entry
        sel = np.zeros((30,30), dtype=np.bool_)
        op = 0
        match action:
            case "CopyFromInput":
                op = 31
            case "ResizeGrid":
                op = 33
                h, w = data[0]
                sel[:h,:w] = 1
            case "ResetGrid":
                op = 32
            case "Submit":
                op = 34
            case "Color":
                h, w = data[0]
                op = data[1]
                sel[h,w] = 1

            case "Fill":
                h0, w0 = data[0]
                h1, w1 = data[1]
                op = data[2]
                sel[h0:h1+1 , w0:w1+1] = 1

            case "FlipX":
                h0, w0 = data[0]
                h1, w1 = data[1]
                op = 27
                sel[h0:h1+1, w0:w1+1] = 1
            case "FlipY":
                h0, w0 = data[0]
                h1, w1 = data[1]
                op = 26
                sel[h0:h1+1, w0:w1+1] = 1
            case "RotateCW":
                h0, w0 = data[0]
                h1, w1 = data[1]
                op = 25
                sel[h0:h1+1, w0:w1+1] = 1
            case "RotateCCW":
                h0, w0 = data[0]
                h1, w1 = data[1]
                op = 24
                sel[h0:h1+1, w0:w1+1] = 1
            case "Move":
                h0, w0 = data[0]
                h1, w1 = data[1]
                match data[2]:
                    case 'U':
                        op = 20
                    case 'D':
                        op = 21
                    case 'R':
                        op = 22
                    case 'L':
                        op = 23

                sel[h0:h1+1, w0:w1+1] = 1
            
            case "Copy":
                h0, w0 = data[0]
                h1, w1 = data[1]
                match data[2]:
                    case 'Input Grid':
                        op = 28
                    case 'Output Grid':
                        op = 29
                sel[h0:h1+1, w0:w1+1] = 1
            case "Paste":
                h, w = data[0]
                op = 30
                sel[h,w] = 1

            case "FloodFill":
                h, w = data[0]
                op = 10 + data[1]
                sel[h,w] = 1

        return op, sel

    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
    
    # def set_task(self, task):
    #     self._task = task
    #     self._goal_dir = self._task['direction']
    #     self.reset()

    # def set_task_idx(self, idx):
    #     self.set_task(self.tasks[idx])