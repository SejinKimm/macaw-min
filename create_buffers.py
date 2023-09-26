import h5py
import pickle
import numpy as np
from typing import Optional, Tuple, List
from collections import defaultdict
from arcle.loaders import ARCLoader, Loader, MiniARCLoader
import gymnasium as gym


arcenv = gym.make('ARCLE/O2ARCv2Env-v0',render_mode=None, data_loader= ARCLoader(), max_grid_size=(30,30), colors = 10, max_episode_steps=None)
minienv = gym.make('ARCLE/O2ARCv2Env-v0',render_mode=None, data_loader=MiniARCLoader(), max_grid_size=(30,30), colors = 10, max_episode_steps=None)

failure_trace = []

def findbyname(name):
    for i, aa in enumerate(ARCLoader().data):
        if aa[4]['id'] == name:
            return i
    for i, aa in enumerate(MiniARCLoader().data):
        if aa[4]['id'] == name:
            return i

def action_convert(action_entry):
    _, action, data, grid = action_entry
    op = 0
    # print(action, data)
    match action:
        case "CopyFromInput":
            op = 31
        case "ResizeGrid":
            op = 33
        case "ResetGrid":
            op = 32
        case "Submit":
            op = 34
        case "Color":
            op = data[1]
        case "Fill":
            op = data[2]
        case "FlipX":
            op = 27
        case "FlipY":
            op = 26
        case "RotateCW":
            op = 25
        case "RotateCCW":
            op = 24
        case "Move":
            match data[2]:
                case 'U':
                    op = 20
                case 'D':
                    op = 21
                case 'R':
                    op = 22
                case 'L':
                    op = 23
        case "Copy":
            match data[2]:
                case 'Input Grid':
                    op = 28
                case 'Output Grid':
                    op = 29
        case "Paste":
            op = 30
        case "FloodFill":
            op = 10 + data[1]

    return op

def create_features(task_dict):

    discount_factor = 0.99
    file_no = 0
    for task in task_dict.keys():
        id_list, obs_init_list = zip(*task_dict[task])
        task_no, subtask_no = task.split('_')

        cnt = sum([len(traces[id]) - 1 for id in id_list])
        obs = np.zeros(shape=(cnt, 30, 30))
        next_obs = np.zeros(shape=(cnt, 30, 30))
        terminal_obs = np.zeros(shape=(cnt, 30, 30))
        terminals = np.zeros(shape=(cnt, 1), dtype=bool)
        actions = np.zeros(shape=(cnt, 1))
        rewards = np.zeros(shape=(cnt, 1))
        mc_rewards = np.zeros(shape=(cnt, 1))
        terminal_discounts = np.zeros(shape=(cnt, 1))

        cnt = 0
        for id, obs_init in zip(id_list, obs_init_list):
            # input image 
            obs_first = np.zeros(shape=(30, 30))
            for x in range(obs_init['grid_dim'][0]):
                for y in range(obs_init['grid_dim'][1]):
                    obs_first[x][y] = obs_init['grid'][x][y]

            # output image
            obs_terminal = np.zeros(shape=(30, 30))
            for x in range(traces[id][-1][-1].shape[0]):
                for y in range(traces[id][-1][-1].shape[1]):
                    obs_terminal[x][y] = traces[id][-1][-1][x][y]

            last_actions = len(traces[id]) - 2 # skip commit actions
            obs_after = obs_terminal.copy()
            for i in range(last_actions, -1, -1):
                if i == 0:
                    obs_before = obs_first.copy()
                else:
                    obs_before = np.zeros(shape=(30, 30))
                    for x in range(traces[id][i-1][-1].shape[0]):
                        for y in range(traces[id][i-1][-1].shape[1]):
                            obs_before[x][y] = traces[id][i-1][-1][x][y]

                obs[cnt] = obs_before.copy()
                next_obs[cnt] = obs_after.copy()
                terminal_obs[cnt] = obs_terminal.copy()
                actions[cnt] = action_convert(traces[id][i])

                #print("!!!!!!!!!", traces[id][i][:-1])
                isTerminal = True
                for x in range(30):
                    for y in range(30):
                        if obs[cnt][x][y] != obs_terminal[x][y]:
                            isTerminal = False
                            break
                    if not isTerminal:
                        break

                if isTerminal:
                    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", id, task_no, subtask_no)
                    terminals[cnt] = True
                    rewards[cnt] = 1 # sparse rewards 
                    mc_rewards[cnt] = rewards[cnt]
                    terminal_discounts[cnt] = discount_factor
                else:
                    mc_rewards[cnt] = rewards[cnt] + discount_factor * mc_rewards[cnt - 1]
                    terminal_discounts[cnt] = discount_factor * terminal_discounts[cnt - 1]

                obs_after = obs_before.copy()
                cnt += 1
        
        with open("/home/sjkim/macaw-min/macaw_offline_data/arc/env_arc_train_task%d.pkl" % (file_no), "wb") as f:
            li = [{}]
            li[0]['task_no'] = task_no
            li[0]['subtask_no'] = subtask_no
            pickle.dump(li, f, pickle.HIGHEST_PROTOCOL)

        with h5py.File('/home/sjkim/macaw-min/macaw_offline_data/arc/buffers_arc_train_%d_sub_task_0.hdf5' % (file_no), 'w') as f:
            f.create_dataset('obs', data=obs.reshape(cnt, 900), maxshape = (cnt, 900))
            f.create_dataset('next_obs', data=next_obs.reshape(cnt, 900), maxshape = (cnt, 900))
            f.create_dataset('terminal_obs', data=terminal_obs.reshape(cnt, 900), maxshape = (cnt, 900))
            f.create_dataset('terminals', data=terminals, maxshape = (cnt, 1))
            f.create_dataset('actions', data=actions, maxshape = (cnt, 1))
            
            f.create_dataset('rewards', data=rewards, maxshape = (cnt, 1))
            f.create_dataset('mc_rewards', data=mc_rewards, maxshape = (cnt, 1))
            f.create_dataset('discount_factor', data=discount_factor, maxshape = ())
            f.create_dataset('terminal_discounts', data=terminal_discounts, maxshape = (cnt, 1))
        
        file_no += 1

if __name__ == "__main__":
    traces = []
    traces_info = []
    with open('/home/sjkim/macaw-min/test.pickle', 'rb') as fp:
        traces:List = pickle.load(fp)
    with open('/home/sjkim/macaw-min/test_info.pickle', 'rb') as fp:
        traces_info:List = pickle.load(fp)

    task_dict = defaultdict(list)
    for idx, (trace, info) in enumerate(zip(traces, traces_info)):
        name, subtask, isGoal = info
        name += "_" + str(subtask)

        if len(traces_info[idx][0]) >10:
            env = minienv
        else:
            env = arcenv
        obs_init, _ = env.reset(options= {'adaptation':False, 'prob_index':findbyname(traces_info[idx][0]), 'subprob_index': traces_info[idx][1]})
        
        task_dict[name].append((idx, obs_init))


    create_features(task_dict)

