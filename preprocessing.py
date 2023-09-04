import json
from glob import glob
import os
import h5py
import numpy as np
from calculate_mc_rewards import create_mc_rewards

directory = os.listdir('train')

# data_path = glob(directory)

task_id = []


total_action_sequence_list = []
total_obs_list = []
total_next_obs_list = []
total_terminals_list = []
total_terminals_obs_list = []
total_reward_list = []
total_mc_reward_list = []
discount_factor = 0.99
total_terminals_discounts_list = []
terminal_obs = None
train_counts = 0
total_episode_list = []

for t, dir_name in enumerate(directory[:10]):
    data_path = glob('./train/'+dir_name+'/*')
    temp_rewards_list = []
    temp_teminals = []
    temp_discounts = []
    for i, file in enumerate(list(reversed(data_path))):
        with open(file, 'r') as f:
            data = json.load(f)
        if i == 0:
            terminal_obs = data['action_sequence']['action_sequence'][-1]['grid']
        action = 'submit' if data['action_sequence']['action_sequence'][-1]['action']['tool'] == 'end' else data['action_sequence']['action_sequence'][-1]['action']['tool']
        grid = data['action_sequence']['action_sequence'][-2]['grid']
        next_grid = data['action_sequence']['action_sequence'][-1]['grid']
        reward = data['action_sequence']['action_sequence'][-1]['reward']
        terminal = True if data['action_sequence']['action_sequence'][-1]['action']['tool'] == 'end' else False

        total_reward_list.append(reward)
        total_terminals_list.append(terminal)
        total_terminals_discounts_list.append(0.99)
        total_action_sequence_list.append(action)
        total_terminals_obs_list.append(total_obs_list)
        total_obs_list.append(grid)
        total_next_obs_list.append(next_grid)
        if data['action_sequence']['action_sequence'][-2]['action']['tool'] == 'start':
            total_episode_list.append(i+1)

total_mc_reward_list = create_mc_rewards(total_reward_list, total_terminals_list, 0.99)

with h5py.File('Test_Dflip.hdf5', 'w') as f:
    f.create_dataset('obs', data= total_obs_list)
    f.create_dataset('next_obs', data = total_obs_list)
    f.create_dataset('terminal_obs', data = total_terminals_obs_list)
    f.create_dataset('terminals', data = total_terminals_list)
    f.create_dataset('actions', data = total_action_sequence_list)
    f.create_dataset('rewards', data = total_reward_list)
    f.create_dataset('mc_rewards', data=total_mc_reward_list)
    f.create_dataset('discount_factor', data = 0.99)
    f.create_dataset('terminal_discounts', data = total_terminals_discounts_list)

print(f'total_trace: {t+1}')
print(f'mean_episode_length: {sum(total_episode_list)/len(total_episode_list)}')