import pickle
import h5py
import numpy as np

# with open("/home/sjkim/macaw/macaw_offline_data/cheetah_dir/env_cheetah_dir_train_task0.pkl", "rb") as f:
#     print(pickle.load(f))
    

with h5py.File("/home/sjkim/macaw-min/macaw_offline_data/arc/buffers_arc_train_0ca9ddb6_sub_task_0.hdf5", "r") as f:
    print("Keys: %s" % f.keys())

    for key in f.keys():
        arr = np.array(f[key])
        print(key, "with shape:", arr.shape)
        if arr.ndim == 0:
            print(arr[()])
        else:
            print(arr)

    rewards = np.array(f['rewards'])
    mc_rewards = np.array(f['mc_rewards'])
    start_idx, end_idx = 0, -1
    if start_idx == 0:
        rewards_sum = 0
    else:
        rewards_sum = mc_rewards[start_idx - 1]
    print("<<< rewards vs mc_rewards >>>")
    for i in range(start_idx, end_idx):
        print(rewards[i], "+ 0.99 *", rewards_sum, "=", rewards[i] + 0.99 * rewards_sum, "vs", mc_rewards[i])
        rewards_sum = rewards[i] + 0.99 * rewards_sum


    # obs = np.array(f['obs'])
    # next_obs = np.array(f['next_obs'])
    # start_idx, end_idx = 0, 5
    # print("\n\n<<< obs vs next_obs >>>")
    # for i in range(start_idx, end_idx):
    #     print("obs[%d]" % (i))
    #     print(obs[i])
    #     print("next_obs[%d]" % (i))
    #     print(next_obs[i])
    #     print()
    #     print()
