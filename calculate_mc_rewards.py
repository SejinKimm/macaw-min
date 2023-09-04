import h5py
import numpy as np

def calculate_mc_rewards(f):
    rewards = f['rewards']
    terminals = f['terminals']
    discount_factor = f['discount_factor'][()]
    mc_rewards = np.copy(rewards)

    for i in range(len(mc_rewards)):
        if terminals[i]:
            mc_rewards[i] = rewards[i]
        else:
            mc_rewards[i] = rewards[i] + discount_factor * mc_rewards[i - 1]

    return mc_rewards

def create_mc_rewards(rewards, terminals, discount_factor):
    mc_rewards = np.copy(rewards)

    for i in range(len(mc_rewards)):
        if terminals[i]:
            mc_rewards[i] = rewards[i]
        else:
            mc_rewards[i] = rewards[i] + discount_factor * mc_rewards[i - 1]

    return mc_rewards

if __name__ == "__main__":
    with h5py.File("/home/woochang/macaw-min/Test_Dflip.hdf5", "r") as f:
        rewards = np.array(f['rewards'])
        real_mc_rewards = np.array(f['mc_rewards'])
        terminals = np.array(f['terminals'])
        calculated_mc_rewards = calculate_mc_rewards(f)

    print("<<< calculated_rewards vs real_mc_rewards >>>")
    cnt = 0
    false_list = []
    for i in range(len(rewards)):
        if terminals[i]:
            cnt += 1
            if calculated_mc_rewards[i] != real_mc_rewards[i]:
                false_list.append(cnt)
        print("!!!!", i, ":", terminals[i], "!!!!", calculated_mc_rewards[i], real_mc_rewards[i], calculated_mc_rewards[i] == real_mc_rewards[i])
    print(cnt)
    if false_list:
        print(false_list)
