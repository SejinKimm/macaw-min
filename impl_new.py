from nn import MLP
from utils import ReplayBuffer
import hydra
from hydra.utils import get_original_cwd
import json
from collections import namedtuple
import pickle
import torch
import torch.optim as O
from typing import List
import higher
from itertools import count
import logging
from utils import Experience
from losses import policy_loss_on_batch, vf_loss_on_batch
from envs_arc import ArcEnv
from arcle.loaders import ARCLoader, Loader, MiniARCLoader
import gymnasium as gym
import numpy as np

LOG = logging.getLogger(__name__)


def rollout_policy(policy: MLP, env, render: bool = False) -> List[Experience]:
    trajectory = []
    idx = env.get_idx()
    # print("!!!!!!!!!!!!!ROLLOUT POLICY IDX:", idx)
    state = env.arcenv.reset(options= {'adaptation':False, 'prob_index':env.findbyname(env.traces_info[idx][0]), 'subprob_index': env.traces_info[idx][1]})
    if render:
        env.arcenv.render()
    done = False
    total_reward = 0
    episode_t = 0
    success = False
    policy.eval()
    current_device = list(policy.parameters())[-1].device
    while not done:
        with torch.no_grad():
            if len(state) == 2:
                action = policy(torch.tensor(state[0]['selected'].reshape(1, -1)).to(current_device).float()).squeeze()
            else:
                action = policy(torch.tensor(state['selected'].reshape(1, -1)).to(current_device).float()).squeeze()
            np_action = action.squeeze().cpu().numpy()
            # print(np_action)
            try:
                np_action = int(np.interp(np_action, (-1, 1), (1, 34)))
            except:
                np_action = 1

        action = dict()
        action['operation'] = np_action
        action['selection'] = np.zeros((30,30), dtype=np.bool_)

        next_state, reward, done, _, info_dict = env.arcenv.step(action)

        if "success" in info_dict and info_dict["success"]:
            success = True

        if render:
            env.arcenv.render()
        trajectory.append(Experience(state, np_action, next_state, reward, done))
        state = next_state
        total_reward += reward
        episode_t += 1
        if episode_t >= env._max_episode_steps or done:
            break

    return trajectory, total_reward, success


def build_networks_and_buffers(args, env, task_config):
    obs_dim = 900
    action_dim = 1

    policy_head = [32, 1] if args.advantage_head_coef is not None else None
    policy = MLP(
        [obs_dim] + [args.net_width] * args.net_depth + [action_dim],
        final_activation=torch.tanh,
        extra_head_layers=policy_head,
        w_linear=args.weight_transform,
    ).to(args.device)

    vf = MLP(
        [obs_dim] + [args.net_width] * args.net_depth + [1],
        w_linear=args.weight_transform,
    ).to(args.device)

    s, e = map(int, task_config.train_tasks)
    train_buffer_paths = [
        (idx, task_config.train_buffer_paths.format(idx)) for idx in range(s, e)
    ]

    train_buffers = [
        (idx, ReplayBuffer(
            args.inner_buffer_size,
            obs_dim,
            action_dim,
            discount_factor=0.99,
            immutable=True,
            load_from=train_buffer,
        ))
        for idx, train_buffer in train_buffer_paths
    ]
    
    s, e = map(int, task_config.test_tasks)
    test_buffer_paths = [
        (idx, task_config.test_buffer_paths.format(idx)) for idx in range(s, e)
    ]

    test_buffers = [
        (idx, ReplayBuffer(
            args.inner_buffer_size,
            obs_dim,
            action_dim,
            discount_factor=0.99,
            immutable=True,
            load_from=test_buffer,
        ))
        for idx, test_buffer in test_buffer_paths
    ]

    return policy, vf, train_buffers, test_buffers

def get_env(args, task_config):
    traces = []
    traces_info = []
    # for task_idx in range(task_config.total_tasks):
    #     with open(task_config.task_paths.format(task_idx), "rb") as f:
    #         task_info = pickle.load(f)
    #         assert len(task_info) == 1, f"Unexpected task info: {task_info}"
    #         tasks.append(task_info[0])
    # if args.advantage_head_coef == 0:
    #     args.advantage_head_coef = None  
    with open(task_config.traces, 'rb') as fp:
        traces:List = pickle.load(fp)
    with open(task_config.traces_info, 'rb') as fp:
        traces_info:List = pickle.load(fp)


    return ArcEnv(traces=traces, traces_info=traces_info, include_goal=True)


def get_opts_and_lrs(args, policy, vf):
    policy_opt = O.Adam(policy.parameters(), lr=args.outer_policy_lr)
    vf_opt = O.Adam(vf.parameters(), lr=args.outer_value_lr)
    policy_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_policy_lr).to(args.device))
        for p in policy.parameters()
    ]
    vf_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_value_lr).to(args.device))
        for p in vf.parameters()
    ]

    return policy_opt, vf_opt, policy_lrs, vf_lrs


@hydra.main(config_path="config", config_name="config.yaml")
def run(args):
    with open(f"{get_original_cwd()}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )
    env = get_env(args, task_config)

    policy, vf, train_task_buffers, test_task_buffers = build_networks_and_buffers(args, env, task_config)
    policy_opt, vf_opt, policy_lrs, vf_lrs = get_opts_and_lrs(args, policy, vf)

    for train_step_idx in count(start=1):
        if train_step_idx % args.rollout_interval == 0:
            LOG.info(f"Train step {train_step_idx}")

        # import pdb; pdb.set_trace()
        for train_task_idx, task_buffers in train_task_buffers:
            env.set_task_idx(train_task_idx)
            # print("!!!!!!!!!!!!!TRAIN TASK IDX:", train_task_idx)

            inner_batch = task_buffers.sample(
                args.inner_batch_size, return_dict=True, device=args.device
            )
            outer_batch = task_buffers.sample(
                args.outer_batch_size, return_dict=True, device=args.device
            )

            # Adapt value function
            opt = O.SGD([{"params": p, "lr": None} for p in vf.parameters()])
            with higher.innerloop_ctx(
                vf, opt, override={"lr": vf_lrs}, copy_initial_weights=False
            ) as (f_vf, diff_value_opt):
                loss = vf_loss_on_batch(f_vf, inner_batch, inner=True)
                diff_value_opt.step(loss)

                meta_vf_loss = vf_loss_on_batch(f_vf, outer_batch)
                total_vf_loss = meta_vf_loss / len(task_config.train_tasks)
                total_vf_loss.backward()

            # Adapt policy using adapted value function
            adapted_vf = f_vf
            opt = O.SGD([{"params": p, "lr": None} for p in policy.parameters()])
            with higher.innerloop_ctx(
                policy, opt, override={"lr": policy_lrs}, copy_initial_weights=False
            ) as (f_policy, diff_policy_opt):
                loss = policy_loss_on_batch(
                    f_policy,
                    adapted_vf,
                    inner_batch,
                    args.advantage_head_coef,
                    inner=True,
                )

                diff_policy_opt.step(loss)
                meta_policy_loss = policy_loss_on_batch(
                    f_policy, adapted_vf, outer_batch, args.advantage_head_coef
                )

                (meta_policy_loss / len(task_config.train_tasks)).backward()

                # Sample adapted policy trajectory
                # if train_step_idx % args.rollout_interval == 0:
                #     adapted_trajectory, adapted_reward, success = rollout_policy(f_policy, env)
                #     LOG.info(f"Task {train_task_idx} reward: {adapted_reward}")

        # Update the policy/value function
        policy_opt.step()
        policy_opt.zero_grad()
        vf_opt.step()
        vf_opt.zero_grad()

        if train_step_idx % args.rollout_interval == 0:
            for test_task_idx, task_buffers in test_task_buffers:
                env.set_task_idx(test_task_idx)
                adapted_trajectory, adapted_reward, success = rollout_policy(policy, env, True)
                LOG.info(f"Task {test_task_idx} reward: {adapted_reward}")

if __name__ == "__main__":
    run()