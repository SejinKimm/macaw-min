from nn import MLP
from envs import HalfCheetahDirEnv
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
import gym

from utils import Experience
from losses import policy_loss_on_batch, vf_loss_on_batch, qf_loss_on_batch

from torch.utils.tensorboard import SummaryWriter

#test colab

LOG = logging.getLogger(__name__)


def rollout_policy(policy: MLP, env, render: bool = False) -> List[Experience]:
    trajectory = []
    state = env.reset()
    if render:
        env.render()
    done = False
    total_reward = 0
    episode_t = 0
    success = False
    policy.eval()
    current_device = list(policy.parameters())[-1].device
    while not done:
        with torch.no_grad():
            action = policy(torch.tensor(state).to(current_device).float()).squeeze()

            np_action = action.squeeze().cpu().numpy()
            np_action = np_action.clip(min=env.action_space.low, max=env.action_space.high)

        next_state, reward, done, info_dict = env.step(np_action)

        if "success" in info_dict and info_dict["success"]:
            success = True

        if render:
            env.render()
        trajectory.append(Experience(state, np_action, next_state, reward, done))
        state = next_state
        total_reward += reward
        episode_t += 1
        if episode_t >= env._max_episode_steps or done:
            break

    return trajectory, total_reward, success


def build_networks_and_buffers(args, env, task_config, is_train=True):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy_head = [32, 1] if args.advantage_head_coef is not None else None
    policy = MLP(
        [obs_dim] + [args.net_width] * args.net_depth + [action_dim],
        final_activation=torch.tanh,
        extra_head_layers=policy_head,
        w_linear=args.weight_transform,
    ).to(args.device)

    q_function = MLP(
        [obs_dim + action_dim] + [args.net_width] * args.net_depth + [1],
        w_linear=args.weight_transform,
    ).to(args.device)

    vf = MLP(
        [obs_dim] + [args.net_width] * args.net_depth + [1],
        w_linear=args.weight_transform,
    ).to(args.device)

    if is_train:
        bp = task_config.train_buffer_paths
    else:
        bp = task_config.test_buffer_paths
    
    buffer_paths = [
        bp.format(idx) for idx in task_config.train_tasks
    ]
    

    buffers = [
        ReplayBuffer(
            args.inner_buffer_size,
            obs_dim,
            action_dim,
            discount_factor=0.99,
            immutable=True,
            load_from=buffer_paths[i],
        )
        for i, task in enumerate(task_config.train_tasks)
    ]

    return policy, vf, buffers, q_function


def get_env(args, task_config, t=False):
    if t:
        tasks=[{'direction': 1}]
    else:
        tasks = []
        for task_idx in range(task_config.total_tasks):
            with open(task_config.task_paths.format(task_idx), "rb") as f:
                task_info = pickle.load(f)
                assert len(task_info) == 1, f"Unexpected task info: {task_info}"
                tasks.append(task_info[0])

        if args.advantage_head_coef == 0:
            args.advantage_head_coef = None

    return HalfCheetahDirEnv(tasks, include_goal=False)


def get_opts_and_lrs(args, policy, vf, qf):
    policy_opt = O.Adam(policy.parameters(), lr=args.outer_policy_lr)
    vf_opt = O.Adam(vf.parameters(), lr=args.outer_value_lr)
    qf_opt = O.Adam(qf.parameters(), lr=args.outer_action_lr)
    policy_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_policy_lr).to(args.device))
        for p in policy.parameters()
    ]
    vf_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_value_lr).to(args.device))
        for p in vf.parameters()
    ]

    qf_lrs = [
        torch.nn.Parameter(torch.tensor(args.inner_action_lr).to(args.device))
        for p in qf.parameters()
    ]

    return policy_opt, vf_opt, qf_opt, policy_lrs, vf_lrs, qf_lrs


@hydra.main(config_path="config", config_name="config.yaml")
def run(args):
    
    with open(f"{get_original_cwd()}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )
    
    env = get_env(args, task_config)
    policy, vf, task_buffers, q_function = build_networks_and_buffers(args, env, task_config)
    policy_opt, vf_opt, qf_opt, policy_lrs, vf_lrs, qf_lrs = get_opts_and_lrs(args, policy, vf, q_function)

    for train_step_idx in count(start=1):
        if train_step_idx % args.rollout_interval == 0:
            LOG.info(f"Train step {train_step_idx}")

        for i, (train_task_idx, task_buffer) in enumerate(
            zip(task_config.train_tasks, task_buffers)
        ):
            inner_batch = task_buffer.sample(
                args.inner_batch_size, return_dict=True, device=args.device
            )
            outer_batch = task_buffer.sample(
                args.outer_batch_size, return_dict=True, device=args.device
            )

            # Adapt action value function
            opt = O.SGD([{"params": p, "lr": None} for p in q_function.parameters()])
            with higher.innerloop_ctx(
                q_function, opt, override={"lr": qf_lrs}, copy_initial_weights=False
            ) as (f_qf, diff_action_opt):
                loss = qf_loss_on_batch(f_qf, inner_batch, inner=True)
                diff_action_opt.step(loss)

                meta_qf_loss = qf_loss_on_batch(f_qf, outer_batch)
                total_qf_loss = meta_qf_loss / len(task_config.train_tasks)
                total_qf_loss.backward()

            # Adapt value function
            opt = O.SGD([{"params": p, "lr": None} for p in vf.parameters()])
            with higher.innerloop_ctx(
                vf, opt, override={"lr": vf_lrs}, copy_initial_weights=False
            ) as (f_vf, diff_value_opt):
                loss = vf_loss_on_batch(f_vf, q_function, inner_batch, inner=True)
                diff_value_opt.step(loss)

                meta_vf_loss = vf_loss_on_batch(f_vf, q_function, outer_batch)
                total_vf_loss = meta_vf_loss / len(task_config.train_tasks)
                total_vf_loss.backward()

            # Adapt policy using adapted value function
            adapted_vf = f_vf
            adapted_qf = f_qf
            opt = O.SGD([{"params": p, "lr": None} for p in policy.parameters()])
            with higher.innerloop_ctx(
                policy, opt, override={"lr": policy_lrs}, copy_initial_weights=False
            ) as (f_policy, diff_policy_opt):
                loss = policy_loss_on_batch(
                    f_policy,
                    adapted_vf,
                    adapted_qf,
                    inner_batch,
                    args.advantage_head_coef,
                    inner=True,
                )

                diff_policy_opt.step(loss)

                meta_policy_loss = policy_loss_on_batch(
                    f_policy, adapted_vf, adapted_qf, outer_batch, args.advantage_head_coef
                )

                (meta_policy_loss / len(task_config.train_tasks)).backward()

                # Sample adapted policy trajectory
                if train_step_idx % args.rollout_interval == 0:
                    adapted_trajectory, adapted_reward, success = rollout_policy(f_policy, env)
                    LOG.info(f"Task {train_task_idx} reward: {adapted_reward}")

        # Update the policy/value function

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5)
        policy_opt.step()
        policy_opt.zero_grad()
        vf_opt.step()
        vf_opt.zero_grad()
        qf_opt.step()
        qf_opt.zero_grad()

if __name__ == "__main__":
    run()