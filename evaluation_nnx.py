"""Evaluate a Flax NNX policy in the existing Robosuite Lift environment.

This module reuses RobomimicEnv from evaluation.py and leaves that file unchanged.
Call evaluate(policy_network=your_trained_nnx_model, ...) after training.
"""

import os

import mediapy as media
import numpy as np
from flax import nnx

from evaluation import RENDER_FPS, RobomimicEnv


@nnx.jit
def _predict_action(policy_network: nnx.Module, observation):
    """Predict actions for a batch of observations using an NNX model."""
    return policy_network(observation)


def evaluate(
    policy_network: nnx.Module,
    dataset_path: str = "lift.hdf5",
    num_episodes: int = 50,
    max_episode_length: int = 400,
    render: bool = False,
    video_dir: str = "videos",
    seed: int = 0,
) -> float:
    """Run the policy in simulation and return its episode success rate.

    The NNX model contains its own learned parameters, so no policy_params
    argument is needed. If render=True, writes episode_*.mp4 in video_dir.
    """
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")

    env = RobomimicEnv(dataset_path, max_episode_length, render)
    num_success = 0

    try:
        for episode in range(num_episodes):
            obs = env.reset(seed=seed + episode)
            frames = [env.render()] if render else []
            terminated = truncated = False

            while not (terminated or truncated):
                # obs[None] has shape (1, 19). The policy returns (1, 7).
                # [0] extracts the (7,) action expected by Robosuite.
                action = np.asarray(_predict_action(policy_network, obs[None]))[0]
                obs, _, terminated, truncated = env.step(action)

                if render:
                    frames.append(env.render())

            num_success += int(terminated)

            if render:
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"episode_{episode}.mp4")
                media.write_video(video_path, frames, fps=RENDER_FPS)

    finally:
        env.close()

    success_rate = num_success / num_episodes
    print(f"Success rate: {success_rate:.2f} ({num_success}/{num_episodes})")
    return success_rate
