import json
import os

import h5py
import jax
import mediapy as media
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state"]
REMOVED_ENV_KWARGS = ["has_renderer", "has_offscreen_renderer", "use_camera_obs", "camera_names", "camera_heights", "camera_widths"]
RENDER_SIZE = 512
RENDER_FPS = 20

class RobomimicEnv:
    def __init__(self, dataset_path, max_episode_length=400, render=False):
        with h5py.File(dataset_path, "r") as f:
            env_meta = json.loads(f["data"].attrs["env_args"])
            env_kwargs = {k: v for k, v in env_meta["env_kwargs"].items() if k not in REMOVED_ENV_KWARGS}
            env_kwargs.update(
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            has_renderer=False,
            has_offscreen_renderer=render,
            use_camera_obs=False,
            reward_shaping=False,
            hard_reset=False,
        )
        self.env = suite.make(env_meta["env_name"], **env_kwargs)
        self.action_low, self.action_high = self.env.action_spec
        self.max_episode_length = max_episode_length
        self.steps = 0

    def _observe(self, obs):
        return np.concatenate([obs[key] for key in STATE_KEYS]).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.steps = 0
        return self._observe(self.env.reset())

    def step(self, action):
        obs, reward, _, _ = self.env.step(np.clip(action, self.action_low, self.action_high))
        self.steps += 1
        terminated = reward >= 1.0
        truncated = self.steps >= self.max_episode_length
        return self._observe(obs), float(reward), terminated, truncated

    def render(self):
        return self.env.sim.render(height=RENDER_SIZE, width=RENDER_SIZE, camera_name="agentview")[::-1]

    def close(self):
        self.env.close()


def evaluate(
    policy_network, 
    policy_params,
    dataset_path="lift.hdf5",
    num_episodes=50,
    max_episode_length=400,
    render=False,
    video_dir="videos",
    seed=0,
):
    get_action = jax.jit(lambda params, obs: policy_network.apply(params, obs))
    env = RobomimicEnv(dataset_path, max_episode_length, render)
    num_success = 0
    try:
        for episode in range(num_episodes):
            obs = env.reset(seed=seed + episode)
            frames = [env.render()] if render else []
            terminated = truncated = False
            while not (terminated or truncated):
                action = np.asarray(get_action(policy_params, obs[None]))[0]
                obs, _, terminated, truncated = env.step(action)
                if render:
                    frames.append(env.render())
            num_success += int(terminated)
            if render:
                os.makedirs(video_dir, exist_ok=True)
                media.write_video(os.path.join(video_dir, f"episode_{episode}.mp4"), frames, fps=RENDER_FPS)
    finally:
        env.close()
    success_rate = num_success / num_episodes
    print(f"Success rate: {success_rate:.2f} ({num_success}/{num_episodes})")
    return success_rate
