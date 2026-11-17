from dataclasses import dataclass
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


@jax.tree_util.register_dataclass
@dataclass
class Sample:
    obs: jnp.ndarray
    actions: jnp.ndarray
    next_obs: jnp.ndarray
    next_actions: jnp.ndarray
    rewards: jnp.ndarray
    terminations: jnp.ndarray
    end_of_episodes: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class ItemBufferState:
    item: Any
    index: jax.Array | int
    n_items: jax.Array | int
    buffer: Any

    @property
    def size(self):
        return jax.tree.leaves(self.buffer)[0].shape[0]


class ItemBuffer:
    def init(self, size, item):
        buffer = jax.tree.map(lambda i: np.zeros((size, i.ravel().shape[0]), dtype=i.dtype), item)
        item = jax.tree.map(lambda x: x.copy(), item)
        return ItemBufferState(item=item, index=0, n_items=0, buffer=buffer)

    def add(self, buffer_state, item):
        idx = buffer_state.index
        item_flat = jax.tree.map(lambda i: i.ravel(), item)

        def update_array(buffer, i_flat):
            buffer[idx] = i_flat
            return buffer

        buffer = jax.tree.map(update_array, buffer_state.buffer, item_flat)
        return ItemBufferState(
            item=buffer_state.item,
            index=(idx + 1) % buffer_state.size,
            n_items=jnp.minimum(buffer_state.n_items + 1, buffer_state.size),
            buffer=buffer,
        )

    def sample(self, buffer_state, key, shape=()):
        idxs = jax.random.randint(key, shape, 0, buffer_state.n_items)
        return jax.tree.map(
            lambda b, i: np.reshape(b[idxs], shape + i.shape),
            buffer_state.buffer,
            buffer_state.item,
        )


def load_dataset(dataset_path="lift.hdf5", num_demonstrations=None):
    with h5py.File(dataset_path, "r") as f:
        data = f["data"]
        num_demonstrations = len(data) if num_demonstrations is None else num_demonstrations
        demos = [data[f"demo_{i}"] for i in range(num_demonstrations)]
        obs = np.concatenate([np.concatenate([d["obs"][k][:] for k in STATE_KEYS], axis=-1) for d in demos]).astype(np.float32)
        actions = np.concatenate([d["actions"][:] for d in demos]).astype(np.float32)
        episode_lengths = np.array([d["actions"].shape[0] for d in demos])

    num_samples = len(actions)
    episode_ends = np.cumsum(episode_lengths) - 1
    is_episode_end = np.zeros((num_samples, 1), dtype=bool)
    is_episode_end[episode_ends] = True
    next_idxs = np.arange(num_samples) + 1
    next_idxs[episode_ends] = episode_ends

    buffer = Sample(
        obs=obs,
        actions=actions,
        next_obs=obs[next_idxs],
        next_actions=actions[next_idxs],
        rewards=is_episode_end.astype(np.float32),
        terminations=is_episode_end,
        end_of_episodes=is_episode_end,
    )
    item = jax.tree.map(lambda x: x[0].copy(), buffer)
    return ItemBufferState(item=item, index=0, n_items=num_samples, buffer=buffer)
