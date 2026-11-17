
import jax
import numpy as np
import optax
from flax import nnx
from tqdm.auto import tqdm

from dataloader import ItemBuffer, load_dataset

# Linen
from model import PolicyNetwork as LinenPolicyNetwork
from train import Trainer
from evaluation import evaluate, RobomimicEnv

# NNX
from model_nnx import PolicyNetwork_NNX as NNXPolicyNetwork
from train_nnx import train_step


@nnx.jit
def get_nnx_action(model, obs):
    return model(obs)


def evaluate_nnx(
    policy,
    dataset_path="lift.hdf5",
    num_episodes=50,
    max_episode_length=400,
    seed=0,
):
    # Reuse exactly the same environment class as Linen
    env = RobomimicEnv(
        dataset_path,
        max_episode_length=max_episode_length,
        render=False,
    )

    num_success = 0

    try:
        for episode in range(num_episodes):
            obs = env.reset(seed=seed + episode)

            terminated = False
            truncated = False

            while not (terminated or truncated):

                action = np.asarray(
                    get_nnx_action(policy, obs[None])
                )[0]

                obs, _, terminated, truncated = env.step(action)

            num_success += int(terminated)

    finally:
        env.close()

    success_rate = num_success / num_episodes

    print(
        f"NNX success rate: {success_rate:.2%} "
        f"({num_success}/{num_episodes})"
    )

    return success_rate


def main():
    # -------------------------
    # 1. Load dataset
    # -------------------------
    dataset = load_dataset("lift.hdf5")
    print(f"Number of samples: {dataset.n_items}")

    replay_buffer = ItemBuffer()

    # Sampling randomness is shared by both models
    key = jax.random.key(42)

    # -------------------------
    # 2. Initialize Linen
    # -------------------------
    linen_policy = LinenPolicyNetwork()

    linen_optimizer = optax.adam(0.001)

    linen_trainer = Trainer(
        linen_policy,
        linen_optimizer
    )

    linen_state = linen_trainer.init(
        jax.random.key(0),
        dataset.item.obs
    )

    # -------------------------
    # 3. Initialize NNX
    # -------------------------
    nnx_policy = NNXPolicyNetwork(
        rngs=nnx.Rngs(0)
    )

    nnx_optimizer = nnx.Optimizer(
        nnx_policy,
        optax.adam(0.001),
        wrt=nnx.Param
    )

    # -------------------------
    # 4. Train both models
    # -------------------------
    progress_bar = tqdm(
        range(1000),
        desc="Training Linen vs NNX"
    )

    linen_running_loss = 0.0
    nnx_running_loss = 0.0

    for step in progress_bar:

        key, sample_key = jax.random.split(key)

        # SAME minibatch for both models
        mini_batch = replay_buffer.sample(
            dataset,
            sample_key,
            (32,)
        )

        x = mini_batch.obs
        y = mini_batch.actions

        # Train Linen
        linen_state, linen_loss = linen_trainer.update(
            linen_state,
            (x, y)
        )

        # Train NNX
        nnx_loss = train_step(
            nnx_policy,
            nnx_optimizer,
            x,
            y
        )

        linen_running_loss += float(linen_loss)
        nnx_running_loss += float(nnx_loss)

        if (step + 1) % 100 == 0:

            progress_bar.set_postfix(
                linen_loss=f"{linen_running_loss / 100:.4f}",
                nnx_loss=f"{nnx_running_loss / 100:.4f}"
            )

            linen_running_loss = 0.0
            nnx_running_loss = 0.0

    # -------------------------
    # 5. Evaluate Linen
    # -------------------------
    linen_params, linen_opt_state = linen_state

    print("\nEvaluating Linen...")

    linen_success = evaluate(
        policy_network=linen_policy,
        policy_params=linen_params,
        dataset_path="lift.hdf5",
        num_episodes=50,
        max_episode_length=400,
        render=False,
        seed=0
    )

    # -------------------------
    # 6. Evaluate NNX
    # -------------------------
    print("\nEvaluating NNX...")

    nnx_success = evaluate_nnx(
        policy=nnx_policy,
        dataset_path="lift.hdf5",
        num_episodes=50,
        max_episode_length=400,
        seed=0
    )

    # -------------------------
    # 7. Compare results
    # -------------------------
    print("\n===== FINAL COMPARISON =====")

    print(f"Linen success rate: {linen_success:.2%}")
    print(f"NNX success rate:   {nnx_success:.2%}")
    print(f"Difference:         {nnx_success - linen_success:+.2%}")


if __name__ == "__main__":
    main()
