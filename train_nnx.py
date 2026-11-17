import jax
import jax.numpy as jnp
from flax import nnx
# from model_nnx import PolicNetwork_NNX
import optax


@nnx.jit
def train_step(network, optimizer, x, y):

    def loss_fn(network, x, y):
        predictions = network(x)
        return jnp.mean((predictions - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(network, x, y)
    optimizer.update(network, grads)

    return loss
