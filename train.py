import jax
from flax import linen as nn 
import optax 
import jax.numpy as jnp 


class Trainer():
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def init(self, key, x):
        params = self.model.init(key, x)
        opt_state = self.optimizer.init(params)

        return params, opt_state

    def loss_fn(self, params, x, y):
        predictions = self.model.apply(params, x)
        return jnp.mean((predictions - y)**2)

    def update(self, state, batch):
        params, opt_state = state
        x, y = batch

        loss, grad = jax.value_and_grad(self.loss_fn)(params, x, y)

        updates, opt_state = self.optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        new_state = params, opt_state
        return new_state, loss
