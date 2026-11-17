import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from tqdm.auto import tqdm

network = nn.Dense(1000)

key = jax.random.key(0)

x = jax.random.uniform(key, (1000,), minval=0, maxval=10)
y = x**2

optimizer = optax.adam(0.001)


class Trainer:
    def __init__(self, network, optimizer):
        self.network = network
        self.optimizer = optimizer

    def init(self, key, x):
        params = self.network.init(key, x)
        opt_state = self.optimizer.init(params)
        return params, opt_state

    @jax.jit
    def update(self, state, batch):
        params, opt_state = state
        x, y = batch

        def loss_fn(params, x, y):
            predictions = network.apply(params, x)  
            return jnp.mean((predictions - y) ** 2)

        grad = jax.grad(loss_fn)(params, x, y)

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        new_state = params, opt_state
        return new_state


trainer = Trainer(network, optimizer)
state = trainer.init(key, x)

updated_state = state
params, opt_state = updated_state

print(jnp.mean((network.apply(params, x) - y) ** 2))
for step in range(100):
    updated_state = trainer.update(updated_state, (x, y))

params, opt_state = updated_state
print(jnp.mean((network.apply(params, x) - y) ** 2))
from tqdm import tqdm
