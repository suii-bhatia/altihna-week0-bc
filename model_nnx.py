import jax
from flax import nnx

class PolicyNetwork_NNX(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(19, 100, rngs=rngs)
        self.linear2 = nnx.Linear(100, 100, rngs=rngs)
        self.linear3 = nnx.Linear(100, 7, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        return self.linear3(x)

        
