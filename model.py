from flax import linen as nn

class PolicyNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(100)(x))
        x = nn.relu(nn.Dense(100)(x))
        # x = nn.relu(nn.Dense(100)(x))
        x = nn.Dense(7)(x)

        return x





    

    
