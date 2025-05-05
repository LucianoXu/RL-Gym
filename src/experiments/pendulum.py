from ..model import *
import gymnasium as gym
from .model_registry import register_agent_fact

class Model(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int):
        super(Model, self).__init__()

        env = gym.make("Pendulum-v1")

        self.layers = nn.Sequential(
            ContinuousInput(
                obs_space=env.observation_space, # type: ignore
                hidden_size=hidden_size
            ),
            RMSMLP(
                hidden_size=hidden_size, hidden_layers=hidden_layers
            ),
            ContinuousOutput(
                act_space=env.action_space, # type: ignore
                hidden_size=hidden_size
            ),
        )

        env.close()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
def fact_a0():
    model = Model(hidden_size=128, hidden_layers=2)
    env = gym.make("Pendulum-v1")
    act_space = env.action_space
    env.close()
    if not isinstance(act_space, gym.spaces.Box):
        raise ValueError("act_space must be a Box space.")
    
    return ContinuousOutputAgent(model, act_space)

register_agent_fact("Pendulum-v1", "a0", fact_a0)


def fact_a1():
    model = Model(hidden_size=128, hidden_layers=4)
    env = gym.make("Pendulum-v1")
    act_space = env.action_space
    env.close()
    if not isinstance(act_space, gym.spaces.Box):
        raise ValueError("act_space must be a Box space.")
    
    return ContinuousOutputAgent(model, act_space)

register_agent_fact("Pendulum-v1", "a1", fact_a1)