import torch
from torch import nn
import gymnasium as gym
import numpy as np

class ContinuousInput(nn.Module):
    '''
    A simple feedforward neural network for 1D continuous input.
    '''
    def __init__(self, obs_space: gym.spaces.Box, hidden_size: int):
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("obs_space must be a Box space.")

        if len(obs_space.shape) != 1:
            raise ValueError("obs_space must be a 1D Box space.")

        super(ContinuousInput, self).__init__()
        self.obs_space = obs_space
        self.hidden_size = hidden_size

        self.fw = nn.Linear(obs_space.shape[0], hidden_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.fw(x)
        x = self.activation(x)
        return x

class DiscreteOutput(nn.Module):
    def __init__(self, act_space: gym.spaces.Discrete, hidden_size: int):
        if not isinstance(act_space, gym.spaces.Discrete):
            raise ValueError("act_space must be a Discrete space.")

        super(DiscreteOutput, self).__init__()
        self.act_space = act_space
        self.hidden_size = hidden_size

        self.fw = nn.Linear(hidden_size, int(act_space.n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.fw(x)
        return x

class ContinuousOutput(nn.Module):
    def __init__(self, act_space: gym.spaces.Box, hidden_size: int):
        if not isinstance(act_space, gym.spaces.Box):
            raise ValueError("act_space must be a Box space.")

        if len(act_space.shape) != 1:
            raise ValueError("act_space must be a 1D Box space.")

        super(ContinuousOutput, self).__init__()
        self.act_space = act_space
        self.hidden_size = hidden_size

        self.fw = nn.Linear(hidden_size, act_space.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.fw(x)
        return x
    


class RMSMLP(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int):
        super(RMSMLP, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        assert hidden_layers > 0, "hidden_layers must be greater than 0"

        # Define the layers
        self.layers = nn.ModuleList()

        for _ in range(hidden_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.RMSNorm(hidden_size)
                )
            )

        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

    
from abc import ABC, abstractmethod
class Agent(ABC):
    def __init__(self, model: nn.Module):
        """
        Initialize the agent with a model.

        Args:
            model (nn.Module): The model to use for the agent.
        """
        self._model = model

    @property
    def model(self) -> nn.Module:
        """
        Get the model of the agent.

        Returns:
            nn.Module: The model of the agent.
        """
        return self._model

    @abstractmethod
    def sample(self, inputs: list) -> tuple[list, torch.Tensor]:
        """
        Sample an action from the model.

        Args:
            inputs (list): The input data.

        Returns:
            tuple: A tuple containing the sampled action and its probability.
        """
        pass

class DiscreteOutputAgent(Agent):
    def __init__(self, model: nn.Module):
        """
        Initialize the agent with a model.

        Args:
            model (nn.Module): The model to use for the agent.
        """
        super(DiscreteOutputAgent, self).__init__(model)

    def sample(self, inputs: list[np.ndarray]) -> tuple[list[int], torch.Tensor]:
        """
        Sample an action from the model.

        Args:
            inputs (list): The input data.

        Returns:
            tuple: A tuple containing the sampled action and its probability.
        """
        input_tensor = torch.tensor(np.array(inputs), dtype=torch.float32)
        logits = self.model(input_tensor)
        return sample_discrete_action(logits)

def sample_discrete_action(logits: torch.Tensor, T: float = 1.0) -> tuple[list[int], torch.Tensor]:
    """
    Sample an action from the logits using softmax.

    Args:
        logits (torch.Tensor): size (L, act_space) The logits from the model.
        T (float): The temperature for softmax.

    Returns:
        The sampled action and the probability of the action.
    """
    probs = torch.softmax(logits / T, dim=-1)
    action = torch.multinomial(probs, num_samples=1)
    return action.squeeze(-1).tolist(), probs.gather(1, action).squeeze(-1)