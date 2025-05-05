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

        # output the mean and std of the action
        self.mean = nn.Linear(hidden_size, act_space.shape[0])

        self.std = nn.Sequential(
            nn.Linear(hidden_size, act_space.shape[0]),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        mean = self.mean(x)
        std = self.std(x) + 1e-6 # add a small value to avoid division by zero
        res = torch.cat((mean, std), dim=-1)
        return res
    


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

def sample_continuous_action(
        mean: torch.Tensor, 
        std: torch.Tensor,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    """
    Sample a continuous action from the mean and standard deviation, using the Gaussian distribution. Transform them to the lower_bound to upper_bound region. 
    Then output the probability density of the action (w.r.t. the final region).
    Args:
        mean (torch.Tensor): The mean of the Gaussian distribution.
        stddev (torch.Tensor): The standard deviation of the Gaussian distribution.
        lower_bound (list[float]): The lower bound of the action space.
        upper_bound (list[float]): The upper bound of the action space.
    Returns:
        The sampled action and the probability density of the action.
    """
    action = torch.normal(mean, std)
    # transform the action to the lower_bound to upper_bound region
    final_action = lower_bound + (upper_bound - lower_bound) * torch.sigmoid(action)

    prob = torch.exp(-0.5 * ((action - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    # calculate the coefficients
    coefs = 1 / (upper_bound - lower_bound) * 1 / (torch.sigmoid(action) * (1 - torch.sigmoid(action)))

    # transform final action to a list of numpy arrays
    final_action = final_action.detach().cpu().numpy()

    return final_action, prob * coefs


class ContinuousOutputAgent(Agent):
    def __init__(self, model: nn.Module, box: gym.spaces.Box):
        """
        Initialize the agent with a model.

        Args:
            model (nn.Module): The model to use for the agent.
        """
        super(ContinuousOutputAgent, self).__init__(model)

        self.box = box

        # get the lower bound and upper bound of the action space
        self.lower_bound = torch.tensor(box.low, dtype=torch.float32)
        self.upper_bound = torch.tensor(box.high, dtype=torch.float32)

    def sample(self, inputs: list[np.ndarray]) -> tuple[np.ndarray, torch.Tensor]:
        """
        Sample an action from the model.

        Args:
            inputs (list): The input data.

        Returns:
            tuple: A tuple containing the sampled action and its probability density.
        """
        input_tensor = torch.tensor(np.array(inputs), dtype=torch.float32)
        bs_size = input_tensor.shape[0]
        model_output = self.model(input_tensor).reshape(bs_size, -1, 2)
        mean, std = model_output[..., 0], model_output[..., 1]

        return sample_continuous_action(
            mean, std,
            self.lower_bound, self.upper_bound
        )
