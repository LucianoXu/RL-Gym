import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, obs_space: int, act_space: int, hidden_size: int, hidden_layers: int):
        super(MLP, self).__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        assert hidden_layers > 0, "hidden_layers must be greater than 0"

        # Define the layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_space, hidden_size))
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, act_space))

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
    

class RMSMLP(nn.Module):
    def __init__(self, obs_space: int, act_space: int, hidden_size: int, hidden_layers: int):
        super(RMSMLP, self).__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

        assert hidden_layers > 0, "hidden_layers must be greater than 0"

        # Define the layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_space, hidden_size))

        for _ in range(hidden_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.RMSNorm(hidden_size)
                )
            )

        self.layers.append(nn.Linear(hidden_size, act_space))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x