from typing import Callable
from ..model import Agent
import torch
import torch.nn as nn

registry : dict[str, dict[str, Callable[[], Agent]]]= {}

def register_agent_fact(env: str, agent_name: str, agent_fact: Callable[[], Agent]):
    """
    Register a model for a specific environment.
    
    Args:
        env (str): The name of the environment.
        model_name (str): The name of the model.
        model (nn.Module): The model to register.
    """
    if env not in registry:
        registry[env] = {}
    registry[env][agent_name] = agent_fact

def get_agent_fact(env: str, agent_name: str) -> Callable[[], Agent]:
    """
    Get a registered model for a specific environment.
    
    Args:
        env (str): The name of the environment.
        model_name (str): The name of the model.
    
    Returns:
        nn.Module: The registered model.
    
    Raises:
        ValueError: If the model is not found in the registry.
    """
    if env in registry and agent_name in registry[env]:
        return registry[env][agent_name]
    else:
        raise ValueError(f"Agent {agent_name} not found for environment {env}.")
    
__all__ = [
    "register_agent_fact",
    "get_agent_fact",
]