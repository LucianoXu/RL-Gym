from src.landingH import landing
import argparse
from src.train import *
from src.model import MLP
import os


def get_lunar_landing_env():
    return gym.make("LunarLander-v3")

model = MLP(obs_space=8, act_space=4, hidden_size=128, hidden_layers=2)
    
REINFORCE_benchmark(get_lunar_landing_env, 
        model=model,
        lr=3e-4,
        batch_size=64,
        device='cpu')
