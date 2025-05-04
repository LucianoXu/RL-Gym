
import argparse
from ..train import *
from ..model import MLP
import os

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("train_bm", help="Execute RL training for benchmarking.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use (cuda/cpu/mps).")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    def get_lunar_landing_env():
        return gym.make("LunarLander-v3")
    
    model = MLP(obs_space=8, act_space=4, hidden_size=128, hidden_layers=2)
        
    REINFORCE_benchmark(get_lunar_landing_env, 
          model=model,
          lr=3e-4,
          batch_size=64,
          device=parsed_args.device,)
