
import argparse
from ..train import *
from ..model import Model
import os

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("train", help="Execute RL training.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("--init", action="store_true", help="Initialize training models.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use (cuda/cpu/mps).")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    def get_lunar_landing_env():
        return gym.make("LunarLander-v3")
    
    if parsed_args.init:
        # Initialize the model
        model = Model(obs_space=8, act_space=4, hidden_size=128, hidden_layers=2)
        # create the checkpoint folder
        os.makedirs(parsed_args.ckpt, exist_ok=True)
        torch.save(model.state_dict(), parsed_args.ckpt+"/model.pth")

    else:
        # Load the model
        model = Model(obs_space=8, act_space=4, hidden_size=128, hidden_layers=2)
        model.load_state_dict(torch.load(parsed_args.ckpt))

    REINFORCE(get_lunar_landing_env, 
          ckpt=parsed_args.ckpt,
          model=model,
          lr=3e-4,
          batch_size=64,
          steps=1000000,
          device=parsed_args.device,)
