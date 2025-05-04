
import argparse
from ..train import *
from ..model import MLP, RMSMLP
from ..landingM import landing
import os

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("landingM", help="landing using model.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("--arch", type=str, default="mlp", help="Model architecture (mlp/rmsmlp).")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the model.")
    parser.add_argument("--hidden_layers", type=int, default=2, help="Number of hidden layers in the model.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # Load the model
    if parsed_args.arch == 'mlp':
        model = MLP(obs_space=8, act_space=4, hidden_size=parsed_args.hidden_size, hidden_layers=parsed_args.hidden_layers)
    elif parsed_args.arch == 'rmsmlp':
        model = RMSMLP(obs_space=8, act_space=4, hidden_size=parsed_args.hidden_size, hidden_layers=parsed_args.hidden_layers)
    else:
        raise ValueError("Invalid architecture.")

    model.load_state_dict(torch.load(parsed_args.ckpt))

    landing(model)