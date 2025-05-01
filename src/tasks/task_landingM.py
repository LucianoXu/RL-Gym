
import argparse
from ..train import *
from ..model import Model
from ..landingM import landing
import os

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("landingM", help="landing using model.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu).")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # Load the model
    model = Model(obs_space=8, act_space=4, hidden_size=128, hidden_layers=2)
    model.load_state_dict(torch.load(parsed_args.ckpt))

    landing(model)