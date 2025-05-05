
import argparse
from ..train import *
from ..model import MLP, RMSMLP
import os

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("train", help="Execute RL training.")
    parser.add_argument("env", type=str, help="Environment ID.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("-i", "--input_size", type=int, default=8, help="Input size of the model.")
    parser.add_argument("-o", "--output_size", type=int, default=4, help="Output size of the model.")
    parser.add_argument("--init", action="store_true", help="Initialize training models.")
    parser.add_argument("--arch", type=str, default="mlp", help="Model architecture (mlp/rmsmlp).")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the model.")
    parser.add_argument("--hidden_layers", type=int, default=2, help="Number of hidden layers in the model.")
    parser.add_argument("--grad_norm_clip", type=float, default=None, help="Gradient norm clipping value.")
    parser.add_argument("--steps", type=int, default=1000000, help="Number of training steps.")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval for saving the model.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use (cuda/cpu/mps).")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # check whether the game is available
    if parsed_args.env not in gym.envs.registry.keys(): # type: ignore[union-attr]
        raise ValueError(f"Environment {parsed_args.env} is not available in gym.")

    lunar_landing_env = {
        "id": parsed_args.env,
    }
    
    # Initialize the model
    if parsed_args.arch == "mlp":
        model = MLP(obs_space=parsed_args.input_size, act_space=parsed_args.output_size, hidden_size=parsed_args.hidden_size, hidden_layers=parsed_args.hidden_layers)
    elif parsed_args.arch == "rmsmlp":
        model = RMSMLP(obs_space=parsed_args.input_size, act_space=parsed_args.output_size, hidden_size=parsed_args.hidden_size, hidden_layers=parsed_args.hidden_layers)
    else:
        raise ValueError("Invalid architecture.")

    if parsed_args.init:
        # create the checkpoint folder
        os.makedirs(parsed_args.ckpt, exist_ok=True)
        torch.save(model.state_dict(), parsed_args.ckpt+"/model.pth")

    else:
        # Load the model
        model.load_state_dict(torch.load(parsed_args.ckpt))

    REINFORCE(
        lunar_landing_env, 
        ckpt=parsed_args.ckpt,
        model=model,
        lr=3e-4,
        batch_size=64,
        steps=parsed_args.steps,
        save_interval=parsed_args.save_interval,
        grad_norm_clip=parsed_args.grad_norm_clip,
        device=parsed_args.device,)
