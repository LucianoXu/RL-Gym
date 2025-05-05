
import argparse
from ..train import *

from ..experiments import get_agent_fact
import os

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("train", help="Execute RL training.")
    parser.add_argument("env", type=str, help="Environment ID.")
    parser.add_argument("agent", type=str, help="Agent name.")

    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
    parser.add_argument("--init", action="store_true", help="Initialize training models.")
    parser.add_argument("--bs", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--grad_norm_clip", type=float, default=None, help="Gradient norm clipping value.")
    parser.add_argument("--steps", type=int, default=1000000, help="Number of training steps.")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval for saving the model.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # check whether the game is available
    if parsed_args.env not in gym.envs.registry.keys(): # type: ignore[union-attr]
        raise ValueError(f"Environment {parsed_args.env} is not available in gym.")

    env_args = {
        "id": parsed_args.env,
    }

    agent_fact = get_agent_fact(parsed_args.env, parsed_args.agent)
    
    # Initialize the model
    agent = agent_fact()

    if parsed_args.init:
        # create the checkpoint folder
        os.makedirs(parsed_args.ckpt, exist_ok=True)
        torch.save(agent.model.state_dict(), parsed_args.ckpt+"/model.pth")

    else:
        # Load the model
        agent.model.load_state_dict(torch.load(parsed_args.ckpt))

    REINFORCE(
        env_args, 
        ckpt=parsed_args.ckpt,
        agent=agent,
        lr=3e-4,
        batch_size=parsed_args.bs,
        steps=parsed_args.steps,
        save_interval=parsed_args.save_interval,
        grad_norm_clip=parsed_args.grad_norm_clip)
