
import argparse
from ..train import *
from ..testM import testM
from ..experiments import get_agent_fact

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("testM", help="Test performance using agent.")
    parser.add_argument("env", type=str, help="Environment ID.")
    parser.add_argument("agent", type=str, help="Agent name.")
    parser.add_argument("--ckpt", type=str, help="Path to the checkpoint folder.", default="./ckpt/temp")
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

    agent.model.load_state_dict(torch.load(parsed_args.ckpt))

    testM(env_args, agent)