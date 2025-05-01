
import argparse
from ..train import *
from ..model import Model
from ..landingH import landing

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("landingH", help="landing using Human.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    landing()