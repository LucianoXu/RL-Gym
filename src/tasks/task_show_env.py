
import argparse
from ..train import *
from ..landingH import landing
import time

def build_parser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("show_env", help="landing using Human.")
    parser.add_argument("game", type=str, help="Game name to be used.")
    parser.set_defaults(func=task)

def task(parsed_args: argparse.Namespace):

    # create the game env
    env = gym.make(parsed_args.game, render_mode="human")

    # check the action space
    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Discrete):
        pass
    else:
        raise ValueError("Only discrete action space is supported.")

    obs, info = env.reset()

    def print_info(step, obs, reward, terminated, truncated, info):
        """
        打印信息
        """
        print("-" * 20)
        print(f"step: {step}")
        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        print(f"info: {info}")
        print("-" * 20)

    # main loop
    try:
        done = False
        step = 0
        while not done:
            step += 1
            current_action = act_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated
            env.render()

            print_info(step, obs, reward, terminated, truncated, info)

            time.sleep(1 / env.metadata["render_fps"])
    except (KeyboardInterrupt, StopIteration):
        # listener 被 Esc 停止或 Ctrl+C
        pass
    finally:
        env.close()
