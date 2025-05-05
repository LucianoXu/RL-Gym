import torch
from .model import Agent
import gymnasium as gym


def print_info(obs, reward, terminated, truncated, info):
    """
    打印信息
    """
    print("-" * 20)
    print(f"obs: {obs}")
    print(f"reward: {reward}")
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")
    print(f"info: {info}")
    print("-" * 20)
    
def testM(
        env_args: dict,
        agent: Agent,
    ):

    # ---------- 2. create the environment ----------
    env = gym.make(**env_args, render_mode="human")
    obs, info = env.reset()


    # ---------- 3. main loop ----------
    try:
        done = False
        step = 0
        while not done:
            step += 1

            # get the action from model
            action, probs = agent.sample([obs])


            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            env.render()

            print_info(obs, reward, terminated, truncated, info)

    except (KeyboardInterrupt, StopIteration):
        # listener 被 Esc 停止或 Ctrl+C
        pass
    finally:
        env.close()
