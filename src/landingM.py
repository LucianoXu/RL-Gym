import torch
import gymnasium as gym
from .train import sample_action


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
    
def landing(
        model: torch.nn.Module
    ):

    # ---------- 2. create the environment ----------
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = env.reset()


    # ---------- 3. main loop ----------
    try:
        done = False
        step = 0
        while not done:
            step += 1

            # get the action from model
            logits = model(torch.tensor([obs], dtype=torch.float32))
            action, probs = sample_action(logits)


            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            env.render()

            print_info(obs, reward, terminated, truncated, info)

    except (KeyboardInterrupt, StopIteration):
        # listener 被 Esc 停止或 Ctrl+C
        pass
    finally:
        env.close()
