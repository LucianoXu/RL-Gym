from typing import Any, Callable, Optional
import gymnasium as gym
import torch
import numpy as np
from torch.utils import tensorboard
from utils.sys import get_command
from utils.ml import get_grad_norm
from torch.profiler import profile, record_function, ProfilerActivity
from dataclasses import dataclass
from .model import Agent

# --------------------------------------------------------------------
# 1.  tiny helper to make the vector-env
# --------------------------------------------------------------------
def make_async_vector_env(env_fact: Callable[[], gym.Env], n_envs: int) -> gym.vector.AsyncVectorEnv:
    return gym.vector.AsyncVectorEnv([env_fact for _ in range(n_envs)])

# --------------------------------------------------------------------
# 2.  dataclass to hold one transition (unchanged logic)
# --------------------------------------------------------------------
@dataclass
class Record:
    obs: Any
    reward: float
    action: Any
    prob:   torch.Tensor          # scalar tensor on same device as model
    
def calc_loglikelihood_reward(ls : list[Record], gamma: float = 1.000) -> tuple[torch.Tensor, float]:
    """
    Calculate the log likelihood of the reward.

    Args:
        ls (list[record]): The list of records.
        gamma (float): The discount factor.

    Returns:
        The log likelihood of the reward and the total reward.
    """
    loglikelihood = torch.zeros(1, dtype=torch.float32, device=ls[0].prob.device)
    total_reward = 0.0
    for record in ls:
        loglikelihood += torch.log(record.prob)
        total_reward += record.reward

    return loglikelihood, total_reward


def sample_episode(
    env_args: dict[str, Any], 
    agent: Agent, 
    num_episodes: int = 1, 
    device: str = 'cpu') -> list[list[Record]]:
    """
    Sample a policy in the environment.

    Args:
        env_fact (Callable[[], gym.Env]): A function that creates a new environment.
        model (torch.nn.Module): The policy model.
        num_episodes (int): The number of episodes to sample.
    """

    envs = [gym.make(**env_args) for _ in range(num_episodes)]

    # Reset the environments
    # use None to represent finished envs
    current_states : list[Optional[list[float]]] = []
    records : list[list[Record]] = [[] for _ in range(num_episodes)]
    for env in envs:
        state, info = env.reset()
        current_states.append(state)

    # remaining envs
    remaining_count = num_episodes

    while remaining_count > 0:

        # gather unfinished envs
        stacked_states = []
        stacked_indices = []
        for i, state in enumerate(current_states):
            if state is not None:
                stacked_states.append(state)
                stacked_indices.append(i)

        # get the action
        action, probs = agent.sample(stacked_states)

        for i in range(len(stacked_indices)):
            idx = stacked_indices[i]
            # step the env
            obs, reward, terminated, truncated, info = envs[idx].step(action[i])
            records[idx].append(Record(current_states[idx], reward, action[i], probs[i]))

            if terminated or truncated:
                current_states[idx] = None
                envs[idx].close()
                remaining_count -= 1
            else:
                current_states[idx] = obs

    return records

@torch.no_grad()
def render_episode(
    env_args: dict[str, Any],
    agent: Agent,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Sample a policy in the environment and render the episode.
    """

    # Use "rgb_array" render_mode
    new_env_args = env_args.copy()
    new_env_args['render_mode'] = 'rgb_array'

    env = gym.make(**new_env_args)
    current_state: list[float] = []
    render_frames: list[np.ndarray] = []

    current_state, info = env.reset()
    frame = env.render()
    if frame is None:
        raise ValueError("Render mode is not set to 'rgb_array'.")
    render_frames.append(frame)  # type: ignore

    while True:

        action, _ = agent.sample([current_state])

        obs, reward, terminated, truncated, info = env.step(action[0])

        # render the frame
        render_frames.append(env.render())  # type: ignore

        if terminated or truncated:

            # Log the rendered episode
                video = np.stack(render_frames)  # shape (T, H, W, C)
                video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0) / 255.0

                return video_tensor
        else:
            current_state = obs


def REINFORCE_benchmark(
    env_args: dict[str, Any],
    agent: Agent,
    lr: float = 3e-4,
    batch_size: int = 64,

    steps: int = 10,
    device: str = "cpu",
):
    
    
    agent.model.to(device)
    agent.model.train()

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

    def one_step():
        with record_function('SAMPLING'):
            # Sample a batch of episodes
            records = sample_episode(env_args, agent, num_episodes=batch_size, device=device)

        # calculate the log likelihood and reward
        loglikelihood_ls, total_reward_ls = [], []
        for record in records:
            loglikelihood, total_reward = calc_loglikelihood_reward(record)
            loglikelihood_ls.append(loglikelihood)
            total_reward_ls.append(total_reward)
        
        # calculate the pseudo loss
        baseline = np.mean(total_reward_ls)
        advantage = np.array(total_reward_ls) - baseline
        J = torch.zeros(1, dtype=torch.float32, device=device)

        for i in range(len(records)):
            J += loglikelihood_ls[i] * advantage[i]

        J /= -len(records)

        with record_function('BACKWARD'):
            # Backpropagation
            J.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f"Step {step}: J = {J.item()}, baseline = {baseline}")

    for step in range(3):
        one_step()

    

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:

        for step in range(steps):
            one_step()
            prof.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    

def REINFORCE(
    env_args: dict[str, Any],
    ckpt: str,
    agent: Agent,
    lr: float = 3e-4,
    batch_size: int = 64,

    grad_norm_clip: Optional[float] = None,

    steps: int = 1000000,
    save_interval: int = 100,

    device: str = "cpu",
):
    
    
    agent.model.to(device)
    agent.model.train()

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

    # write the tensorboard
    writer = tensorboard.SummaryWriter(log_dir=ckpt)
    writer.add_text("command", get_command())

    try:
        for step in range(steps):

            # Sample a batch of episodes
            records_ls = sample_episode(env_args, agent, num_episodes=batch_size, device=device)

            # calculate the log likelihood and reward
            loglikelihood_ls, total_reward_ls = [], []
            for records in records_ls:
                loglikelihood, total_reward = calc_loglikelihood_reward(records)
                loglikelihood_ls.append(loglikelihood)
                total_reward_ls.append(total_reward)
            
            # calculate the pseudo loss
            baseline = np.mean(total_reward_ls)
            advantage = np.array(total_reward_ls) - baseline
            J = torch.zeros(1, dtype=torch.float32, device=device)

            for i in range(len(records_ls)):
                J += loglikelihood_ls[i] * advantage[i]

            J /= -len(records_ls)

            # Backpropagation
            J.backward()

            # get the normalized gradient
            norm_grad = get_grad_norm(agent.model)
            if isinstance(grad_norm_clip, float):
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), grad_norm_clip)

            optimizer.step()
            optimizer.zero_grad()

            print(f"Step {step}: J = {J.item()}, baseline = {baseline}")

            # Write to tensorboard
            writer.add_scalar("J", J.item(), step)
            writer.add_scalar("avg reward", baseline, step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("grad norm", norm_grad, step)
            writer.flush()


            if step % save_interval == 0:
                # record the episode
                print("Recording episode...")
                video = render_episode(env_args, agent, device=device)
                writer.add_video("video", video, step, fps=30)
                writer.flush()

                # Save the model
                torch.save(agent.model.state_dict(), f"{ckpt}/model_{step}.pth")
                print(f"Model saved at step {step}")

    finally:
        # Close the writer and save the final model
        writer.close()
        torch.save(agent.model.state_dict(), f"{ckpt}/model_final.pth")

