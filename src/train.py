from typing import Callable, Optional
import gymnasium as gym
import torch
import numpy as np
from torch.utils import tensorboard
from utils.sys import get_command
from utils.ml import get_grad_norm
from torch.profiler import profile, record_function, ProfilerActivity
from dataclasses import dataclass

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
    obs:   np.ndarray | torch.Tensor
    reward: float
    action: int
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


def sample_action(logits: torch.Tensor, T: float = 1.0) -> tuple[list[int], torch.Tensor]:
    """
    Sample an action from the logits using softmax.

    Args:
        logits (torch.Tensor): size (L, act_space) The logits from the model.
        T (float): The temperature for softmax.

    Returns:
        The sampled action and the probability of the action.
    """
    probs = torch.softmax(logits / T, dim=-1)
    action = torch.multinomial(probs, num_samples=1)
    return action.squeeze(-1).tolist(), probs.gather(1, action).squeeze(-1)


def sample_episode(
    env_fact: Callable[[],gym.Env], 
    model: torch.nn.Module, 
    num_episodes: int = 1, 
    device: str = 'cpu') -> list[list[Record]]:
    """
    Sample a policy in the environment.

    Args:
        env_fact (Callable[[], gym.Env]): A function that creates a new environment.
        model (torch.nn.Module): The policy model.
        num_episodes (int): The number of episodes to sample.
    """

    envs = [env_fact() for _ in range(num_episodes)]

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

        # convert to tensor
        stacked_states = torch.tensor(stacked_states, dtype=torch.float32)

        # get the action
        logits = model(stacked_states)
        action, probs = sample_action(logits)

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


# --------------------------------------------------------------------
# vectorised sampler
# --------------------------------------------------------------------
def sample_episode_vector(
    env_fact: Callable[[],gym.Env], 
    model: torch.nn.Module, 
    num_episodes: int = 1, 
    device: str = 'cpu') -> list[list[Record]]:
    """
    Sample a policy in the environment.

    Args:
        env_fact (Callable[[], gym.Env]): A function that creates a new environment.
        model (torch.nn.Module): The policy model.
        num_episodes (int): The number of episodes to sample.
    """

    envs = gym.vector.AsyncVectorEnv([env_fact for _ in range(num_episodes)])

    records : list[list[Record]] = [[] for _ in range(num_episodes)]
    
    # Reset the environments
    # use None to represent finished envs
    current_states, infos = envs.reset()

    # remaining envs
    unfinished_indices = list(range(num_episodes))

    while len(unfinished_indices) > 0:

        # get the action
        logits = model(torch.tensor(current_states, device=device))
        action, probs = sample_action(logits)

        for i in range(num_episodes):
            if i not in unfinished_indices:
                action = action[:i] + [0] + action[i:]

        obs, reward, terminated, truncated, info = envs.step(action)

        new_current_states = []
        new_unfinished_indices = []

        for i in range(len(unfinished_indices)):
            idx = unfinished_indices[i]
            records[idx].append(Record(current_states[i], reward[idx], action[idx], probs[i]))

            if terminated[idx] or truncated[idx]:
                pass
            else:
                new_current_states.append(obs[idx])
                new_unfinished_indices.append(idx)

        current_states = new_current_states
        unfinished_indices = new_unfinished_indices

    return records



def REINFORCE_benchmark(
    env_fact: Callable[[], gym.Env],
    model: torch.nn.Module,
    lr: float = 3e-4,
    batch_size: int = 64,

    steps: int = 10,
    device: str = "cpu",
):
    
    
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def one_step():
        with record_function('SAMPLING'):
            # Sample a batch of episodes
            records = sample_episode(env_fact, model, num_episodes=batch_size, device=device)

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
    env_fact: Callable[[], gym.Env],
    ckpt: str,
    model: torch.nn.Module,
    lr: float = 3e-4,
    batch_size: int = 64,

    steps: int = 1000000,
    save_interval: int = 100,

    device: str = "cpu",
):
    
    
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # write the tensorboard
    writer = tensorboard.SummaryWriter(log_dir=ckpt)
    writer.add_text("command", get_command())

    try:
        for step in range(steps):

            # Sample a batch of episodes
            records = sample_episode(env_fact, model, num_episodes=batch_size, device=device)

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

            # Backpropagation
            J.backward()

            # get the normalized gradient
            norm_grad = get_grad_norm(model)

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
                # Save the model
                torch.save(model.state_dict(), f"{ckpt}/model_{step}.pth")
                print(f"Model saved at step {step}")

    finally:
        # Close the writer and save the final model
        writer.close()
        torch.save(model.state_dict(), f"{ckpt}/model_final.pth")

