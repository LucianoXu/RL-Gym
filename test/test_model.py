from src.model import Model
import torch
from src.train import sample_action, sample_episode
import gymnasium as gym

def test_sample_action():

    model = Model(obs_space=8, act_space=4, hidden_size=128, hidden_layers=2)

    logits = model(torch.randn(4, 8))
    action, probs = sample_action(logits, T=1.0)
    print("logits:", logits)
    print("action:", action)
    print("probs:", probs)

def test_sample_episode():
    
    def get_lunar_landing_env():
        return gym.make("LunarLander-v3")

    model = Model(obs_space=8, act_space=4, hidden_size=128, hidden_layers=2)

    records = sample_episode(get_lunar_landing_env, model, num_episodes=10)

    print(records)
