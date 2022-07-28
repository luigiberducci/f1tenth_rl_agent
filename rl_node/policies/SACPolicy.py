import numpy as np
import stable_baselines3
import torch


class SACMlpPolicy(torch.nn.Module):

    def __init__(self, latent_pi=None, mu=None):
        super(SACMlpPolicy, self).__init__()
        self.latent_pi = latent_pi
        self.mu = mu
        self.actor = torch.nn.Sequential(self.latent_pi, self.mu)

    @staticmethod
    def from_actor(actor: stable_baselines3.sac.policies.Actor):
        return SACMlpPolicy(actor.latent_pi, actor.mu)

    def forward(self, observation: torch.Tensor, deterministic: bool = True):
        assert deterministic, "only deterministic prediction is supported"
        # assume observation are normalized
        mu = self.actor(observation)
        squashed_actions = torch.tanh(mu)  # from distributions.py: 236, 241
        return squashed_actions