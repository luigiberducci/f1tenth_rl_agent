import numpy as np
import stable_baselines3
import torch


class SACMlpPolicy(torch.nn.Module):

    def __init__(self, latent_pi=None, mu=None, log_std=None):
        super(SACMlpPolicy, self).__init__()
        self.latent_pi = latent_pi
        self.mu = mu
        self.log_std = log_std
        self.actor = torch.nn.Sequential(self.latent_pi, self.mu)

    @staticmethod
    def from_actor(actor: stable_baselines3.sac.policies.Actor):
        return SACMlpPolicy(actor.latent_pi, actor.mu, actor.log_std)

    def forward(self, observation: torch.Tensor, deterministic: bool):
        # assume observation are normalized
        z = self.latent_pi(observation)
        mu, log_std = self.mu(z), self.log_std(z)

        LOG_STD_MIN, LOG_STD_MAX = -20, 2
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # from sac/policies.py:171

        action_std = torch.ones_like(mu) * log_std.exp()  # from distributions.py:151
        #gaussian_distr = torch.distributions.Normal(mu, action_std)  # from distributions.py:151

        if deterministic:
            gaussian_actions = mu
        else:
            gaussian_actions = mu + torch.rand_like(action_std) * action_std

        squashed_actions = torch.tanh(gaussian_actions)  # from distributions.py: 236, 241
        return squashed_actions