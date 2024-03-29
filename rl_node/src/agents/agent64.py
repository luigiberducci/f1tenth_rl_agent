import collections
import pathlib
from typing import Dict, Tuple

from dataclasses import dataclass
import marshmallow_dataclass
import numpy as np
import torch

import yaml

from .agent import Agent
from .utils import *


@dataclass
class ActionConfig:
    min_steering: float = None
    max_steering: float = None
    min_speed: float = None
    max_speed: float = None
    cap_max_speed: float = None
    cap_min_speed: float = None
    delta_speed: bool = None
    dt: float = None
    frame_skip: float = None
    max_accx: float = None


@dataclass
class ObservationConfig:
    scan_size: int = None
    min_range: float = None
    max_range: float = None
    n_last_actions: int = None
    n_last_observations: int = None
    observe_dummy_time: bool = None


@dataclass
class Agent64Config:
    policy_class: str = None
    observation_config: ObservationConfig = None
    action_config: ActionConfig = None


class Agent64(Agent):

    def __init__(self, config_filepath: pathlib.Path):
        with open(config_filepath, "r") as f:
            params = yaml.load(f, Loader=yaml.Loader)
        config_schema = marshmallow_dataclass.class_schema(Agent64Config)()
        self.config = config_schema.load(params)

        self.model = None
        self._speed_ms = np.array([0.0], dtype=np.float32)
        self._last_actions = None
        self._last_lidars = None
        self._last_velxs = None
        self.reset()

    def reset(self):
        n_last_actions = self.config.observation_config.n_last_actions
        n_last_observations = self.config.observation_config.n_last_observations
        self._last_actions = collections.deque([[0.0] * 2] * n_last_actions, maxlen=n_last_actions)
        self._last_lidars = collections.deque([[0.0] * 64] * n_last_observations, maxlen=n_last_observations)
        self._last_velxs = collections.deque([[0.0] * 1] * n_last_observations, maxlen=n_last_observations)
        self._speed_ms = np.array([0.0], dtype=np.float32)

    def load(self, model_filepath: pathlib.Path) -> bool:
        self.model = torch.jit.load(model_filepath)
        self.model.eval()
        return True

    def get_action(self, observation: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        observation_proc = self.preprocess_observation(observation)

        if isinstance(self.model, torch.nn.Module):
            observation_proc = torch.tensor(observation_proc, requires_grad=False)
            flat_action = self.model(observation_proc, deterministic=True)
            flat_action = flat_action.detach().numpy()
        else:
            # assume sb3 model
            assert self.model.observation_space.contains(observation_proc), f"invalid observation:\n{observation_proc}"
            flat_action, _ = self.model.predict(observation_proc, deterministic=True)

        norm_speed, norm_steering = flat_action
        if self.config.action_config.delta_speed:
            norm_speed = self.compute_speed_from_delta(norm_speed)

        speed = linear_scaling(norm_speed,
                               [-1, +1],
                               [self.config.action_config.min_speed, self.config.action_config.max_speed])
        steer = linear_scaling(norm_steering,
                               [-1, +1],
                               [self.config.action_config.min_steering, self.config.action_config.max_steering])

        # note: action-history wrapper is called after delta-speed transform, and uses (steer, speed)
        self._last_actions.append(np.array([norm_steering, norm_speed]))

        # return both normalized and unnormalized actions
        norm_action = {"speed": norm_speed, "steering": norm_steering}
        unnorm_action = {"speed": speed, "steering": steer}
        return norm_action, unnorm_action

    @staticmethod
    def adaptation(steer, speed, speed_multiplier, steering_multiplier, min_speed):
        speed *= speed_multiplier
        speed = max(speed, min_speed)
        steer *= steering_multiplier
        return steer, speed

    def preprocess_observation(self, observation: Dict[str, float]):
        ranges = np.array(observation["lidar"], dtype=np.float32)
        velocity = np.array([observation["velocity"]], dtype=np.float32)

        # preprocess velocity
        velocity = linear_scaling(velocity,
                                  [self.config.action_config.min_speed, self.config.action_config.max_speed],
                                  [-1, +1])

        # preprocess lidar: down-sample and scale
        n_beams = self.config.observation_config.scan_size
        min_range = self.config.observation_config.min_range
        max_range = self.config.observation_config.max_range

        selected_ranges = [ranges[int(len(ranges) // (n_beams - 1) * i)] for i in range(n_beams)]
        selected_ranges = np.nan_to_num(selected_ranges, nan=max_range)
        proc_ranges = linear_scaling(selected_ranges, [min_range, max_range], [-1, +1], clip=True)

        # concatenate observations
        self._last_lidars.append(proc_ranges)
        self._last_velxs.append(velocity)

        last_lidars = np.array(self._last_lidars, dtype=np.float32).flatten()
        last_velxs = np.array(self._last_velxs, dtype=np.float32).flatten()
        last_actions = np.array(self._last_actions, dtype=np.float32).flatten()

        # flat observations
        flat_observation = np.concatenate([
            last_actions,
            last_lidars,
            last_velxs,
        ], axis=0)

        if self.config.observation_config.observe_dummy_time:
            dummy_time = np.ones((1,), dtype=np.float32)    # 1: max remaining time, 0: no remaining time
            flat_observation = np.concatenate([flat_observation, dummy_time], axis=0)

        return flat_observation

    def compute_speed_from_delta(self, delta_speed):
        if not self.config.action_config.delta_speed:
            raise ValueError("trying to compute delta-speed in model trained without delta-speed")
        max_delta_speed = self.config.action_config.max_accx * self.config.action_config.frame_skip * self.config.action_config.dt
        delta_speed = delta_speed * max_delta_speed  # ranges in +-max delta speed

        self._speed_ms = self._speed_ms + delta_speed

        cap_minspeed = self.config.action_config.cap_min_speed
        cap_maxspeed = self.config.action_config.cap_max_speed
        self._speed_ms = np.clip(self._speed_ms, cap_minspeed, cap_maxspeed)

        minspeed = self.config.action_config.min_speed
        maxspeed = self.config.action_config.max_speed
        norm_speed = -1 + 2 * (self._speed_ms - minspeed) / (maxspeed - minspeed)

        return norm_speed
