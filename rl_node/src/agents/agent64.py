import collections
import pathlib
from typing import Dict

from dataclasses import dataclass
import marshmallow_dataclass
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


@dataclass
class ObservationConfig:
    scan_size: int = None
    min_range: float = None
    max_range: float = None
    n_last_cmd: int = None


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
        self._last_actions = None
        self.reset()

    def reset(self):
        n_last_actions = self.config.observation_config.n_last_cmd
        self._last_actions = collections.deque([[0.0] * 2] * n_last_actions, maxlen=n_last_actions)

    def load(self, model_filepath: pathlib.Path) -> bool:
        self.model = torch.jit.load(model_filepath)
        self.model.eval()
        return True

    def get_action(self, observation: Dict[str, float], normalized=False) -> Dict[str, float]:
        observation_proc = self.preprocess_observation(observation)

        observation_proc = torch.tensor(observation_proc, requires_grad=False)
        flat_action = self.model(observation_proc, deterministic=True)
        flat_action = flat_action.detach().numpy()

        self._last_actions.append(flat_action)

        norm_speed, norm_steering = flat_action

        if normalized:
            speed, steer = norm_speed, norm_steering
        else:
            speed = linear_scaling(norm_speed,
                                   [-1, +1],
                                   [self.config.action_config.min_speed, self.config.action_config.max_speed])
            steer = linear_scaling(norm_steering,
                                   [-1, +1],
                                   [self.config.action_config.min_steering, self.config.action_config.max_steering])
        return {"speed": speed, "steering": steer}

    def preprocess_observation(self, observation: Dict[str, float]):
        ranges = np.array(observation["lidar"], dtype=np.float32)
        velocity = np.array([observation["velocity"]], dtype=np.float32)
        last_actions = np.array(self._last_actions, dtype=np.float32).flatten()

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

        # flat observations
        flat_observation = np.concatenate([last_actions,
                                           proc_ranges,
                                           velocity], axis=0)
        return flat_observation
