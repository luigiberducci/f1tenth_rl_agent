import collections
import pathlib
from dataclasses import dataclass
from typing import Dict

from yamldataclassconfig import YamlDataClassConfig
import numpy as np
from stable_baselines3 import SAC

from rl_node.utils import utils
from rl_node.agent.agent import Agent


@dataclass
class ActionConfig(YamlDataClassConfig):
    min_steering: float = None
    max_steering: float = None
    min_speed: float = None
    max_speed: float = None


@dataclass
class ObservationConfig(YamlDataClassConfig):
    scan_size: int = None
    min_range: float = None
    max_range: float = None
    n_last_cmd: int = None


@dataclass
class Agent64Config(YamlDataClassConfig):
    observation_config: ObservationConfig = None
    action_config: ActionConfig = None


class Agent64(Agent):

    def __init__(self, config_filepath: pathlib.Path):
        self.config = Agent64Config()
        self.config.load(config_filepath)
        self.model = None
        self._last_actions = None
        self.reset()

    def reset(self):
        n_last_actions = self.config.observation_config.n_last_cmd
        self._last_actions = collections.deque([[0.0] * 2] * n_last_actions, maxlen=n_last_actions)

    def load(self, model_filepath: pathlib.Path) -> bool:
        self.model = SAC.load(model_filepath, print_system_info=True)
        return True

    def get_action(self, observation: Dict[str, float], normalized=False) -> Dict[str, float]:
        observation_proc = self.preprocess_observation(observation)
        assert self.model.observation_space.contains(observation_proc), "fail sanity check on obs space"

        flat_action, _ = self.model.predict(observation_proc, deterministic=True)
        self._last_actions.append(flat_action)

        norm_speed, norm_steering = flat_action

        if normalized:
            speed, steer = norm_speed, norm_steering
        else:
            speed = utils.linear_scaling(norm_speed,
                                         [-1, +1],
                                         [self.config.action_config.min_speed, self.config.action_config.max_speed])
            steer = utils.linear_scaling(norm_steering,
                                         [-1, +1],
                                         [self.config.action_config.min_steering, self.config.action_config.max_steering])
        return {"speed": speed, "steering": steer}

    def preprocess_observation(self, observation: Dict[str, float]):
        ranges = np.array(observation["lidar"], dtype=np.float32)
        velocity = np.array([observation["velocity"]], dtype=np.float32)
        last_actions = np.array(self._last_actions, dtype=np.float32).flatten()

        # preprocess velocity
        velocity = utils.linear_scaling(velocity,
                                        [self.config.action_config.min_speed,self.config.action_config.max_speed],
                                        [-1, +1])

        # preprocess lidar: down-sample and scale
        n_beams = self.config.observation_config.scan_size
        min_range = self.config.observation_config.min_range
        max_range = self.config.observation_config.max_range

        selected_ranges = [ranges[int(len(ranges) // (n_beams - 1) * i)] for i in range(n_beams)]
        selected_ranges = np.nan_to_num(selected_ranges, nan=max_range)
        proc_ranges = utils.linear_scaling(selected_ranges, [min_range, max_range], [-1, +1], clip=True)

        # flat observations
        flat_observation = np.concatenate([last_actions,
                                           proc_ranges,
                                           velocity], axis=0)
        return flat_observation
