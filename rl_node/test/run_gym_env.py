import pathlib
from typing import Dict

import gym
from racecar_gym import SingleAgentScenario
from racecar_gym.envs import gym_api
import time
import numpy as np
from stable_baselines3 import SAC

from rl_node.src.agents.agent64 import Agent64

env = gym.make("SingleAgentLecture_hall_Gui-v0")

model_name = "model_20220714"
model_file = f"torch_{model_name}"
frame_skip = 10

# make env
scenario = SingleAgentScenario.from_spec(
    path='racecar_scenario.yml',
    rendering=True
)
env = gym_api.SingleAgentRaceEnv(scenario=scenario)
env.seed(0)

# make agents
agent_config_filepath = pathlib.Path(f"../checkpoints/{model_file}.yaml")
checkpoint_filepath = pathlib.Path(f"../checkpoints/{model_file}.pt")
agent = Agent64(agent_config_filepath)
result_load = agent.load(checkpoint_filepath)
assert result_load

sb3_agent = SAC.load(f"../checkpoints/sb3/{model_name}.zip")

done = False

# Currently, there are two reset modes available: 'grid' and 'random'.
# Grid: Place agents on predefined starting position.
# Random: Random poses on the track.
obs = env.reset(mode='grid')
t = 0

action = None

steering_history = []
speed_history = []


def check_actions(action1: Dict[str, float], action2: Dict[str, float], tolerance: float = 1e-5):
    for a in ["speed", "steering"]:
        if abs(action1[a] - action2[a]) > tolerance:
            return False
    return True


while not done:
    custom_obs = {"lidar": obs["lidar"],
                  "velocity": obs["velocity"][0]}

    if action is None or t % frame_skip == 0:
        observation = agent.preprocess_observation(custom_obs)
        assert sb3_agent.observation_space.contains(observation)
        action, _ = sb3_agent.predict(observation, deterministic=True)
        action2 = agent.get_action(custom_obs, normalized=True)
        action = {"speed": action[0], "steering": action[1]}
        assert check_actions(action, action2), f"{action} != {action2}"
        unnorm_action = agent.get_action(custom_obs, normalized=False)

    steering_history.append(unnorm_action["steering"])
    speed_history.append(unnorm_action["speed"])

    obs, rewards, done, states = env.step(action)
    time.sleep(0.01)
    t += 1

env.close()

import matplotlib.pyplot as plt

times = np.linspace(0, 0.01 * (t + 1), t)
plt.plot(times, steering_history, label="steering")
plt.plot(times, speed_history, label="speed")
plt.legend()
plt.show()
