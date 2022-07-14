import pathlib

import gym
from racecar_gym import SingleAgentScenario
from racecar_gym.envs import gym_api
import time

from rl_node.utils import utils
from rl_node.agent.agent64 import Agent64

env = gym.make("SingleAgentLecture_hall_Gui-v0")

model_file = "model_20220714"
frame_skip = 10

# make env
scenario = SingleAgentScenario.from_spec(
    path='racecar_scenario.yml',
    rendering=True
)
env = gym_api.SingleAgentRaceEnv(scenario=scenario)

# make agent
agent_config_filepath = pathlib.Path(f"../models/{model_file}.yaml")
checkpoint_filepath = pathlib.Path(f"../models/{model_file}.zip")
agent = Agent64(agent_config_filepath)
result_load = agent.load(checkpoint_filepath)
assert result_load

done = False

# Currently, there are two reset modes available: 'grid' and 'random'.
# Grid: Place agents on predefined starting position.
# Random: Random poses on the track.
obs = env.reset(mode='grid')
t = 0

action = None

steering_history = []
speed_history = []

while not done:
    custom_obs = {"lidar": obs["lidar"],
                  "velocity": obs["velocity"][0]}

    if action is None or t % frame_skip == 0:
        action = agent.get_action(custom_obs, normalized=True)

    steering_history.append(action["steering"])
    speed_history.append(action["speed"])

    obs, rewards, done, states = env.step(action)
    t += 1

env.close()


import matplotlib.pyplot as plt

plt.plot(steering_history, label="steering")
plt.plot(speed_history, label="speed")
plt.show()
