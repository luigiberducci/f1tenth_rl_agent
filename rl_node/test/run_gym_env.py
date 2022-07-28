import pathlib
from typing import Dict

import gym
from racecar_gym import SingleAgentScenario
from racecar_gym.envs import gym_api
import numpy as np
from stable_baselines3 import SAC

from rl_node.src.agents.agent64 import Agent64


def make_env():
    scenario = SingleAgentScenario.from_spec(
        path='racecar_scenario.yml',
        rendering=True
    )
    env = gym_api.SingleAgentRaceEnv(scenario=scenario)
    env.seed(0)
    return env

def make_agents(model_name: str):
    model_file = f"torch_{model_name}"  # this is the torch-jit model
    agent_config_filepath = pathlib.Path(f"../checkpoints/{model_file}.yaml")
    checkpoint_filepath = pathlib.Path(f"../checkpoints/{model_file}.pt")
    agent = Agent64(agent_config_filepath)
    result_load = agent.load(checkpoint_filepath)
    assert result_load

    sb3_model = SAC.load(f"../checkpoints/sb3/{model_name}.zip")    # this is the baseline model
    sb3_agent = Agent64(agent_config_filepath)
    sb3_agent.model = sb3_model

    return agent, sb3_agent


def check_dict_actions(action1: Dict[str, float], action2: Dict[str, float], tolerance: float = 1e-4):
    for a in ["speed", "steering"]:
        if abs(action1[a] - action2[a]) > tolerance:
            return False
    return True


def generic_test(model_filename: str):
    env = make_env()
    agent, sb3_agent = make_agents(model_filename)

    frame_skip = 10

    done = False

    env.seed(0)
    #check_env(env)

    obs = env.reset(mode='grid')
    t = 0
    action = unnorm_action = None

    steering_history = []
    speed_history = []
    actual_speed_history = []

    while not done: # and t < 100:
        custom_obs = {"lidar": obs["lidar"],
                      "velocity": obs["velocity"][0]}

        if action is None or t % frame_skip == 0:
            norm_action1, action1 = sb3_agent.get_action(custom_obs)
            norm_action2, action2 = agent.get_action(custom_obs)

            assert check_dict_actions(action1, action2), f"action differs, action != sb3_action\n{action1} != {action2}"
            action = norm_action1
            unnorm_action = action1

        steering_history.append(unnorm_action["steering"])
        speed_history.append(unnorm_action["speed"])
        actual_speed_history.append(obs["velocity"][0])

        obs, rewards, done, states = env.step(action)
        #time.sleep(0.01)
        t += 1

    print(f"[info] simulation time: {obs['time']}")
    env.close()

    import matplotlib.pyplot as plt

    times = np.linspace(0, 0.01 * (t + 1), t)
    plt.plot(times, steering_history, label="steering cmd")
    plt.plot(times, speed_history, label="speed cmd")
    plt.plot(times, actual_speed_history, label="actual speed")
    plt.legend()
    plt.show()

    assert True


if __name__=="__main__":
    """
    example: python run_gym_env.py -f model_20220714
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filename", "-f", type=pathlib.Path, required=True)
    args = parser.parse_args()

    generic_test(args.model_filename)