import pathlib
from typing import Dict

import gym
from racecar_gym import MultiAgentScenario
from racecar_gym.agents import FollowTheGap, FollowTheWall
from racecar_gym.envs import gym_api
import numpy as np
from stable_baselines3 import SAC

from rl_node.src.agents.agent64 import Agent64
import matplotlib.pyplot as plt


def make_env():
    scenario = MultiAgentScenario.from_spec(
        path='multi_racecar_scenario.yml',
        rendering=True
    )
    env = gym_api.MultiAgentRaceEnv(scenario=scenario)
    env.seed(0)
    return env


def make_agents(model_name: str):
    model_file = f"torch_{model_name}"  # this is the torch-jit model
    agent_config_filepath = pathlib.Path(f"../checkpoints/{model_file}.yaml")
    checkpoint_filepath = pathlib.Path(f"../checkpoints/{model_file}.pt")
    agent = Agent64(agent_config_filepath)
    result_load = agent.load(checkpoint_filepath)
    assert result_load

    sb3_model = SAC.load(f"../checkpoints/sb3/multi/{model_name}.zip")  # this is the baseline model
    sb3_agent = Agent64(agent_config_filepath)
    sb3_agent.model = sb3_model

    return agent, sb3_agent


def check_dict_actions(action1: Dict[str, float], action2: Dict[str, float], tolerance: float = 1e-4):
    for a in ["speed", "steering"]:
        if abs(action1[a] - action2[a]) > tolerance:
            return False
    return True


def generic_test(model_filename: str, n_episodes: int):
    env = make_env()
    agent, sb3_agent = make_agents(model_filename)

    npc = FollowTheGap()
    # npc = FollowTheWall()

    frame_skip = 10

    env.seed(0)
    # check_env(env)

    dist_comfs = []

    for ep in range(n_episodes):
        done = False
        obs = env.reset(mode='grid')
        npc.reset({"base_speed": 1.25,
                   "variable_speed": 0.0,
                   "gap_threshold": 1.5})

        t = 0
        action = unnorm_action = {}

        steering_history = []
        speed_history = []
        actual_speed_history = []
        distance_history = []

        while not done:  # and t < 100:

            for agent_id in env.scenario.agents:
                custom_obs = {"lidar": obs[agent_id]["lidar"],
                              "velocity": obs[agent_id]["velocity"][0]}

                if action is None or t % frame_skip == 0:
                    if agent_id == "A":
                        norm_action, _ = npc.get_action(custom_obs, return_norm_actions=True)
                    else:
                        norm_action1, action1 = sb3_agent.get_action(custom_obs)
                        norm_action2, action2 = agent.get_action(custom_obs)

                        assert check_dict_actions(action1,
                                                  action2), f"action differs, action != sb3_action\n{action1} != {action2}"
                        norm_action = norm_action1
                        unnorm_action = action1

                    action[agent_id] = norm_action

            obs, rewards, dones, states = env.step(action)
            done = any(dones.values())
            # time.sleep(0.01)
            t += 1

            agents_distance = ((states["B"]["lap"] + states["B"]["progress"]) - (states["A"]["lap"] + states["A"]["progress"])) * 13.5

            steering_history.append(unnorm_action["steering"])
            speed_history.append(unnorm_action["speed"])
            actual_speed_history.append(obs["B"]["velocity"][0])
            distance_history.append(agents_distance)



        print(f"[info] simulation time: {obs['B']['time']}")
        times = np.linspace(0, 0.01 * (t + 1), t)
        plt.plot(times, steering_history, label="steering cmd")
        plt.plot(times, speed_history, label="speed cmd")
        plt.plot(times, actual_speed_history, label="actual speed")

        mind, maxd = -2, -1
        plt.plot(times, distance_history, label="ego2npc distance (m)")
        plt.hlines(y=-0.3, xmin=times[0], xmax=times[-1], label="safety dist (m)", color="red")
        plt.hlines(y=maxd, xmin=times[0], xmax=times[-1], label="max comf dist (m)")
        plt.hlines(y=mind, xmin=times[0], xmax=times[-1], label="min comf dist (m)")
        plt.legend()
        plt.show()

        dist_perc = np.sum([1 for d in distance_history if mind <= d <= maxd]) / len(distance_history)
        dist_comfs.append(dist_perc)
    env.close()

    print(f"mean comf sat: {np.mean(dist_comfs)}, {dist_comfs}")
    assert True


if __name__ == "__main__":
    """
    example: python run_gym_env.py -f model_20220714
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filename", "-f", type=pathlib.Path, required=True)
    parser.add_argument("--n_episodes", "-n", type=int, default=1)
    args = parser.parse_args()

    generic_test(args.model_filename, args.n_episodes)
