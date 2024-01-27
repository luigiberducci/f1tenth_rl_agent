# Introduction
This repository contains the implementation of a ROS node 
for deploying a torch agent model on the [F1Tenth](https://f1tenth.org) car.

[![IMAGE ALT TEXT HERE](res/video.png)](https://drive.google.com/file/d/1-VEucXos_Dgt9a-In_9JjS4aum47JBjJ/view?usp=sharing)



# Building the node
For portability, the node runs in a docker container.

To build the docker image, run the following commands from the project directory:
```
cd <path-to-current-dir>/rl_node/
docker build -t racecar/rl_node .
```

This image will be then used to run the node, as defined in the `docker_starter.sh`.


# Running the node
We trained the model using SAC from [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) library.
The details on training are reported in the manuscript.

However, before deploying on the real racecar, 
we recommend the following steps:
1. Convert sb3 model to **torch-script**
2. **Test** the agent robustness to **domain randomization** with racecar-gym.
3. **Test** the **node** in ROS with the f1tenth-simulator.
4. Ok, now try to **deploy it**!

Details on each step follow. 

## Convert sb3 model to torch-script
Since we don't want to depend on `sb3` policies, 
we first export the core model from sb3 format to `torch-script`.

![alt text](res/sb3_to_torchscript.png)


The exported model is a minimal implementation for inference, 
so we assume to run it in deterministic mode
(*we won't keep the entire policy, e.g., we discard the stddev and keep only the mean action*).

The script `rl_node/utils/convert_sb3_to_torch.py` implement the export for policy trained with `SAC`.
The conversion depends on `sb3` and `torch`, even if the deployed node won't need `sb3`.
For this reason, the additional dependencies are reported in `rl_node/utils/requirements.txt`.

1. To convert a model `rl_node/checkpoint/sb3/model_20220714.zip`,
you can run the following command from the project directory:
```
cd rl_node/utils/
python convert_sb3_to_torch.py --model_file ../checkpoints/sb3/model_20220714.zip --output_dir ../checkpoints
```

If the `rl_node` package is not found, you need to add the project directory to the `PYTHONPATH`.
You can run the following command instead of the previous one:
```
export PYTHONPATH="$PYTHONPATH;$(pwd)"

rl_node/utils/
python convert_sb3_to_torch.py --model_file ../checkpoints/sb3/model_20220714.zip --output_dir ../checkpoints
```

The output will be a torch-script model `node_rl/checkpoints/model_20220714.pt`.

2. You also need to create a configuration file for the model in the `checkpoints` directory, 
describing the policy class, observation and action configurations.
For the `Agent64` implementation, look at any sample file, e.g., `rl_node/checkpoints/torch_model_20220714.yaml`

## Test in racecar gym
The first sanity check is to test the model in the training environment.

Since the gym environment in not needed once deployed in the real car, 
we recommend to create a separate virtual environment for testing
with the requirements in `rl_node/test/requirements.txt`.

1. Create a virtual environment and install the requirements:
```
cd rl_node/test
python3 -m venv venv
source venv/bin/activate

# the next two lines account for bug with gym 0.21.0
pip install pip==21
pip install setuptools==65.5.0 "wheel<0.40.0"

pip install -r requirements.txt
```

#### :warning: Note on observation preprocessing
In training, we heavily use gym-wrappers. You should ensure the same preprocessing is performed by the agent.

As an example, for the agent using 64 lidar beams, we implemented `Agent64`.

In general, each agent should implement the `rl_node.src.agents.agent.Agent` interface.

To the test `Agent64`, you can run the following script:
```
cd rl_node/test
python run_gym_env.py -f single_model_500000_20220730 -no_sb3 --n_episodes 10
```

Again, if the `rl_node` package is not found, you need to add the project directory to the `PYTHONPATH`.
You can run the following command instead of the previous one:
```
cd <path-to-project-dir>
export PYTHONPATH="$PYTHONPATH;$(pwd)"

cd rl_node/test
python run_gym_env.py -f model_20220714
```

The simulation in `racecar_gym` will run the agent in the `lecture_hall`
and randomize the environment parameters at each episode, in the 
range defined in the scenario file `rl_node/test/racecar_scenario.yaml`.

![alt text](res/racecar_gym_dr.gif)

## Test in f1tenth simulator
Having validated the model in the gym environment,
we now test the ROS node in the f1tenth simulator.

We assume the `f1tenth-simulator` and the `rl_node` are in the ros workspace.
For installation of the `f1tenth-simulator`, refer to the [documentation](https://f1tenth.readthedocs.io/en/stable/going_forward/simulator/index.html).

1. Catkin make and source the ros workspace:
```
catkin-make
source devel/setup.bash
```

3. In one terminal, launch the simulator:
```
roslaunch rl_node simulator.launch
```

2. In another terminal, launch the agent node, specifying the yaml file to control topics and adaptation parameters:
```
roslaunch rl_node only_agent.launch params:=params_sim.yaml
```

You should see the ros node controlling the car in the simulator.

![alt text](res/f110_ros_sim.gif)

## Deploy on real car

If the previous steps are successful, you can deploy the node on the real car.

To to that, you can launch the `rl_node` with the `hardware.launch` file.
```
roslaunch rl_node hardware.launch
```

A part all the `f1tenth system`, it will start three nodes: 
1. `rl_node`: the inference node using the trained model. 
2. `safety_node`: the emergency braking system.
3. `filter_node`: a velocity filter to publish the current velocity estimation. 
In its simplest implementation, it simply forwards the velocity given by the vesc rpm conversion.


# Citation
If you use this code for your research, please cite our paper:

```
@article{berducci2021hierarchical,
  title={Hierarchical potential-based reward shaping from task specifications},
  author={Berducci, Luigi and Aguilar, Edgar A and Ni{\v{c}}kovi{\'c}, Dejan and Grosu, Radu},
  journal={arXiv preprint arXiv:2110.02792},
  year={2021}
}
```
