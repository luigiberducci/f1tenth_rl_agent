# F1Tenth RL Node in ROS
Implementation of a ROS node to deploy a torch model of a RL agent on an [F1Tenth](https://f1tenth.org) racecar.

[![IMAGE ALT TEXT HERE](res/video.png)](https://drive.google.com/file/d/1-VEucXos_Dgt9a-In_9JjS4aum47JBjJ/view?usp=sharing)


We trained the model using SAC from [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) library.
The details on training are reported in the [manuscript](https://arxiv.org/abs/2110.02792).


## Content

We describe the steps from having a trained model to deploying it on the car:
1. **Model preparation**: how to convert sb3 model to torch-script.
2. **Robustness Test**: assess the agent robustness with **domain randomization**.
3. **ROS Porting Test**: run the **node** in ROS to ensure the model is loaded correctly.
4. **Deployment**: the fun part!


## The ROS node

For better portability, we decided to implement the node `rl_node` to run in a docker container.
So, the first step is to build the docker image.

Run the following commands from the project directory:
```
cd <path-to-current-dir>/rl_node/
docker build -t racecar/rl_node .
```

This image `racecar/rl_node` will be then used by node, as in the `docker_starter.sh`.


## Model preparation
Since we don't want to depend on `sb3` policies after training, 
we first export the core model from sb3 format to `torch-script`.

![alt text](res/sb3_to_torchscript.png)


The exported model is a minimal implementation for inference, 
so we assume to run it in deterministic mode
(*we won't keep the entire policy, e.g., we discard the stddev and keep only the mean action*).

The script `rl_node/utils/convert_sb3_to_torch.py` implement the export for policy trained with `SAC`.
The conversion depends on `sb3` and `torch`, even if the deployed node won't need `sb3`.
For this reason, the additional dependencies are reported in `rl_node/utils/requirements.txt`.

1. To convert a model `rl_node/checkpoint/sb3/<model-filename>.zip`,
you can run the following command from the project directory:
```
cd rl_node/utils/
python convert_sb3_to_torch.py --model_file ../checkpoints/sb3/<model-filename>.zip --output_dir ../checkpoints
```

If the `rl_node` package is not found, you need to add the project directory to the `PYTHONPATH`.
You can run the following command instead of the previous one:
```
export PYTHONPATH="$PYTHONPATH;$(pwd)"

cd rl_node/utils/
python convert_sb3_to_torch.py --model_file ../checkpoints/sb3/<model-filename>.zip --output_dir ../checkpoints
```

The output will be a torch-script model `node_rl/checkpoints/<model-filename>.pt`.

2. You also need to create a configuration file for the model in the `checkpoints` directory, 
describing the policy class, observation and action configurations.
For the `Agent64` implementation, look at any sample file, e.g., `rl_node/checkpoints/<model-filename>.yaml`

## Robustness Test
The first sanity check is to test the model in the training environment.
To assess its robustness to small discrepancies with the real world,
we simulate the agent with domain randomization.

To reproduce this step, we recommend to create a separate virtual environment for testing.
Since the gym environment is only needed for testing,
we describe the requirements in `rl_node/test/requirements.txt`.

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


2. To test a model in `rl_node/checkpoints/<model-filename>.pt` of type `Agent64`, you can run the following script:
```
cd rl_node/test
python run_gym_env.py -f <model-filename> --n_episodes <n-rnd-episodes> -no_sb3 
```

For example, to test the model `rl_node/checkpoints/torch_single_model_20220730.pt` for 100 episodes, you can run:
```
python run_gym_env.py -f torch_single_model_20220730 --n_episodes 100 -no_sb3 
```

Again, if the `rl_node` package is not found, you need to add the project directory to the `PYTHONPATH`.


The simulation will use the `racecar_gym` environment, loading the map of the `lecture_hall`.
At each episode, the simulation parameters will be randomized according to the ranges defined in
the scenario file `rl_node/test/racecar_scenario.yaml`.

![alt text](res/racecar_gym_dr.gif)

## ROS Porting Test
Having validated the model robustness in the environment, 
we finally test the ROS node to ensure the model is loaded correctly
and the node correctly subscribes to the sensor topics.

We assume the `f1tenth-simulator` and the `rl_node` are in the ros workspace.
For installation of the `f1tenth-simulator`, refer to the [documentation](https://f1tenth.readthedocs.io/en/stable/going_forward/simulator/index.html).

1. Catkin make and source the ros workspace:
```
source /opt/ros/<ros-distro>/setup.bash
catkin-make
source devel/setup.bash
```

2. In one terminal, launch the simulator:
```
roslaunch rl_node simulator.launch
```

3. In another terminal, launch the agent node, specifying the yaml file to control topics and adaptation parameters:
```
roslaunch rl_node only_agent.launch params:=params_sim.yaml
```

The node will load the model as specified in the params file.
Moreover, to account for easy adaptation to the real world,
the node will scale the actions using proportional gains.


![alt text](res/f110_ros_sim.gif)

## Deploy on real car

If the previous steps are successful, you can deploy the node on the real car.

We recommend to first try with a constant velocity,
enabling the `debug_mode` in the params file and setting the `debug_speed` to the desired velocity.

1. From the car, launch the `rl_node` with the `hardware.launch` file.
```
roslaunch rl_node hardware.launch
```

It will start the `f1tenth system` and three nodes: 
1. `rl_node`: the node that publishes the action from the model. 
2. `safety_node`: the emergency braking system which stops the car based on time-to-collision (ttc) estimation.
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
