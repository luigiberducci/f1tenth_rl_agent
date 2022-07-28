# Introduction
This repo implement a ros node running in docker and using a torch model for inference.
To build the docker image, run the following commands from the project directory:
```
cd rl_node
docker build -t racecar/rl_node .
```

This image will be then used to run the node, as defined in the `docker_starter.sh`.

Before deploying on the real racecar, the following steps are recommended:
1. Convert sb3 model to torch-script
2. Test in racecar gym
3. Test in f1tenth-simulator
4. Ok, now try to deploy it!

Details on each step follow. For the preliminary testing, we recommend the use of a virtual environment.


## Convert sb3 model to torch-script
Since we don't want to depend on `sb3` policies, we first export the core model to `torch-script`.
The exported model is a minimal implementation for inference, so we assume to run it in deterministic mode
(*we won't carry the entire policy, e.g., we discard the stddev and keep only the mean*).

The script `rl_node/utils_convert/convert_sb3_to_torch.py` implement the export for policy trained with `SAC`.
The conversion depends on `sb3` and `torch`, even if the deployed node won't need `sb3`.
For this reason, the additional dependencies are reported in `rl_node/utils_convert/requirements.txt`.

1. To convert a model `rl_node/checkpoint/sb3/model_20220714.zip`,
you can run the following command from the project directory:
```
cd rl_node/utils_convert/
python convert_sb3_to_torch.py --model_file ../checkpoints/sb3/model_20220714.zip --output_dir ../checkpoints
```

If the `rl_node` package is not found, you need to add the project directory to the `PYTHONPATH`.
You can run the following command instead of the previous one:
```
export PYTHONPATH="$PYTHONPATH;$(pwd)"

rl_node/utils_convert/
python convert_sb3_to_torch.py --model_file ../checkpoints/sb3/model_20220714.zip --output_dir ../checkpoints
```

The output will be a torch-script model `node_rl/checkpoints/model_20220714.pt`.

2. You also need to create a configuration file for the model in the `checkpoints` directory, 
describing the policy class, observation and action configurations.
For the `Agent64` implementation, look at any sample file, e.g., `rl_node/checkpoints/torch_model_20220714.yaml`

## Test in racecar gym
The first sanity check is to test the model in the gym environment.
Since the gym environment in not needed once deployed, 
we add it as additional requirement in `rl_node/test/requirements.txt`.

**Important:** In training, we heavily use gym-wrappers. You should ensure the same preprocessing is performed by the agent.

As an example, for the agent using 64 lidar beams, we implemented `Agent64`.
In general, each agent should implement the `rl_node.src.agents.agent.Agent` interface.

To the test `Agent64`, you can run the following script:
```
cd rl_node/test
python run_gym_env.py -f model_20220714
```

Again, if the `rl_node` package is not found, you need to add the project directory to the `PYTHONPATH`.
You can run the following command instead of the previous one:
```
export PYTHONPATH="$PYTHONPATH;$(pwd)"

cd rl_node/test
python run_gym_env.py -f model_20220714
```

The simulation in `racecar_gym` will run the agent in the `lecture_hall`,
and plot the action and velocity profile once completed 1 lap.

## Test in f1tenth simulator
To test the ros node, we assume the `f1tenth-simulator` is in the ros workspace.

1. Catkin make and source the ros workspace:
```
catkin-make
source devel/setup.bash
```

3. Launch the simulator:
```
roslaunch rl_node simulator.launch
```

2. Launch the agent node, specifying the yaml file to control topics and adaptation parameters:
```
roslaunch rl_node only_agent.launch params:=params_sim.yaml
```

## Deploy on real car
To deploy on the real car, launch the `rl_node` with the `hardware.launch` file.
```
roslaunch rl_node hardware.launch
```

A part all the `f1tenth system`, it will start three nodes: 
1. `rl_node`: the inference node using the trained model. 
2. `safety_node`: the emergency braking system.
3. `filter_node`: a velocity filter to publish the current velocity estimation. 
In its simplest implementation, it simply forwards the velocity given by the vesc rpm conversion.
