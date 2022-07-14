#!/bin/bash


IMAGE_NAME="rshaping:latest"
PY_CMD="roslaunch reward_shaping_node only_agent.launch"
NODE_DIR=$(rospack find reward_shaping_node)

CMD="docker run -it --rm --net host -v $NODE_DIR:/reward_shaping_node $IMAGE_NAME $PY_CMD"
echo $CMD
$($CMD)