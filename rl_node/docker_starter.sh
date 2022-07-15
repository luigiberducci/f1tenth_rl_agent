#!/bin/bash


IMAGE_NAME="rl_node:latest"
PY_CMD="python3 /rl_node/src/node.py"
NODE_DIR=$(rospack find rl_node)

CMD="docker run -it --rm --net host -v $NODE_DIR:/rl_node $IMAGE_NAME $PY_CMD"
echo $CMD
$CMD