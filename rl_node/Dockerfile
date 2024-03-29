FROM ubuntu:focal

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ros-noetic-tf2-geometry-msgs \
    ros-noetic-ackermann-msgs ros-noetic-joy ros-noetic-map-server \
    ros-noetic-dynamic-reconfigure 

# install pip
RUN apt-get update && apt install -y curl build-essential git
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

# install dependencies
COPY . /rl_node
WORKDIR /rl_node
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 11311

ENTRYPOINT ["./ros_entrypoint.sh"]
CMD ["python3", "src/node.py"]
