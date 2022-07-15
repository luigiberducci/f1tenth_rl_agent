#!/usr/bin/env python3
import pathlib
import sys

# ROS Imports
import rospy
import rospkg
import yaml

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from agents.agent64 import Agent64

class AgentNode:
    _model_is_loaded = False

    def __init__(self):
        # Read params
        scan_topic = rospy.get_param("/rl_node/scan_topic", default="/scan")
        odom_topic = rospy.get_param("/rl_node/odom_topic", default="/odom")
        drive_topic = rospy.get_param("/rl_node/nav_topic", default="/drive")
        model_file = rospy.get_param('/rl_node/model', default='model_20220714')

        frame_skip = rospy.get_param('/rl_node/frame_skip', default=10)
        steering_multiplier = rospy.get_param('/rl_node/steering_multiplier', default=1.0)
        speed_multiplier = rospy.get_param('/rl_node/speed_multiplier', default=1.0)

        debug_mode = rospy.get_param('/rl_node/debug_mode', default=False)
        debug_speed = rospy.get_param('/rl_node/debug_speed', default=1.0)

        # Topics & Subscriptions,Publishers
        self.lidar_sub = rospy.Subscriber(scan_topic, LaserScan, self.lidar_callback, queue_size=1)
        self.vel_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1)

        # Internal variables
        self._current_speed = 0.0
        self._multipliers = {"steering": steering_multiplier, "speed": speed_multiplier}
        self._debug = {"enabled": debug_mode, "debug_speed": debug_speed}
        self._ctrl_interval = rospy.Duration(0.01 * frame_skip)
        self._last_time = rospy.Time()

        # Load model
        self.agent, result_load = self.build_agent(model_file)
        self._model_is_loaded = result_load

        # debug
        rospy.loginfo(f"[**INFO**] Model and param configuration have been loaded.")

    def build_agent(self, model_file):
        agent_config_filepath = pathlib.Path(f"checkpoints/{model_file}.yaml")
        checkpoint_filepath = pathlib.Path(f"checkpoints/{model_file}.pt")
        agent = Agent64(agent_config_filepath)
        result_load = agent.load(checkpoint_filepath)
        return agent, result_load

    def odom_callback(self, data):
        """ store the current speed from vesc """
        self._current_speed = data.twist.twist.linear.x

    def lidar_callback(self, data):
        """
        Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        if not self._model_is_loaded or (rospy.Time.now() - self._last_time) < self._ctrl_interval:
            return
        self._last_time = rospy.Time.now()

        observation = {"lidar": data.ranges, "velocity": self._current_speed}
        action = self.agent.get_action(observation, normalized=False)
        steer, speed = action["steering"], action["speed"]

        steer, speed = self.adaptation(steer, speed, self._multipliers)
        speed = self._debug["debug_speed"] if self._debug["enabled"] else speed

        self._drive(steer, speed)
        rospy.loginfo(f"Action: angle: {steer}, speed: {speed}\n")

    @staticmethod
    def adaptation(steer, speed, multipliers):
        steer *= multipliers["steering"]
        speed *= multipliers["speed"]
        return steer, speed

    def _drive(self, angle, speed):
        # Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)


def main(args):
    #param_file = pathlib.Path("cfg/simulation_params.yaml")
    #result_load = load_params(param_file)
    #assert result_load, "failed to load parameters"

    rospy.init_node("rl_node", anonymous=True)
    node = AgentNode()

    rospy.sleep(0.1)
    rospy.spin()


def load_params(param_file: pathlib.Path):
    with open(param_file, "r") as f:
        params = yaml.load(f, yaml.Loader)
    for k, v in params.items():
        rospy.set_param(k, v)
    return True


def callback(config, level):
    rospy.loginfo(f"Reconfigure Request: {config}")
    return config


if __name__ == '__main__':
    main(sys.argv)
