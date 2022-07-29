#!/usr/bin/env python3
import pathlib
import sys

# ROS Imports
import rospy
from ddynamic_reconfigure_python.ddynamic_reconfigure import DDynamicReconfigure
from marshmallow_dataclass import dataclass

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from agents.agent64 import Agent64


class AgentNode:
    _model_is_loaded = False

    @dataclass
    class NodeConfig:
        scan_topic: str = "/scan"
        odom_topic: str = "/odom"
        drive_topic: str = "/drive"
        model_file: str = "none"

        frame_skip: int = 10
        steering_multiplier: float = 1.0
        speed_multiplier: float = 1.0
        min_speed: float = 1.0

        debug_mode: bool = False
        debug_speed: float = 1.0

        def update(self, params: dict):
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def __init__(self):
        # Read params
        self.config = self.read_params()

        # Topics & Subscriptions,Publishers
        self.lidar_sub = rospy.Subscriber(self.config.scan_topic, LaserScan, self.lidar_callback, queue_size=1)
        self.vel_sub = rospy.Subscriber(self.config.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(self.config.drive_topic, AckermannDriveStamped, queue_size=1)

        # Internal variables
        self._first_callback = True
        self.agent = None
        self._model_is_loaded = False
        self._current_speed = 0.0
        self._last_time = rospy.Time()

    def read_params(self):
        config = self.NodeConfig()
        config.scan_topic = rospy.get_param("/node/scan_topic", default="/scan")
        config.odom_topic = rospy.get_param("/node/odom_topic", default="/odom")
        config.drive_topic = rospy.get_param("/node/nav_topic", default="/drive")
        config.model_file = rospy.get_param('/node/model', default='torch_model_20220714')

        config.frame_skip = rospy.get_param('/node/frame_skip', default=10)
        config.steering_multiplier = rospy.get_param('/node/steering_multiplier', default=1.0)
        config.speed_multiplier = rospy.get_param('/node/speed_multiplier', default=1.0)
        config.min_speed = rospy.get_param('/node/min_speed', default=1.0)

        config.debug_mode = rospy.get_param('/node/debug_mode', default=False)
        config.debug_speed = rospy.get_param('/node/debug_speed', default=1.0)
        return config

    def build_agent(self, model_file):
        agent_config_filepath = pathlib.Path(f"checkpoints/{model_file}.yaml")
        checkpoint_filepath = pathlib.Path(f"checkpoints/{model_file}.pt")
        if not (agent_config_filepath.exists() and checkpoint_filepath.exists()):
            return None, False
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
        if not self._model_is_loaded:
            # Load model
            self.agent, result_load = self.build_agent(self.config.model_file)
            if not result_load:
                rospy.loginfo(f"[**INFO**] No model loaded, model file: {self.config.model_file}")
                return

            self._model_is_loaded = result_load
            rospy.loginfo(f"[**INFO**] Model and param configuration have been loaded.")

        if (rospy.Time.now() - self._last_time) < rospy.Duration(0.01 * self.config.frame_skip):
            return
        self._last_time = rospy.Time.now()

        observation = {"lidar": data.ranges, "velocity": self._current_speed}
        norm_action, unnorm_action = self.agent.get_action(observation, self.config)
        steer, speed = unnorm_action["steering"], unnorm_action["speed"]

        #steer, speed = self.adaptation(steer, speed, self.config.speed_multiplier, self.config.steering_multiplier, self.config.min_speed)
        #speed = self.config.debug_speed if self.config.debug_mode else speed

        self._drive(steer, speed)
        rospy.loginfo(f"Topic: {self.config.drive_topic}, Action: angle: {steer}, speed: {speed}\n")

    @staticmethod
    def adaptation(steer, speed, speed_multiplier, steering_multiplier, min_speed):
        #speed *= speed_multiplier
        #speed = max(speed, min_speed)
        #steer *= steering_multiplier
        return steer, speed

    def _drive(self, angle, speed):
        # Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def reconfigure_callback(self, config, level):
        if self._first_callback:
            rospy.loginfo(f"Skip First Reconfigure Request")
            self._first_callback = False
            return config
        rospy.loginfo(f"Reconfigure Request:")
        rospy.loginfo("\n\t" + "\n\t".join([f"{k}: {v}" for k, v in config.items() if k != "groups"]))
        self.config.update(config)
        return config


def main(args):
    rospy.init_node("node", anonymous=True)
    node = AgentNode()

    # Create a D(ynamic)DynamicReconfigure
    ddynrec = DDynamicReconfigure("dyn_rec")

    # Add variables (name, description, default value, min, max, edit_method)
    ddynrec.add_variable("model_file", "string variable", "")

    ddynrec.add_variable("frame_skip", "integer variable", 10, 1, 100)
    ddynrec.add_variable("steering_multiplier", "float/double variable", 1.0, 0.0, 1.0)
    ddynrec.add_variable("speed_multiplier", "float/double variable", 1.0, 0.0, 1.0)
    ddynrec.add_variable("min_speed", "float/double variable", 1.0, 0.0, 2.0)

    ddynrec.add_variable("debug_mode", "bool variable", True)
    ddynrec.add_variable("debug_speed", "float/double variable", 1.5, 0.0, 3.0)

    # Start the server
    ddynrec.start(callback=node.reconfigure_callback)

    rospy.sleep(0.1)
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
