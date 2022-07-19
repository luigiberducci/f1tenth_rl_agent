#!/usr/bin/env python3
import sys
from dataclasses import dataclass

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from ddynamic_reconfigure_python.ddynamic_reconfigure import DDynamicReconfigure


class FilterNode:
    @dataclass
    class FilterNodeConfig:
        # topics
        odom_topic: str = "/odom"
        imu_topic: str = "/imu/data"
        filter_topic: str = "/filter/odom"

        # filter
        velx_threshold: float = 0.10
        accx_threshold: float = 0.25

        def update(self, params: dict):
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def __init__(self):
        # Read params
        self.config = self.read_params()

        # Topics & Subscriptions,Publishers
        self.odom_sub = rospy.Subscriber(self.config.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.imu_sub = rospy.Subscriber(self.config.imu_topic, Imu, self.imu_callback, queue_size=1)
        self.filter_pub = rospy.Publisher(self.config.filter_topic, Odometry, queue_size=1)

        # Internal variables
        self._first_callback = True
        self._velx = 0.0
        self._accx = 0.0
        self._filtered_vel = 0.0
        self._time = rospy.Time.now()

    def read_params(self):
        config = self.FilterNodeConfig()
        config.odom_topic = rospy.get_param("/filter_node/odom_topic", default="/vesc/odom")
        config.imu_topic = rospy.get_param("/filter_node/imu_topic", default="/imu/data")
        config.filter_topic = rospy.get_param("/filter_node/filter_topic", default="/filter/odom")
        config.velx_threshold = rospy.get_param('/filter_node/velx_threshold', default=0.0)
        config.accx_threshold = rospy.get_param('/filter_node/accx_threshold', default=0.0)
        return config

    def odom_callback(self, data):
        """ store the current speed from vesc """
        velx = data.twist.twist.linear.x
        velx = 0.0 if abs(velx) < self.config.velx_threshold else velx
        self._velx = velx

    def imu_callback(self, data):
        """ store the current acceleration from imu """
        accx = data.linear_acceleration.x
        accx = 0.0 if abs(accx) < self.config.accx_threshold else accx
        dt = (data.header.stamp - self._time).to_sec()
        # estimate and publish velocity
        self._filtered_vel += dt * self._accx
        self._pub_velocity(self._filtered_vel)
        # update info
        self._accx = accx
        self._time = rospy.Time.now()

    def _pub_velocity(self, velx):
        # Publish odom message
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.twist.twist.linear.x = velx
        self.filter_pub.publish(odom_msg)

    def reconfigure_callback(self, config, level):
        if self._first_callback:
            rospy.loginfo(f"Skip First Reconfigure Request")
            self._first_callback = False
            return self.config
        rospy.loginfo(f"Reconfigure Request:")
        rospy.loginfo("\n\t" + "\n\t".join([f"{k}: {v}" for k, v in config.items() if k != "groups"]))
        self.config.update(config)
        return config


def main(args):
    rospy.init_node("node", anonymous=True)
    node = FilterNode()

    # Create a D(ynamic)DynamicReconfigure
    ddynrec = DDynamicReconfigure("dyn_rec")

    # Add variables (name, description, default value, min, max, edit_method)
    ddynrec.add_variable("odom_topic", "string variable", "/vesc/odom")
    ddynrec.add_variable("imu_topic", "string variable", "/imu/data")
    ddynrec.add_variable("filter_topic", "string variable", "/filter/odom")

    ddynrec.add_variable("velx_threshold", "float/double variable", 0.0, 0.0, 1.0)
    ddynrec.add_variable("accx_threshold", "float/double variable", 0.0, 0.0, 1.0)

    # Start the server
    ddynrec.start(callback=node.reconfigure_callback)

    rospy.sleep(0.1)
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
