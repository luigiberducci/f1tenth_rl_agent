#!/usr/bin/env python3
import sys
from dataclasses import dataclass

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


class FilterNode:

    @dataclass
    class FilterNodeConfig:
        odom_topic: str = "/odom"
        imu_topic: str = "/imu/data"
        filter_topic: str = "/filter/odom"

        def update(self, params: dict):
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def __init__(self):
        # Read params
        self.config = self.FilterNodeConfig()

        # Topics & Subscriptions,Publishers
        self.odom_sub = rospy.Subscriber(self.config.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.imu_sub = rospy.Subscriber(self.config.imu_topic, Imu, self.imu_callback, queue_size=1)
        self.filter_pub = rospy.Publisher(self.config.filter_topic, Odometry, queue_size=1)

        # Internal variables
        self._velx = 0.0
        self._accx = 0.0
        self._filtered_vel = 0.0
        self._time = rospy.Time.now()

    def odom_callback(self, data):
        """ store the current speed from vesc """
        velx = data.twist.twist.linear.x
        velx = 0.0 if abs(velx) < 0.1 else velx
        self._velx = velx

    def imu_callback(self, data):
        """ store the current acceleration from imu """
        self._accx = data.linear_acceleration.x
        dt = (data.header.stamp - self._time).to_sec()
        self._filtered_vel += dt * self._accx   # estimate velocity
        self._pub_velocity(self._filtered_vel)  # publish new odom
        self._time = rospy.Time.now()

    def _pub_velocity(self, velx):
        # Publish odom message
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.twist.twist.linear.x = velx
        self.filter_pub.publish(odom_msg)




def main(args):
    rospy.init_node("node", anonymous=True)
    node = FilterNode()

    rospy.sleep(0.1)
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)