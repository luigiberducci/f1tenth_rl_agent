#!/usr/bin/env python
import rospy
import nav_msgs.msg as nav
import sensor_msgs.msg as sensors
import std_msgs.msg as std
import ackermann_msgs.msg as ackermann
import math

# safety node taken from f1tenth_demo
# https://github.com/CPS-TUWien/f1tenth_demo/blob/master/scripts/safety_brake.py

class Safety(object):

    def __init__(self, ttc_min, deceleration):
        self.speed = 0
        self.ttc = 100
        self.ttc_min = ttc_min
        self.deceleration = deceleration
        self.engaged = False
        self.odometry_sub = rospy.Subscriber('/vesc/odom', nav.Odometry, self.odom_callback, queue_size=1)
        self.joy_sub = rospy.Subscriber('/vesc/joy', sensors.Joy, self.joy_callback, queue_size=1)
        self.laser_sub = rospy.Subscriber('/scan', sensors.LaserScan, self.scan_callback, queue_size=1)
        self.emb_pub = rospy.Publisher('/brake_bool', std.Bool, queue_size=1)
        self.ackermann_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/safety', ackermann.AckermannDriveStamped, queue_size=1)
        print("safety_brake: INIT")

    def odom_callback(self, odom_msg):
        self.speed = odom_msg.twist.twist.linear.x

    def joy_callback(self, joy_msg):
        #print("safety_brake: joy callback: ", joy_msg)
        if joy_msg.buttons[0] == 1:
            if self.engaged != False:
                print("safety_brake: GO (joy button)".format(self.speed, self.ttc))
            self.engaged = False
        if joy_msg.buttons[1] == 1:
            if self.engaged != True:
                print("safety_brake: STOP (joy button)".format(self.speed, self.ttc))
            self.engaged = True

    def scan_callback(self, scan_msg):
        if self.should_stop(scan=scan_msg):
            self.emb_pub.publish(data=True)
            self.ackermann_pub.publish(drive=ackermann.AckermannDrive(speed=0))
        else:
            self.emb_pub.publish(data=False)

    def should_stop(self, scan):
        if self.engaged:
            return True

        self.ttc = 200
        self.distance = 0
        for i, range in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            distance = max(self.speed * math.cos(angle), 0)
            if distance > 0:
                ttc = range / distance
                time_to_stop = self.speed / self.deceleration
                if ttc < self.ttc:
                    self.ttc = ttc
                if ttc < self.ttc_min or ttc < time_to_stop:
                    print("safety_brake: speed={} ttc={} tts={} STOP".format(self.speed, ttc, time_to_stop))
                    self.engaged = True
                    return True
        return False

def main():
    rospy.init_node('safety_node')
    ttc_min = 0.2 # seconds
    max_deceleration = 18.0 # meters/second^2
    sn = Safety(ttc_min=ttc_min, deceleration=max_deceleration)
    rospy.spin()

if __name__ == '__main__':
    main()