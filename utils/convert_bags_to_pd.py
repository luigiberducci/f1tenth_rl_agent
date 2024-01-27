import argparse
import pathlib

import rosbag
import matplotlib.pyplot as plt

TOPICS = {
    "simulation": {
        "drive": "/drive",
        "scan": "/scan",
        "odom": "/odom"
    },
    "hardware": {
        "drive": "/vesc/high_level/ackermann_cmd_mux/input/nav_0",
        "scan": "/scan",
        "odom": "/filter/odom"
    }
}

def extract_linear_acceleration_from_odom(msg):
    return msg.linear_acceleration.x

def extract_lateral_acceleration_from_odom(msg):
    return msg.linear_acceleration.y

def extract_speed_cmd_from_ackermann(msg):
    return msg.drive.speed

def extract_steering_cmd_from_ackermann(msg):
    return msg.drive.steering_angle

def main(args):
    data = {"time": [],
            "speed": [],
            "steer": [],
            "acc_x": [],
            "acc_y": [],
            }

    for file in args.files:
        bag = rosbag.Bag(file)
        n_msgs = 0

        platform = "simulation" if args.sim else "hardware"
        topics = [TOPICS[platform][topic] if topic in TOPICS[platform] else topic for topic in args.topics]

        for topic, msg, t in bag.read_messages(topics=topics):
            time_s = t.to_sec()
            speed_cmd = extract_speed_cmd_from_ackermann(msg)
            steer_cmd = extract_steering_cmd_from_ackermann(msg)
            accx = extract_linear_acceleration_from_odom(msg)
            accy = extract_lateral_acceleration_from_odom(msg)

            data["speed"].append(speed_cmd)
            data["steer"].append(steer_cmd)
            data["acc_x"].append(accx)
            data["acc_y"].append(accy)

            data["time"].append(time_s)

            n_msgs += 1
        bag.close()
        print(f"[info] processed {n_msgs} messages for {len(args.topics)} topics")

        plt.plot(data["time"], data["speed"], label="speed")
        plt.plot(data["time"], data["steer"], label="steer")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=pathlib.Path)
    parser.add_argument("--topics", nargs="+", type=str, default=TOPICS["hardware"].keys())
    parser.add_argument("-sim", action="store_true")
    args = parser.parse_args()
    main(args)
