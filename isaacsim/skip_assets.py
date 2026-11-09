#!/usr/bin/env python3
"""skip_assets.py: type a layer number of the pallets (1 = bottom ... top) and the crates of that layer with the parts in them disappear from the running robot_sim.py; type the same number again and they come back as they were.

    python3 isaacsim/skip_assets.py        (q or an empty line quits; talks to robot_sim.py over /crate_layer_toggle, std_msgs/Int32)

Needs the ROS environment of robot_sim.py (same ROS domain).
"""
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

TOPIC = '/crate_layer_toggle'


def main():
    rclpy.init()
    node = Node('skip_assets')
    pub = node.create_publisher(Int32, TOPIC, 10)
    end = time.time() + 5.0
    while pub.get_subscription_count() == 0 and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print('robot_sim.py is not running (nobody listens on ' + TOPIC + ').')
        rclpy.shutdown()
        return
    try:
        while True:
            text = input('layer number (1 = bottom, again = back, q = quit): ').strip()
            if text.lower() in ('', 'q'):
                break
            try:
                number = int(text)
            except ValueError:
                print('type a layer number, e.g. 6')
                continue
            pub.publish(Int32(data=number))
            rclpy.spin_once(node, timeout_sec=0.1)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
