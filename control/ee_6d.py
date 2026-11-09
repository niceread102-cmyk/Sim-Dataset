"""ee_6d.py: type a target, it goes out on /target_pose (geometry_msgs/Pose: TCP in the world frame) to ik_node.py or robot_sequence.py.

    python3 control/ee_6d.py [robot:=<name>] [sequence:=<name>] [id:=<ROS domain id>]

Input: x y z [roll pitch yaw | qx qy qz qw] [standard | turn over], or `lens x y z [yaw]` = the TCP that puts the wrist camera lens there.
id:= sends in that ROS domain (the ROS domain id of the launch). `lens` needs robot:= or sequence:= (the robot of the sequence).
"""
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from scipy.spatial.transform import Rotation

import cell_config

DEFAULT_RPY = (180.0, 0.0, 0.0)


def fixed_pose(urdf, a, b):
    parent = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}

    def up(n):
        out = [n]
        while n in parent:
            n = parent[n].find('parent').get('link')
            out.append(n)
        return out

    def frame(n, top):
        T = np.eye(4)
        while n != top:
            j = parent[n]
            if j.get('type') != 'fixed':
                return None
            o = j.find('origin')
            step = np.eye(4)
            if o is not None:
                step[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in (o.get('rpy') or '0 0 0').split()]).as_matrix()
                step[:3, 3] = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
            T = step @ T
            n = j.find('parent').get('link')
        return T
    pa, pb = up(a), up(b)
    top = next((n for n in pa if n in pb), None)
    Ta, Tb = (frame(a, top), frame(b, top)) if top is not None else (None, None)
    return None if Ta is None or Tb is None else np.linalg.inv(Ta) @ Tb


def lens_calibration(robot=None, sequence=None):
    if not robot:
        robot = cell_config.load_sequence_file(cell_config.sequence_of(sequence)).get('robot')
    robot = cell_config.pick_robot(robot)
    ry = cell_config.load_robot(robot)
    ik = yaml.safe_load(open(cell_config.BRINGUP_DIR / robot / 'config' / 'ik_node.yaml'))
    T = fixed_pose(cell_config.find_urdf(robot), ik['ee_link'], ry['wrist_camera']['link'])
    if T is None:
        raise RuntimeError(f"{ry['wrist_camera']['link']} is not held to {ik['ee_link']} by fixed joints in the URDF of {robot}")
    return T, np.array(ik['tcp_offset'], float)


def lens_to_tcp(x, y, z, yaw, cal):
    T_ee_lens, tcp = cal
    T_lens = np.eye(4)
    T_lens[:3, :3] = Rotation.from_euler('Z', yaw, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    T_lens[:3, 3] = [x, y, z]
    T_ee = T_lens @ np.linalg.inv(T_ee_lens)
    return T_ee[:3, 3] + T_ee[:3, :3] @ tcp, Rotation.from_matrix(T_ee[:3, :3])

class TargetPublisher(Node):
    def __init__(self):
        super().__init__('target_publisher')
        self.target_pub = self.create_publisher(Pose, '/target_pose', 10)
        self.variant_pub = self.create_publisher(String, '/place_variant', 10)

    def publish_variant(self, variant):
        self.variant_pub.publish(String(data=variant))
        time.sleep(0.05)

    def publish_target(self, x, y, z, roll=None, pitch=None, yaw=None):
        if roll is None:
            roll, pitch, yaw = DEFAULT_RPY

        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z

        r = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=True)
        q = r.as_quat()
        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]

        self.target_pub.publish(msg)
        self.get_logger().info(
            f'Published: pos=[{x:.3f}, {y:.3f}, {z:.3f}] '
            f'rpy=[{roll:.4f}, {pitch:.4f}, {yaw:.4f}] '
            f'quat=[{q[3]:.6f}, {q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}]')

    def publish_target_quat(self, x, y, z, qx, qy, qz, qw):
        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        self.target_pub.publish(msg)
        self.get_logger().info(
            f'Published: pos=[{x:.3f}, {y:.3f}, {z:.3f}] '
            f'quat=[{qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}]')

def is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def main(args=None):
    id = cell_config.option(sys.argv[1:], 'id')
    rclpy.init(args=args, **({} if id is None else {'domain_id': int(id)}))
    if id is not None:
        print(f"ROS domain {int(id)}: the robot of the launch with id:={id}")
    node = TargetPublisher()
    robot, sequence = cell_config.option(sys.argv[1:], 'robot'), cell_config.option(sys.argv[1:], 'sequence')
    cal = None

    try:
        while rclpy.ok():
            user_input = input("input target pos x y z [roll pitch yaw | qx qy qz qw] [standard | turn over] | lens x y z yaw (or type 'q' to quit): ")

            if user_input.lower() == 'q':
                break
            if user_input.lower().startswith('lens'):
                try:
                    x, y, z, *rest = map(float, user_input.split()[1:])
                    if cal is None:
                        cal = lens_calibration(robot, sequence)
                        print(f"lens calibration (flange -> lens): xyz {np.round(cal[0][:3, 3], 4).tolist()} m, TCP offset {cal[1].tolist()} m")
                    pos, r = lens_to_tcp(x, y, z, rest[0] if rest else 0.0, cal)
                except (ValueError, RuntimeError, KeyError, TypeError) as e:
                    print(f"lens x y z [yaw]: {e}")
                    continue
                node.publish_target_quat(*pos, *r.as_quat())
                roll, pitch, yaw_ = r.as_euler('XYZ', degrees=True)
                print(f"lens at {x:.3f} {y:.3f} {z:.3f} -> TCP {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} {roll + 0.0:.1f} {pitch + 0.0:.1f} {yaw_ + 0.0:.1f}")
                continue

            values = user_input.split()
            n = next((i for i, v in enumerate(values) if not is_number(v)), len(values))
            variant = ' '.join(values[n:]).lower().replace(' ', '_')
            values = values[:n]
            try:
                if len(values) in (3, 6, 7) and variant:
                    node.publish_variant(variant)
                if len(values) == 3:
                    x, y, z = map(float, values)
                    node.publish_target(x, y, z)
                elif len(values) == 6:
                    x, y, z, roll, pitch, yaw = map(float, values)
                    node.publish_target(x, y, z, roll, pitch, yaw)
                elif len(values) == 7:
                    x, y, z, qx, qy, qz, qw = map(float, values)
                    node.publish_target_quat(x, y, z, qx, qy, qz, qw)
                else:
                    raise ValueError

            except ValueError:
                print("Invalid input. Enter 3 numbers (x y z), 6 numbers (x y z roll pitch yaw), or 7 numbers (x y z qx qy qz qw), optionally followed by standard | turn over.")
                
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
