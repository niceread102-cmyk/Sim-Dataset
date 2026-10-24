import os
import sys
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from scipy.spatial.transform import Rotation

DEFAULT_RPY = (180.0, 0.0, 0.0)
SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Simulation/
sys.path.insert(0, os.path.join(SIM, 'isaacsim'))
import sim_config  # noqa: E402  the maps and robots of isaacsim/configs/


def lens_calibration(robot=None, map_name=None, di=None):
    """(flange -> lens 4 x 4, TCP offset in the flange frame) of the robot (robot:=, else scene.robot of the map:=): the pose of the link of wrist_camera
    (configs/robot/<robot>.yaml) in the end frame of its ik_node.yaml, read from the URDF (a real robot: its hand-eye calibration), and the TCP offset of that ik_node.yaml"""
    import make_robot_config as mrc
    robot, map_name = sim_config.with_current(robot, map_name)              # without robot:= / map:= the ones of the running robot_sim.py
    if di is not None:
        robot = sim_config.option(sys.argv[1:], 'robot')                       # di:= picks the robot of the map (unless robot:= names one)
    robot = sim_config.robot_entry(sim_config.scene_of(map_name), robot, di)["name"]
    ry = sim_config.load_robot(robot)
    ik = yaml.safe_load(open(os.path.join(sim_config.BRINGUP_DIR, robot, 'config', 'ik_node.yaml')))
    T = mrc.Urdf(mrc.find_urdf(robot)).relative_pose(ik['ee_link'], ry['wrist_camera']['link'])
    if T is None:
        raise RuntimeError(f"{ry['wrist_camera']['link']} is not held to {ik['ee_link']} by fixed joints in the URDF of {robot}")
    return T, np.array(ik['tcp_offset'], float)


def lens_to_tcp(x, y, z, yaw, cal):
    """TCP pose (position, flange rotation) that puts the lens at (x, y, z) m, looking straight down, the image x axis turned by yaw deg about the
    vertical from the world +x (yaw = the angle of a crate: its long side along the image x axis)"""
    T_ee_lens, tcp = cal
    T_lens = np.eye(4)
    T_lens[:3, :3] = Rotation.from_euler('Z', yaw, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])    # optical z down, y down in the image
    T_lens[:3, 3] = [x, y, z]
    T_ee = T_lens @ np.linalg.inv(T_ee_lens)
    return T_ee[:3, 3] + T_ee[:3, :3] @ tcp, Rotation.from_matrix(T_ee[:3, :3])

class TargetPublisher(Node):
    def __init__(self):
        super().__init__('target_publisher')
        self.target_pub = self.create_publisher(Pose, '/target_pose', 10)
        self.variant_pub = self.create_publisher(String, '/place_variant', 10)    # state of the part for robot_sequence.py: standard | turn_over

    def publish_variant(self, variant):
        self.variant_pub.publish(String(data=variant))
        time.sleep(0.05)    # before the pose, so that the sequence has it when the target comes

    def publish_target(self, x, y, z, roll=None, pitch=None, yaw=None):
        if roll is None:
            roll, pitch, yaw = DEFAULT_RPY

        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z

        # r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True) # extrinsic
        r = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=True) # intrinsic        
        q = r.as_quat()  # [x, y, z, w]
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
    di = sim_config.option(sys.argv[1:], 'di')                                  # python3 ee_6d_di.py di:=<id>: the robot of the map with this ROS domain id
    rclpy.init(args=args, **({} if di is None else {'domain_id': int(di)}))    # the targets go to the robot of that domain (the launch with the same di:=)
    if di is not None:
        print(f"ROS domain {int(di)}: the robot of the launch with di:={di}")
    node = TargetPublisher()
    robot, map_name = sim_config.option(sys.argv[1:], 'robot'), sim_config.option(sys.argv[1:], 'map')    # python3 ee_6d.py [robot:=<name>] [map:=<map>] (default: scene.robot of the map)
    cal = None    # the lens calibration is read at the first `lens` command (robots without a wrist camera need none)

    try:
        while rclpy.ok():
            user_input = input("input target pos x y z [roll pitch yaw | qx qy qz qw] [standard | turn over] | lens x y z yaw (or type 'q' to quit): ")

            if user_input.lower() == 'q':
                break
            if user_input.lower().startswith('lens'):    # the lens of the wrist camera over (x, y, z), turned by yaw: the scan pose of a crate
                try:
                    x, y, z, *rest = map(float, user_input.split()[1:])
                    if cal is None:
                        cal = lens_calibration(robot, map_name, di)
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
            variant = ' '.join(values[n:]).lower().replace(' ', '_')   # a word after the numbers: the state of the part (standard | turn over), optional
            values = values[:n]
            try:
                if len(values) in (3, 6, 7) and variant:    # without it a sequence with `target: object` decides it from the pose of the part
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