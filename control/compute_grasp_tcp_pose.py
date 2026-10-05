import numpy as np
from rclpy.node import Node

# Pose where the TCP points at the floor = Rx(180)
DEFAULT_TCP_R = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1],
                ])


# Angles (deg) -> rotation matrix. Intrinsic XYZ (= Rx @ Ry @ Rz), the same convention as from_euler('XYZ') in ee_6d.py
# (Note: rpy in ROS/URDF is extrinsic xyz = Rz @ Ry @ Rx, so the same numbers give a different result)
def rpy_to_R(roll, pitch, yaw):
    r, p, y = np.deg2rad([roll, pitch, yaw])

    Rx = np.array([
            [1,         0,          0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r),  np.cos(r)],
            ])

    Ry = np.array([
            [ np.cos(p), 0, np.sin(p)],
            [         0, 1,         0],
            [-np.sin(p), 0, np.cos(p)],
            ])

    Rz = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y),  np.cos(y), 0],
            [        0,          0, 1],
            ])

    return Rx @ Ry @ Rz


# Rotation matrix -> angles (deg). Inverse of rpy_to_R (intrinsic XYZ)
#   R[0,2] = sin(p),  R[1,2] = -sin(r)cos(p),  R[2,2] = cos(r)cos(p)
#   R[0,0] = cos(p)cos(y),  R[0,1] = -cos(p)sin(y)
def R_to_rpy(R):
    pitch = np.arctan2(R[0, 2], np.hypot(R[0, 0], R[0, 1]))
    if np.isclose(np.cos(pitch), 0):        # pitch = ±90 (gimbal lock): roll and yaw cannot be told apart -> set yaw = 0 and compute only roll
        roll = np.arctan2(R[2, 1], R[1, 1])
        yaw  = 0.0
    else:
        roll = np.arctan2(-R[1, 2], R[2, 2])
        yaw  = np.arctan2(-R[0, 1], R[0, 0])
    return np.rad2deg([roll, pitch, yaw])


def main(args=None):
    print("input object roll pitch yaw (deg) : ")
    r, p, y = map(float, input().split())
    print("R : %.1f, P : %.1f, Y : %.1f" % (r, p, y))
    print(np.round(rpy_to_R(r, p, y), 4))
    tcp_r, tcp_p, tcp_y = R_to_rpy(rpy_to_R(r, p, y) @ DEFAULT_TCP_R)
    print("TCP R : %.1f, P : %.1f, Y : %.1f" % (tcp_r, tcp_p, tcp_y))
    print("%.1f %.1f %.1f" % (tcp_r, tcp_p, tcp_y))   # to paste as-is after x y z into ee_6d.py


if __name__ == '__main__':
    main()
