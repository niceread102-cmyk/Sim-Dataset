"""compute_grasp_tcp_pose.py: object pose -> the TCP pose that grips it from above, and whether the object lies upside down (turn_over).

    python3 control/compute_grasp_tcp_pose.py        (asks for x y z roll pitch yaw; robot_sim.py and robot_sequence.py use tcp_rpy())

Angles in deg, intrinsic XYZ like ee_6d.py.
"""
import numpy as np

DEFAULT_TCP_R = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1],
                ])

FLIPPED_TCP_R = np.array([
                [-1,  0, 0],
                [ 0, -1, 0],
                [ 0,  0, 1],
                ])


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


def R_to_rpy(R):
    pitch = np.arctan2(R[0, 2], np.hypot(R[0, 0], R[0, 1]))
    if np.isclose(np.cos(pitch), 0):
        roll = np.arctan2(R[2, 1], R[1, 1])
        yaw  = 0.0
    else:
        roll = np.arctan2(-R[1, 2], R[2, 2])
        yaw  = np.arctan2(-R[0, 1], R[0, 0])
    return np.rad2deg([roll, pitch, yaw])


def tcp_rpy(roll, pitch, yaw):
    R = rpy_to_R(roll, pitch, yaw)
    flipped = bool(R[2, 2] < 0)
    tcp_r, tcp_p, tcp_y = R_to_rpy(R @ (FLIPPED_TCP_R if flipped else DEFAULT_TCP_R))
    return tcp_r, tcp_p, tcp_y, flipped


def main(args=None):
    print("input object [x y z] roll pitch yaw (deg) : ")
    v = list(map(float, input().split()))
    if len(v) not in (3, 6):
        print("Invalid input. Enter 3 numbers (roll pitch yaw) or 6 numbers (x y z roll pitch yaw).")
        return
    xyz = v[:-3]
    r, p, y = v[-3:]
    if xyz:
        print("x : %.3f, y : %.3f, z : %.3f" % tuple(xyz))
    tcp_r, tcp_p, tcp_y, flipped = tcp_rpy(r, p, y)
    print("R : %.1f, P : %.1f, Y : %.1f" % (r, p, y))
    print(np.round(rpy_to_R(r, p, y), 4))
    if flipped:
        print("upside down (hole axis points down): flipped back by 180 deg about its y, then gripped from above like an upright one")
    print("TCP R : %.1f, P : %.1f, Y : %.1f" % (tcp_r, tcp_p, tcp_y))
    print(" ".join(["%.3f" % c for c in xyz] + ["%.1f" % a for a in (tcp_r, tcp_p, tcp_y)] + ["turn_over" if flipped else "standard"]))


if __name__ == '__main__':
    main()
