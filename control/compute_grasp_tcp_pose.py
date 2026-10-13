import numpy as np

# Pose where the TCP points at the floor = Rx(180)
DEFAULT_TCP_R = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1],
                ])

# Same grip for an object lying upside down (its hole axis, the object z, points down, e.g. flipped by 180 deg about its y):
# flip it back first, Ry(180), then use DEFAULT_TCP_R. Ry(180) @ Rx(180) = Rz(180). The TCP still points at the floor.
FLIPPED_TCP_R = np.array([
                [-1,  0, 0],
                [ 0, -1, 0],
                [ 0,  0, 1],
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
    print("input object [x y z] roll pitch yaw (deg) : ")
    v = list(map(float, input().split()))
    if len(v) not in (3, 6):
        print("Invalid input. Enter 3 numbers (roll pitch yaw) or 6 numbers (x y z roll pitch yaw).")
        return
    xyz = v[:-3]                            # empty when only rpy is given; x y z are just carried through, not used in the math
    r, p, y = v[-3:]
    if xyz:
        print("x : %.3f, y : %.3f, z : %.3f" % tuple(xyz))
    R = rpy_to_R(r, p, y)
    flipped = R[2, 2] < 0                   # the object z (hole axis) points below the horizon: the object is upside down
    print("R : %.1f, P : %.1f, Y : %.1f" % (r, p, y))
    print(np.round(R, 4))
    if flipped:
        print("upside down (hole axis points down): flipped back by 180 deg about its y, then gripped from above like an upright one")
    tcp_r, tcp_p, tcp_y = R_to_rpy(R @ (FLIPPED_TCP_R if flipped else DEFAULT_TCP_R))
    print("TCP R : %.1f, P : %.1f, Y : %.1f" % (tcp_r, tcp_p, tcp_y))
    # to paste as-is into ee_6d.py: "x y z roll pitch yaw" if x y z were given, otherwise only "roll pitch yaw" (paste it after x y z)
    print(" ".join(["%.3f" % c for c in xyz] + ["%.1f" % a for a in (tcp_r, tcp_p, tcp_y)]))


if __name__ == '__main__':
    main()
