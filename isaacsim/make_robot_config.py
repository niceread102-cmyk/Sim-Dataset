#!/usr/bin/env python3
"""Config files of a new robot from its URDF: the simulation yaml and the whole MoveIt bringup folder.

    python3 make_robot_config.py robot:=hdr50_22 [--ee-link tool0] [--tcp 0 0 0.15] [--seed 0 30 -60 0 45 0]
                                 [--force] [--no-check] [--out DIR]

Input   assets/robot/<name>/urdf/<name>.urdf   (meshes relative to it, e.g. ../meshes/visual/base.stl, or package://<pkg>/... found in the URDF
        folder or a folder above it, as xacro writes them). <name> is the name of the URDF FILE: a
        robot with several URDFs in one folder (m20_35_18d.urdf, m20_35_18d_jegg2510n.urdf with the gripper) is one robot per file:
        robot:=m20_35_18d_jegg2510n makes its own yaml and bringup folder, the arm-only m20_35_18d stays as it is.
Output  isaacsim/configs/robot/<name>.yaml        name, start pose, stiffness, damping, speed, MoveIt speed and joint limits, controller
        bringup/<name>/config/                      <name>.srdf, kinematics, joint_limits, ompl_planning, moveit_controllers, ik_node
        bringup/<name>/rviz/demo.rviz
then       lab robot_sim.py robot:=<name> [map:=<map>]                 (Isaac Sim side, the joint drive force limit is the URDF effort)
           ros2 launch robot.launch.py robot:=<name> [map:=<map>]      (bringup/integration/launch: MoveIt + ik_node)

What comes from the URDF: the base link (below the empty `world` root), the arm joints from there to the end frame, their velocity
limits (acceleration = --accel-factor x velocity: MoveIt's time parametrization needs one), the robot name of the SRDF, the links
that touch by construction (collision check off: parent - child, fixed-joint bodies, siblings such as gripper fingers), the joints
that can turn more than a full turn (wrap_joints_deg of ik_node.yaml: ik_node picks their angle nearest to the current one).
Chosen here (override with the options):
  end frame   --ee-link: IK target, z axis = tool direction (rpy 180 0 0 = tool pointing down). Default: the tool0 / tcp / flange
              frame below the last joint.  --tcp: the TCP (gripper tip) in that frame. Default: where a link with `tcp` in its name
              (gripper_tcp of a URDF with the gripper) is in that frame, else 0.
  gripper     joints that are not part of the arm and not mimic joints (the fingers of a URDF with the gripper) are written as
              gripper_joints: to the sim yaml: /gripper/command drives them, robot_sequence.py waits for them.
  ready pose  --seed (deg): start pose of the sim = IK seed = home of the SRDF. Default: the best pose found by forward kinematics
              (tool pointing down, in front of the robot, elbow up, away from joint limits and singularities) that MoveIt accepts.
  sim drives  stiffness 20000 N*m/deg, damping 1000, speed 0.3 (the values of m20_35_18d): tune them in the sim.
  MoveIt      velocity_scaling 0.15, acceleration_scaling 0.05, no narrower joint limits (joint_limits_deg {}): change them in the robot yaml.
The MoveIt check (isolated ROS domain, needs ROS sourced; --no-check skips it) finds the link pairs that collide in every pose
(meshes that overlap at the joints), turns their collision check off, and takes the first ready pose that is collision free.
Files that already exist are kept (--force overwrites them) except joint_limits.yaml: it holds nothing but what the URDF says, so
every run writes it again from the URDF (velocity limits, acceleration, the position ranges in the comment on top). An existing
robot yaml is read (controller, joint_pos).
--out DIR writes everything below DIR instead of the project (no check).
Nothing else has to be registered: run_isaacsim.py, robot_sim.py, robot.launch.py and control/robot_sequence.py find the robot by its
files (configs/robot/<name>.yaml, bringup/<name>/). The robot stands at the origin; a map puts it somewhere else (scene.robot_pose).
"""
import argparse
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

import sim_config

HERE = Path(__file__).resolve().parent                    # Simulation/isaacsim
SIM_DIR = HERE.parent                                     # Simulation
BRINGUP_DIR = SIM_DIR / "bringup"
ROBOTS_DIR = HERE / "configs" / "robot"
LAUNCH = BRINGUP_DIR / "integration" / "launch" / "robot.launch.py"
GROUP = "arm"                                             # planning group of every robot
MOVABLE = ("revolute", "continuous", "prismatic")
REFRESH = ("lim",)                                        # files that are written again at every run (only the URDF in them)
EE_NAMES = ("tool0", "link_tcp", "tcp", "fanuc_flange", "tool", "flange", "ee_link")  # end frame candidates (z = tool direction), first match wins
CHECK_DOMAIN = "88"                                       # isolated ROS domain of the MoveIt check


# ---------------------------------------------------------------- URDF

def _vec(s):
    return np.array([float(x) for x in (s or "0 0 0").split()])


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_R(axis, q):
    """Rotations about `axis` by the angles q (N,) -> (N, 3, 3)."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _axis(j):
    return _vec(j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0")


class Urdf:
    def __init__(self, path: Path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.name = self.root.get("name")
        self.links = {ln.get("name"): ln for ln in self.root.findall("link")}
        self.joints = self.root.findall("joint")
        self.parent_joint = {j.find("child").get("link"): j for j in self.joints}

    def base_link(self) -> str:
        """The root link, or the link below it when the root is an empty frame (`world`) holding the robot by a fixed joint."""
        root = next(n for n in self.links if n not in self.parent_joint)
        kids = [j for j in self.joints if j.find("parent").get("link") == root]
        empty = all(self.links[root].find(t) is None for t in ("visual", "collision", "inertial"))
        if empty and len(kids) == 1 and kids[0].get("type") == "fixed":
            return kids[0].find("child").get("link")
        return root

    def chain(self, base: str, tip: str) -> list:
        """Joints from base to tip."""
        out, link = [], tip
        while link != base:
            j = self.parent_joint.get(link)
            if j is None:
                sys.exit(f"{tip} is not below {base} in {self.path.name}")
            out.append(j)
            link = j.find("parent").get("link")
        return out[::-1]

    def relative_pose(self, a: str, b: str):
        """4x4 pose of link b in the frame of link a (both held together by fixed joints below their common ancestor), None otherwise."""
        def up(n):
            out = [n]
            while n in self.parent_joint:
                n = self.parent_joint[n].find("parent").get("link")
                out.append(n)
            return out
        pa, pb = up(a), up(b)
        top = next((n for n in pa if n in pb), None)

        def frame(n):
            T = np.eye(4)
            while n != top:
                j = self.parent_joint[n]
                if j.get("type") != "fixed":
                    return None
                T = self.origin(j) @ T
                n = j.find("parent").get("link")
            return T
        Ta, Tb = frame(a), frame(b)
        return None if top is None or Ta is None or Tb is None else np.linalg.inv(Ta) @ Tb

    def tcp_offset(self, ee: str):
        """Position of the first link with `tcp` in its name (not the end frame itself) in the end frame, None if there is none."""
        for n in self.links:
            if "tcp" in n.lower() and n != ee:
                T = self.relative_pose(ee, n)
                if T is not None:
                    return [round(float(x), 6) + 0.0 for x in T[:3, 3]]   # + 0.0: no -0.0
        return None

    def default_ee_link(self, base: str) -> str:
        """The end frame: below the last revolute joint of the longest revolute chain from the base (the arm; gripper joints
        and the like are not counted), through fixed joints only. The first name of EE_NAMES that is there, else the last."""
        parents = {j.find("parent").get("link") for j in self.joints}
        arm = max((self.chain(base, n) for n in self.links if n not in parents and n != base),
                  key=lambda c: sum(j.get("type") in ("revolute", "continuous") for j in c))
        last = [j for j in arm if j.get("type") in ("revolute", "continuous")][-1]
        todo, below = [last.find("child").get("link")], []
        while todo:
            n = todo.pop()
            below.append(n)
            todo += [j.find("child").get("link") for j in self.joints if j.find("parent").get("link") == n and j.get("type") == "fixed"]
        return next((w for w in EE_NAMES if w in below), below[-1])

    def linked_pairs(self) -> set:
        """Pairs of links that touch by construction and are not worth a collision check: parent - child, links held
        together by fixed joints (they never move against each other) and links mounted on the same parent (the two
        fingers of a gripper, which overlap in the start pose)."""
        group = {n: n for n in self.links}

        def find(n):
            while group[n] != n:
                group[n] = group[group[n]]
                n = group[n]
            return n

        pairs, kids = set(), {n: [] for n in self.links}
        for j in self.joints:
            a, b = j.find("parent").get("link"), j.find("child").get("link")
            kids[a].append(b)
            pairs.add(tuple(sorted((a, b))))
            if j.get("type") == "fixed":
                group[find(a)] = find(b)
        bodies = {}
        for n in self.links:
            bodies.setdefault(find(n), []).append(n)
        for ls in list(bodies.values()) + list(kids.values()):   # rigid bodies, siblings (frames without geometry left out)
            ls = [n for n in ls if self.links[n].find("collision") is not None]
            pairs |= {tuple(sorted((x, y))) for i, x in enumerate(ls) for y in ls[i + 1:]}
        return pairs

    def origin(self, j) -> np.ndarray:
        o = j.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, :3] = _rpy_to_R(*_vec(o.get("rpy")))
            T[:3, 3] = _vec(o.get("xyz"))
        return T

    def fk(self, chain: list, q: np.ndarray):
        """Batch forward kinematics: q (N, movable joints of the chain) -> end frame (N, 4, 4) and the origin (N, 3) of every
        movable joint, all in the frame of the chain's base."""
        T = np.tile(np.eye(4), (len(q), 1, 1))
        k, origins = 0, []
        for j in chain:
            T = T @ self.origin(j)
            if j.get("type") in MOVABLE:
                origins.append(T[:, :3, 3].copy())
                R = np.tile(np.eye(4), (len(q), 1, 1))
                R[:, :3, :3] = _axis_R(_axis(j), q[:, k])
                T = T @ R
                k += 1
        return T, origins

    def gravity_torque(self, q: dict) -> dict:
        """|gravity torque| (N*m) about every movable joint for the pose q {joint: rad} (links with an <inertial> mass)."""
        T = {}

        def pose(n):
            if n not in T:
                j = self.parent_joint.get(n)
                if j is None:
                    T[n] = np.eye(4)
                else:
                    M = pose(j.find("parent").get("link")) @ self.origin(j)
                    if j.get("type") in MOVABLE:
                        R = np.eye(4)
                        R[:3, :3] = _axis_R(_axis(j), np.array([q.get(j.get("name"), 0.0)]))[0]
                        M = M @ R
                    T[n] = M
            return T[n]

        mass = {n: float(ln.find("inertial/mass").get("value")) for n, ln in self.links.items() if ln.find("inertial/mass") is not None}
        com = {n: _vec(self.links[n].find("inertial/origin").get("xyz")) if self.links[n].find("inertial/origin") is not None
               else np.zeros(3) for n in mass}

        def above(n):
            out = []
            while n in self.parent_joint:
                out.append(self.parent_joint[n].get("name"))
                n = self.parent_joint[n].find("parent").get("link")
            return out
        out = {}
        for j in (j for j in self.joints if j.get("type") in MOVABLE):
            frame = pose(j.find("parent").get("link")) @ self.origin(j)       # the joint before its own rotation
            axis, p = frame[:3, :3] @ _axis(j), frame[:3, 3]
            out[j.get("name")] = abs(float(sum(axis @ np.cross((pose(n) @ np.append(com[n], 1))[:3] - p, [0, 0, -m * 9.81])
                                               for n, m in mass.items() if j.get("name") in above(n))))
        return out


def find_urdf(name: str):
    """assets/robot/<name>/urdf/<name>.urdf, or <name>.urdf in another assets/robot/*/urdf/ folder (several URDFs of one robot share a folder)."""
    return Path(sim_config.find_urdf(name))


# ---------------------------------------------------------------- ready pose

def position_limits(joints: list):
    lo, hi = [], []
    for j in joints:
        lim = j.find("limit")
        if j.get("type") == "continuous" or lim is None or lim.get("lower") is None:
            lo.append(-math.pi), hi.append(math.pi)
        else:
            lo.append(float(lim.get("lower"))), hi.append(float(lim.get("upper")))
    return np.array(lo), np.array(hi)


def position_range(j) -> str:
    """Position range of a movable URDF joint as the URDF says it, to read: deg (m for a prismatic joint), `continuous` without
    limits. Two decimals (lite6 joint3 starts at -3.5 deg), ±x for a range around zero."""
    lim = j.find("limit")
    if j.get("type") == "continuous" or lim is None or lim.get("lower") is None:
        return "continuous"
    lo, hi = float(lim.get("lower")), float(lim.get("upper"))
    unit = " m" if j.get("type") == "prismatic" else ""
    if not unit:
        lo, hi = math.degrees(lo), math.degrees(hi)
    lo, hi = round(lo, 2) + 0.0, round(hi, 2) + 0.0       # + 0.0: no "-0"
    return (f"±{hi:g}" if lo == -hi else f"{lo:g} ~ {hi:g}") + unit


def wrap_joints(joints: list) -> dict:
    """Joints whose position range is wider than a full turn (360 deg, or no limits at all): name -> (min, max) in deg, for the
    wrap_joints_deg of ik_node.yaml. q and q +- 360 are one pose there, so the IK may return any of them and the arm would spin
    once round. A narrower joint has one angle per pose and is left out, a prismatic joint never turns."""
    out = {}
    for j in joints:
        if j.get("type") == "prismatic":
            continue
        lim = j.find("limit")
        if j.get("type") == "continuous" or lim is None or lim.get("lower") is None:
            out[j.get("name")] = (-math.inf, math.inf)
        elif float(lim.get("upper")) - float(lim.get("lower")) > 2 * math.pi + 1e-6:
            out[j.get("name")] = (math.degrees(float(lim.get("lower"))), math.degrees(float(lim.get("upper"))))
    return out


def wrap_joints_text(wrap: dict) -> str:
    """wrap_joints_deg block of ik_node.yaml (with its comment)"""
    num = lambda v: ".inf" if v == math.inf else "-.inf" if v == -math.inf else f"{v + 0.0:.6f}".rstrip("0").rstrip(".")
    rows = "".join(f"\n  {j}: [{num(lo)}, {num(hi)}]" for j, (lo, hi) in wrap.items()) if wrap else " {}"
    return ("# Joints that can turn more than a full turn (position range wider than 360 deg in the URDF) and that range (deg). The IK can\n"
            "# return the angle of such a joint a full turn off (q +- 360) and the arm would spin once round to reach it: ik_node sends the\n"
            "# equivalent angle nearest to the current one (inside this range and joint_limits_deg). Joints not listed are sent as solved.\n"
            f"wrap_joints_deg:{rows}\n")


def ready_candidates(urdf: Urdf, chain: list, count: int = 12):
    """Candidate ready poses (rad), best first, and the reach (m). Tool pointing down (z axis of the end frame), in front of
    the robot, elbow up (joint 3 above joint 2), away from the joint limits and from singularities. The first and the last
    joint stay at 0, the others are searched on a 15 degree grid (roll joints of more than 360 degrees within +-90)."""
    movable = [j for j in chain if j.get("type") in MOVABLE]
    n = len(movable)
    lo, hi = position_limits(movable)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    step = math.radians(15)
    axes = []
    for i in range(n):
        if i in (0, n - 1):
            axes.append(np.array([min(max(0.0, lo[i]), hi[i])]))
            continue
        a, b = lo[i] + 0.08 * span[i], hi[i] - 0.08 * span[i]
        if hi[i] - lo[i] > 2 * math.pi + 1e-6:
            a, b = max(a, -math.pi / 2), min(b, math.pi / 2)
        axes.append(np.arange(math.ceil(a / step), math.floor(b / step) + 1) * step)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, n)
    T, org = urdf.fk(chain, grid)
    p, tool_z = T[:, :3, 3], T[:, :3, 2]
    reach = float(np.linalg.norm(p[:, [0, 2]], axis=1).max())
    idx = []
    for tilt, h_min, d_min in ((10, 0.10, 0.25), (20, 0.05, 0.15), (35, 0.0, 0.0)):   # relax if nothing fits
        ok = (tool_z[:, 2] <= -math.cos(math.radians(tilt))) & (p[:, 2] >= h_min * reach) & (p[:, 0] >= d_min * reach)
        if n >= 3:
            ok &= org[2][:, 2] >= org[1][:, 2]
        idx = np.flatnonzero(ok)
        if len(idx):
            break
    if not len(idx):
        return [], reach
    q = grid[idx]
    margin = np.minimum((q - lo) / span, (hi - q) / span).min(axis=1)              # 0 at a limit .. 0.5 in the middle
    place = np.abs(p[idx, 0] - 0.5 * reach) / reach + np.abs(p[idx, 2] - 0.45 * reach) / reach
    cheap = 2.0 * margin - place - 0.3 * (np.abs(q[:, 0]) + np.abs(q[:, -1])) / math.pi
    order = np.argsort(-cheap)[:300]
    sel = idx[order]
    # conditioning of the Jacobian (position in m, rotation x 0.5 reach) of the best few hundred: far from singularities
    eps, J = 1e-4, np.zeros((len(sel), 6, n))
    T0, _ = urdf.fk(chain, grid[sel])
    for k in range(n):
        d = grid[sel].copy()
        d[:, k] += eps
        Tk, _ = urdf.fk(chain, d)
        dR = Tk[:, :3, :3] @ np.transpose(T0[:, :3, :3], (0, 2, 1))
        J[:, :3, k] = (Tk[:, :3, 3] - T0[:, :3, 3]) / eps
        J[:, 3, k] = (dR[:, 2, 1] - dR[:, 1, 2]) / (2 * eps)
        J[:, 4, k] = (dR[:, 0, 2] - dR[:, 2, 0]) / (2 * eps)
        J[:, 5, k] = (dR[:, 1, 0] - dR[:, 0, 1]) / (2 * eps)
    J[:, 3:, :] *= 0.5 * reach
    s = np.linalg.svd(J, compute_uv=False)
    total = cheap[order] + 1.5 * s[:, -1] / np.maximum(s[:, 0], 1e-9)
    out = []
    for k in np.argsort(-total):
        if all(np.abs(grid[sel[k]] - o).max() > math.radians(30) for o in out):    # candidates that really differ
            out.append(grid[sel[k]])
        if len(out) >= count:
            break
    return out, reach


# ---------------------------------------------------------------- file texts

def srdf_text(urdf: Urdf, base: str, ee: str, arm: list, home: dict, always: list) -> str:
    pairs = {p: "Adjacent" for p in urdf.linked_pairs()}
    for p in always:
        pairs[tuple(sorted(p))] = "Always"
    out = ['<?xml version="1.0" ?>',
           f"<!-- {urdf.path.name}: generated by isaacsim/make_robot_config.py. Group {GROUP} = the chain {base} -> {ee}. The pairs below are not "
           "collision checked: links that touch by construction (Adjacent) and meshes that overlap in every pose (Always). -->",
           f'<robot name="{urdf.name}">',
           f'  <group name="{GROUP}">', f'    <chain base_link="{base}" tip_link="{ee}" />', "  </group>",
           f'  <group_state name="home" group="{GROUP}">']
    out += [f'    <joint name="{j}" value="{home[j]:.4f}" />' for j in arm]
    out += ["  </group_state>", f'  <virtual_joint name="virtual_joint" type="fixed" parent_frame="world" child_link="{base}" />']
    out += [f'  <disable_collisions link1="{a}" link2="{b}" reason="{why}" />' for (a, b), why in sorted(pairs.items())]
    return "\n".join(out + ["</robot>", ""])


def joint_limits_text(name, rows, factor, ranges, efforts) -> str:
    out = [f"# {name}: written from the URDF by isaacsim/make_robot_config.py at every run: change the URDF (or --accel-factor), not this file.",
           "# Velocities are the URDF velocity limits; MoveIt's time-optimal trajectory generation needs an acceleration:",
           f"# {factor:g} x the velocity. With has_acceleration_limits: false it throws an \"undefined\" error and nothing can be executed.",
           "# Position and effort limits are the URDF's (MoveIt reads position from there; the sim drive force limit is the effort), listed",
           "# here to read: position deg (m), velocity rad/s (m/s), effort N*m (N):"]
    out += [f"#   {j}: position {r}, velocity {v:g}, effort {e:g}" for (j, v), r, e in zip(rows, ranges, efforts)]
    out += ["joint_limits:"]
    for j, vel in rows:
        out += [f"  {j}:", "    has_velocity_limits: true", f"    max_velocity: {vel!r}",
                "    has_acceleration_limits: true", f"    max_acceleration: {factor * vel!r}"]
    return "\n".join(out) + "\n"


def kinematics_text(name) -> str:
    return (f"# {name}: generated by isaacsim/make_robot_config.py\n{GROUP}:\n  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin\n"
            "  kinematics_solver_search_resolution: 0.005\n  kinematics_solver_timeout: 0.05\n")


def ompl_text(name, arm) -> str:
    return f"""# {name}: generated by isaacsim/make_robot_config.py, the structure of bringup/m20_35_18d (MoveIt2 Jazzy: planning_plugins
# (plural) + request_adapters / response_adapters).
planning_plugins:
  - ompl_interface/OMPLPlanner
request_adapters:
  - default_planning_request_adapters/ResolveConstraintFrames
  - default_planning_request_adapters/ValidateWorkspaceBounds
  - default_planning_request_adapters/CheckStartStateBounds
  - default_planning_request_adapters/CheckStartStateCollision
response_adapters:
  - default_planning_response_adapters/AddTimeOptimalParameterization
  - default_planning_response_adapters/ValidateSolution
  - default_planning_response_adapters/DisplayMotionPath

planner_configs:
  RRTConnect:
    type: geometric::RRTConnect
    range: 0.0

{GROUP}:
  default_planner_config: RRTConnect
  planner_configs:
    - RRTConnect
  projection_evaluator: joints({arm[0]},{arm[1]})
  longest_valid_segment_fraction: 0.01
"""


def controllers_text(name, controller, arm) -> str:
    joints = "\n".join(f"      - {j}" for j in arm)
    return f"""# {name}: generated by isaacsim/make_robot_config.py. The FollowJointTrajectory action server is provided by isaacsim/robot_sim.py
# (controller: of configs/robot/{name}.yaml). The action name is "<controller name>/<action_ns>", here
# /{controller}/follow_joint_trajectory.
moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager

# MoveIt only moves the robot: its built-in execution time limit (TIMED_OUT) is switched off.
trajectory_execution:
  execution_duration_monitoring: false

moveit_simple_controller_manager:
  controller_names:
    - {controller}

  {controller}:
    action_ns: follow_joint_trajectory
    type: FollowJointTrajectory
    default: true
    joints:
{joints}
"""


def ik_node_text(name, base, ee, arm, tcp, seed_deg, wrap) -> str:
    fmt = lambda v: "[" + ", ".join(f"{x:g}" for x in v) + "]"
    return f"""# Parameters of control/ik_node.py for {name} (generated by isaacsim/make_robot_config.py).
base_frame: {base}   # frame of /target_pose and the IK: the Isaac Sim world (robot.launch.py puts where the robot stands in the map, scene.robot_pose, into MoveIt's URDF)
group_name: {GROUP}                   # config/{name}.srdf
ee_link: {ee}                    # z axis = tool direction, so rpy (180, 0, 0) = tool pointing down
arm_joints: [{", ".join(arm)}]
tcp_offset: {fmt(tcp)}            # m, TCP (gripper tip) in the ee_link frame

# IK seed (deg), in arm_joints order: the ready pose (= start pose of the sim).
# ik_node.py solves from the current joint angles: this pose is only its start before /joint_states arrives (and the start of the
# first IK of control/robot_sequence.py).
ik_seed_deg: {fmt(seed_deg)}

# Speed (velocity_scaling, acceleration_scaling) and narrower joint limits (joint_limits_deg) are the robot's: isaacsim/configs/robot/{name}.yaml

{wrap_joints_text(wrap)}"""


def sim_yaml_text(name, urdf_rel, controller, home, stiffness, damping, speed, gripper) -> str:
    start = ", ".join(f"{j}: {v:.4f}" for j, v in home.items() if abs(v) > 1e-9)
    grip = (f"gripper_joints: [{', '.join(gripper)}]        # joints of the URDF that /gripper/command drives (0 = lower limit, 1 = upper limit; mimic joints follow)\n"
            if gripper else "")
    return f"""# lab robot_sim.py robot:={name}    (URDF: ../../../{urdf_rel}, MoveIt: ros2 launch robot.launch.py robot:={name})
# Generated by isaacsim/make_robot_config.py. Only the robot is here, the same in every map: what stands around it is the map (configs/map/<map>.yaml, map:=<map>).
name: {name}
gripper: none
{grip}pos_m: [0.0, 0.0, 0.0]            # mount of the robot base (m) = 0 for every robot: where it stands in a map is scene.robot_pose of the map
rpy_deg: [0, 0, 0]
joint_pos: {{{start}}}
                                  # start pose (rad) = the ready pose = ik_seed_deg of bringup/{name}/config/ik_node.yaml
stiffness: {stiffness:g}                  # joint drives, N*m/deg (prismatic: N/m). Higher = less sag. The force limit is the URDF effort
damping: {damping:g}                     # higher = less swinging, slower settling
speed: {speed:g}                        # max joint speed = speed x URDF velocity limit
velocity_scaling: 0.15            # MoveIt (ik_node.py, robot_sequence.py): share of the joint velocity limits the motions use (1.0 = full)
acceleration_scaling: 0.05        # MoveIt: share of the acceleration limits (bringup/<robot>/config/joint_limits.yaml)
joint_limits_deg: {{}}              # narrower position range per joint than the URDF (deg), e.g. {{J1: [-90, 90]}}: MoveIt plans and solves the IK inside it (the start pose has to be inside too)
physics_hz: 60                    # physics steps per simulated second
controller: /{controller}  # <controller>/joint_trajectory, <controller>/follow_joint_trajectory (bringup/{name}/config/moveit_controllers.yaml)
"""


# ---------------------------------------------------------------- MoveIt check

def _stop(proc):
    """Stop ros2 launch and everything it started (own process group): INT, then TERM, then KILL."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=15)
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        os.killpg(proc.pid, signal.SIGKILL)       # whatever is left of the group
    except ProcessLookupError:
        pass


def moveit_check(name, arm, lo, hi, candidates, log):
    """Start move_group (ros2 launch robot.launch.py in the isolated ROS domain) and ask /check_state_validity about random
    poses and the candidate ready poses. Returns (link pairs that collide in (nearly) every pose, index of the first
    collision free candidate or None, share of the random poses in collision)."""
    import rclpy
    from moveit_msgs.msg import RobotState
    from moveit_msgs.srv import GetStateValidity
    from sensor_msgs.msg import JointState
    proc = subprocess.Popen(["ros2", "launch", str(LAUNCH), f"robot:={name}", "rviz:=false", "ik_node:=false"],
                            stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rclpy.init(args=[])   # robot:=<name> of the command line is not a ROS remap rule
        node = rclpy.create_node("make_robot_config")
        cli = node.create_client(GetStateValidity, "/check_state_validity")
        t0 = time.monotonic()
        while not cli.service_is_ready() and time.monotonic() - t0 < 90:
            if proc.poll() is not None:
                raise RuntimeError(f"ros2 launch ended with code {proc.returncode}, see {log}")
            rclpy.spin_once(node, timeout_sec=0.3)
        if not cli.service_is_ready():
            raise RuntimeError(f"move_group did not come up within 90 s, see {log}")
        time.sleep(3)

        def check(q):
            req = GetStateValidity.Request()
            req.group_name = GROUP
            req.robot_state = RobotState()
            req.robot_state.joint_state = JointState(name=arm, position=[float(x) for x in q])
            f = cli.call_async(req)
            rclpy.spin_until_future_complete(node, f, timeout_sec=10)
            if f.result() is None:
                raise RuntimeError("no answer of /check_state_validity")
            return f.result()

        rng, counts, bad, N = np.random.default_rng(0), {}, 0, 300
        for _ in range(N):
            r = check(rng.uniform(lo, hi))
            bad += not r.valid
            for c in r.contacts:
                k = tuple(sorted((c.contact_body_1, c.contact_body_2)))
                counts[k] = counts.get(k, 0) + 1
        always = [k for k, c in counts.items() if c >= 0.9 * N]
        first = next((i for i, q in enumerate(candidates) if check(q).valid), None)
        return always, first, bad / N
    finally:
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        _stop(proc)


# ---------------------------------------------------------------- main

def _robot(a: str) -> str:
    return a[len("robot:="):] if a.startswith("robot:=") else a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot", metavar="robot:=<name>", type=_robot, help="robot name = assets/robot/<name>/urdf/<name>.urdf")
    ap.add_argument("--ee-link", help="IK target frame (default: the tool0 / tcp / flange frame below the last joint)")
    ap.add_argument("--tcp", nargs=3, type=float, metavar=("X", "Y", "Z"), help="TCP in the ee frame, m (default: from a tcp link of the URDF, else 0)")
    ap.add_argument("--seed", nargs="+", type=float, help="ready pose in deg, one value per arm joint (default: searched)")
    ap.add_argument("--controller", help="controller name (default: <name>_traj_controller)")
    ap.add_argument("--stiffness", type=float, default=20000.0, help="sim drive stiffness, N*m/deg")
    ap.add_argument("--damping", type=float, default=1000.0, help="sim drive damping")
    ap.add_argument("--speed", type=float, default=0.3, help="sim max joint speed = speed x URDF velocity limit")
    ap.add_argument("--accel-factor", type=float, default=2.0, help="MoveIt max acceleration = this x max velocity")
    ap.add_argument("--force", action="store_true", help="overwrite files that exist")
    ap.add_argument("--no-check", action="store_true", help="no MoveIt check (collision pairs, collision free ready pose)")
    ap.add_argument("--out", type=Path, help="write below this folder instead of the project (no check)")
    args = ap.parse_args()
    name = args.robot

    urdf_path = find_urdf(name)
    if not urdf_path.is_file():
        sys.exit(f"URDF not found: assets/robot/{name}/urdf/{name}.urdf (or {name}.urdf in another assets/robot/*/urdf/)")
    tree = ET.parse(urdf_path)                  # mesh paths of xacro (package://...) -> paths relative to the urdf folder, written back into the URDF
    fixed = 0
    for m in tree.getroot().iter("mesh"):
        found = sim_config.find_mesh(m.get("filename"), str(urdf_path.parent))
        if found and "://" in m.get("filename", ""):
            m.set("filename", os.path.relpath(found, urdf_path.parent))
            fixed += 1
    if fixed:
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
        print(f"[INFO] {fixed} mesh path(s) of {urdf_path.name} set relative to the urdf folder (e.g. ../meshes/visual/...)", flush=True)
    miss = [m.get("filename") for m in tree.getroot().iter("mesh") if not sim_config.find_mesh(m.get("filename"), str(urdf_path.parent))]
    if miss:
        sys.exit(f"{len(miss)} mesh(es) of {urdf_path.name} not found (relative to the urdf folder, or package:// in it or a folder above it), e.g. {miss[0]}")
    urdf = Urdf(urdf_path)
    base = urdf.base_link()
    ee = args.ee_link or urdf.default_ee_link(base)
    root = next(n for n in urdf.links if n not in urdf.parent_joint)
    frame = root if root != base else base      # the empty `world` above the robot, else the base link: the frame of /target_pose
    chain = urdf.chain(base, ee)
    movable = [j for j in chain if j.get("type") in MOVABLE]
    arm = [j.get("name") for j in movable]
    if len(arm) < 2:
        sys.exit(f"the chain {base} -> {ee} has {len(arm)} movable joint(s)")
    lo, hi = position_limits(movable)
    gripper = [j.get("name") for j in urdf.joints if j.get("type") in MOVABLE and j.get("name") not in arm and j.find("mimic") is None]
    if args.tcp is None:
        auto = urdf.tcp_offset(ee)
        args.tcp, tcp_how = (auto, "from the tcp link of the URDF") if auto else ([0.0, 0.0, 0.0], "none")
    else:
        tcp_how = "--tcp"
    vel = [(j.get("name"), float(j.find("limit").get("velocity", 0)) if j.find("limit") is not None else 0.0) for j in movable]
    vel = [(n, v if v > 0 else math.pi) for n, v in vel]
    efforts = [float(j.find("limit").get("effort", 0)) if j.find("limit") is not None else 0.0 for j in movable]

    root = args.out or SIM_DIR
    yaml_file = (root / "configs" / "robot" if args.out else ROBOTS_DIR) / f"{name}.yaml"
    cfg_dir = root / "bringup" / name / "config"
    files = {"sim": yaml_file, "srdf": cfg_dir / f"{name}.srdf", "kin": cfg_dir / "kinematics.yaml", "lim": cfg_dir / "joint_limits.yaml",
             "ompl": cfg_dir / "ompl_planning.yaml", "ctrl": cfg_dir / "moveit_controllers.yaml", "ik": cfg_dir / "ik_node.yaml",
             "rviz": root / "bringup" / name / "rviz" / "demo.rviz"}
    existing = (yaml.safe_load(open(yaml_file)) or {}) if yaml_file.is_file() else {}
    controller = (args.controller or str(existing.get("controller") or f"/{name}_traj_controller")).strip("/")

    # ready pose candidates: --seed, joint_pos of an existing robot yaml, or the search
    if args.seed:
        if len(args.seed) != len(arm):
            sys.exit(f"--seed needs {len(arm)} values, one per arm joint {arm}")
        candidates, how = [np.radians(args.seed)], "--seed"
    elif existing.get("joint_pos"):
        jp = existing["joint_pos"]
        candidates, how = [np.array([next((float(v) for k, v in jp.items() if re.fullmatch(k, j)), 0.0) for j in arm])], \
            f"joint_pos of {yaml_file.name}"
    else:
        candidates, reach = ready_candidates(urdf, chain)
        how = f"searched with forward kinematics, reach {reach:.2f} m"
        if not candidates:
            print("[WARN] no pose with the tool pointing down found: the ready pose is all zeros, give --seed", flush=True)
            candidates = [np.zeros(len(arm))]

    always, made, created = [], {}, set()

    def texts(q):
        home = {j: float(v) for j, v in zip(arm, q)}
        return {"sim": sim_yaml_text(name, urdf_path.relative_to(SIM_DIR), controller, home, args.stiffness, args.damping, args.speed, gripper),
                "srdf": srdf_text(urdf, base, ee, arm, home, always), "kin": kinematics_text(name),
                "lim": joint_limits_text(name, vel, args.accel_factor, [position_range(j) for j in movable], efforts), "ompl": ompl_text(name, arm),
                "ctrl": controllers_text(name, controller, arm),
                "ik": ik_node_text(name, frame, ee, arm, args.tcp, [math.degrees(v) for v in q], wrap_joints(movable))}

    def put(key, text):
        """Write a file unless it exists (a file written in this run may be written again)."""
        f = files[key]
        if f.exists() and not args.force and key not in created and key not in REFRESH:
            made[key] = False
            return
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        created.add(key)
        made[key] = True

    for k, t in texts(candidates[0]).items():
        put(k, t)
    src = next((p for p in sorted(BRINGUP_DIR.glob("*/rviz/demo.rviz")) if p.parent.parent.name != name), None)
    if src is None:
        print("[WARN] no bringup/*/rviz/demo.rviz to copy as the RViz layout", flush=True)
    else:   # the layout of another robot with this robot's fixed frame and planning group
        s = re.sub(r"(?m)^(\s*Fixed Frame:\s*).*$", rf"\g<1>{base}", src.read_text())
        put("rviz", re.sub(r"(?m)^(\s*Planning Group:\s*).*$", rf"\g<1>{GROUP}", s))

    # MoveIt check: link pairs that collide in every pose, first collision free ready pose
    ready_i, note = 0, "not checked"
    if not args.no_check and not args.out:
        if shutil.which("ros2") is None:
            note = "skipped: ROS 2 is not sourced (source /opt/ros/jazzy/setup.bash and run again with --force)"
        else:
            os.environ["ROS_DOMAIN_ID"], os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = CHECK_DOMAIN, "LOCALHOST"  # apart from other ROS terminals
            log = Path(tempfile.gettempdir()) / f"make_robot_config_{name}.log"
            try:
                print("[INFO] MoveIt check: starting move_group in the isolated ROS domain (about a minute) ...", flush=True)
                found, first, share = moveit_check(name, arm, lo, hi, candidates, log)
                if found and "srdf" in created:
                    always = found
                    put("srdf", texts(candidates[0])["srdf"])
                    print(f"[INFO] collision check off for pairs that collide in every pose: {found}; checking again ...", flush=True)
                    found, first, share = moveit_check(name, arm, lo, hi, candidates, log)
                ready_i = first if first is not None else 0
                note = (f"{share * 100:.0f} % of random poses collide (self collision, joint limits are the URDF's); "
                        + ("ready pose collision free" if first is not None else "NO collision free ready pose among the candidates, give --seed")
                        + (f"; still colliding in every pose: {found}" if found else ""))
                if ready_i:
                    t = texts(candidates[ready_i])
                    for k in ("sim", "srdf", "ik"):
                        if k in created:
                            put(k, t[k])
                log.unlink(missing_ok=True)
            except Exception as e:  # noqa: BLE001
                note = f"check failed: {e}"

    # summary
    q = candidates[ready_i]
    T, _ = urdf.fk(chain, q[None])
    tilt = math.degrees(math.acos(float(np.clip(-T[0, 2, 2], -1, 1))))
    tau = urdf.gravity_torque({j: float(v) for j, v in zip(arm, q)})
    print(f"\nrobot {name} (URDF robot {urdf.name}): chain {base} -> {ee}, controller /{controller}/follow_joint_trajectory")
    print(f"end frame {ee}, TCP {[round(x, 4) for x in args.tcp]} m ({tcp_how})" + (f", gripper joints {gripper}" if gripper else ", no gripper joints"))
    print(f"ready pose ({how}): " + ", ".join(f"{j} {math.degrees(v):.0f}" for j, v in zip(arm, q)) + " deg")
    print(f"  tool at ({T[0, 0, 3]:.2f}, {T[0, 1, 3]:.2f}, {T[0, 2, 3]:.2f}) m, {tilt:.0f} deg off straight down | check: {note}")
    print(f"  {'joint':14s} {'position [deg]':>16s} {'vel [rad/s]':>12s} {'effort [N*m]':>13s} {'gravity at ready':>17s}")
    for j in movable:
        n, lim = j.get("name"), j.find("limit")
        pos = position_range(j)
        eff = float(lim.get("effort", 0)) if lim is not None else 0.0
        warn = "   <- over 70 % of the effort: the sim arm may sag" if eff > 0 and tau[n] > 0.7 * eff else ""
        print(f"  {n:14s} {pos:>16s} {dict(vel)[n]:12.3f} {eff:13.1f} {tau[n]:12.1f} N*m{warn}")
    print("files:")
    for k, f in files.items():
        print(f"  {'written' if made.get(k) else 'kept   '}  {f if args.out else os.path.relpath(f, SIM_DIR)}")
    if not args.out:
        print(f"next:  lab robot_sim.py robot:={name}     |     cd bringup/integration/launch && ros2 launch robot.launch.py robot:={name}")


if __name__ == "__main__":
    main()
