# robot-control — FANUC M-20iD/35 Pick & Place

Control a FANUC M-20iD/35 using Isaac Sim, ROS 2 Jazzy, and MoveIt 2.

By entering the TCP position and orientation of the gripper tip as `x y z roll pitch yaw` through `ee_6d.py`, the robot automatically executes the following sequence:

Approach → Insert → Grasp → Home → Place in Basket → Home

- `isaacsim/` — Simulator and Isaac Sim ↔ ROS 2 bridge
- `bringup/` — MoveIt 2 launch files
- `control/` — Robot sequence node and TCP input tool

## Run

Use three terminals.

The repository is assumed to be located at `~/project/Simulation`.

Each terminal requires:

```bash
source /opt/ros/jazzy/setup.bash
```

**1. Isaac Sim**

```bash
cd ~/project/Simulation/isaacsim
conda activate sim      # alias: env
lab robot_sim.py        # alias: ~/IsaacLab/isaaclab.sh -p robot_sim.py
```

**2. MoveIt 2 + Sequence Node**  
Run this in a terminal with Conda deactivated.

```bash
cd ~/project/Simulation
ros2 launch bringup/m20_35_18d/launch/m20_35_18d.launch.py sequence:=true
```

**3. Control the TCP with `ee_6d.py`**

```bash
cd ~/project/Simulation/control
python3 ee_6d.py

# x y z [roll pitch yaw]
# Example: 1.0 0.0 0.15 180 0 0
# Units: m, deg
# 180 0 0 = tool pointing downward
```

## Configuration

- Robot, gripper, bin, and basket:
  `isaacsim/configs/robots/robot_sim.yaml`

- Sequence settings such as home pose, place pose, and wait times:
  `control/robot_sequence.yaml`

- If the basket position is changed, update both:
  - `basket.pos_m` in `robot_sim.yaml`
  - `poses.place` x/y values in `robot_sequence.yaml`

## Technologies
Isaac Sim 6.0 / Isaac Lab (PhysX, USD) · ROS 2 Jazzy · MoveIt 2 (OMPL, KDL IK) · URDF/SRDF · Python (rclpy, SciPy)
- **Isaac Sim 6.0 / Isaac Lab** — NVIDIA robotics simulation platform. Uses PhysX to simulate the robot, gripper, bin, parts, and basket, and converts URDF/STL assets to USD to build the simulation scene.
- **ROS 2 Jazzy** — Connects the simulator, MoveIt, and control nodes through ROS topics, services, and actions.
- **MoveIt 2** — Motion planning framework. OMPL plans collision-free paths, KDL solves inverse kinematics (IK), and the resulting trajectories are sent to the simulator through `FollowJointTrajectory`.
- **`robot_sim.py`** — Isaac Sim ↔ ROS 2 bridge. Publishes `/joint_states` and receives MoveIt trajectories and gripper commands (`/gripper/command`) to control the simulated robot.
- **`robot_sequence.py`** — Receives `/target_pose` and generates the sequence: approach → insert → grasp → place in basket. Each step is solved using seeded IK, with the previous solution used as the next seed to maintain a consistent arm configuration and avoid unnecessary joint flipping. Arm stop and gripper completion are determined from `/joint_states`.
- **`ee_6d.py`** — Accepts EE/TCP coordinates as `x y z roll pitch yaw` and publishes them to `/target_pose`. SciPy is used to convert Euler angles to quaternions.
- **URDF / SRDF** — Robot model definitions. URDF describes the links, joints, and joint limits of the M-20iD/35 with the JEGG2510N gripper, while SRDF defines the MoveIt planning groups.
