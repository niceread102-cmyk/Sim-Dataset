# Simulation

Bin-picking simulation in Isaac Sim with robot control through ROS 2 / MoveIt 2.

## Tested environment

| | version |
|---|---|
| Architecture | aarch64 (ARM64, NVIDIA DGX Spark / GB10) |
| OS | Ubuntu 24.04.4 LTS |
| Isaac Sim | v6.0.0-rc.59 |
| Isaac Lab | v3.0.0-beta |
| Python | 3.12.13 |
| Conda | 26.3.2 |
| ROS 2 | Jazzy (+ MoveIt 2: `sudo apt install ros-jazzy-moveit`) |

Install Isaac Sim and Isaac Lab first and create a conda environment for them (this README calls it `sim`):
- Isaac Sim: https://github.com/isaac-sim/IsaacSim/tree/v6.0.0
- Isaac Lab: https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-beta

We run Isaac Sim from `~/IsaacSim` (built from source). If yours is somewhere else, change `sim.experience` in `isaacsim/configs/map/map.yaml`.
If you hit a version or dependency error, resolve it for your own setup.

## Get the code

```bash
mkdir Simulation        # the folder name is up to you
cd Simulation
git clone -b Integrated-Sim https://github.com/f2x-project/Sim-Dataset .
```

All the commands below start in this `Simulation` folder.

## Folders

| folder | what it does |
|---|---|
| `assets/` | robot URDF and meshes (one folder per robot) and the objects (`assets/object/stl/`: STL files, `assets/object/usd/`: USD assets made in the Isaac Sim GUI) |
| `isaacsim/` | Isaac Sim scripts and configs |
| `bringup/` | MoveIt 2 config, one folder per robot, and the launch of every robot (`bringup/integration/launch/robot.launch.py`) |
| `control/` | robot control nodes (target pose input, pick & place sequence) |

Scripts in `isaacsim/`:

- `robot_sim.py`: starts a robot in Isaac Sim, controlled over ROS 2.
- `generate_isaacsim.py`: generates the simulated vision data (bin-picking dataset).
- `generate_supervisor.py`: runs the generator shard by shard and restarts it when Isaac Sim hangs or
  crashes, so data collection keeps going even on a machine with little memory. Run the same command
  again to resume.
- `make_robot_config.py`: reads a robot URDF from `assets/robot/` and writes the robot yaml and the MoveIt
  bringup folder.
- `make_object_config.py`: reads an object STL from `assets/object/stl/` and writes its part config yaml.
- `check_reach.py`: without Isaac Sim, whether the robot reaches the parts (or the crate floors) of the top layer of the pallets of a map.
- `sim_config.py`: finds and reads the maps, robots and machines of `configs/` (used by all the scripts, the launch and the control nodes).

## Shortcuts used below

The commands use two shell aliases. Put them in `~/.bashrc`, or type the long form:

```bash
alias env="conda activate sim"                   # enter the conda environment
alias lab="~/IsaacLab/isaaclab.sh -p"            # run a script with Isaac Lab's python
```

The steps below are written with the long form, so they work without the aliases.

## Settings (`isaacsim/configs/`)

All the settings are yaml files here, so the robot, the objects and the scene are changed without touching the code.
Robots, maps and machines are independent of each other: a robot yaml is the same in every map, a map names its robot only as a default.

| folder / file | what you can change |
|---|---|
| `robots/<name>.yaml` | one file per robot (made by `make_robot_config.py`), the same in every map: start pose (`joint_pos`), joint drive `stiffness` / `damping`, `speed` (simulation), `velocity_scaling` / `acceleration_scaling` (MoveIt), `joint_limits_deg` (narrower joint limits), `physics_hz`, gripper, controller name. `pos_m` / `rpy_deg` are 0: where the robot stands is the map's |
| `map/<map>.yaml` | one file per map (`map:=<map>`), a map of its own: its robot (`scene.robot`) and where it stands (`scene.robot_pose`), the environment USD of `assets/map/` (`scene.map`), pallets with crates and parts (`scene.pallets`, with the physics per layer), machines (`scene.machines`: name + where it stands), objects, bin, basket, platform. Without `map:=` the robot stands alone on the floor |
| `machines/<name>.yaml` | one file per machine (the laser marker, the conveyor), the same in every map: URDF, materials, collision, speeds, carriers (belt / chain), friction of the jigs (read again while the sim runs), the parameters of its cycle node (`isaacsim/machine/<name>.py`) |
| `objects/<name>.yaml` | one file per object (made by `make_object_config.py`): `mass_kg`, `friction`, `restitution`, collision shape (`collision`: convexDecomposition / convexHull / sdf, `max_convex_hulls`), `color`, `roughness`, `metallic`, `units`, symmetry, grasp points |
| `objects/bin.yaml` | the bin: mesh, size, colour, collision |
| `map/map.yaml` | the data generation (not a map): which objects (`parts:`), the bin, how the pile is made (`loading:`), taking parts out while shooting (`loading.pick`), lights (`randomization:`), output folders (`outputs:`), `run:` (number of scenes, views, output folder) |
| `cameras/*.yaml` | camera presets: intrinsics, mount height, depth sensor noise (`camera.noise`) |
| `sequence/*.yaml` | pick & place sequence: the task on a map (place poses, transfer, waits, trajectory), approach distance, tolerances |

Terms: an **object** is the thing itself (the STL in `assets/object/stl/`); its **part config** is the yaml in
`configs/object/` that tells the simulator how it behaves (rigid body, collision, colour).

Any value can also be changed for one run on the command line with `key=value`, for example
`lab generate_isaacsim.py obj:=SIDE-PLATE camera.noise.enabled=false`.

Only the robot and the floor (no `map:=`), or the robot in a map:
```bash
lab robot_sim.py robot:=<name>
lab robot_sim.py map:=<map> [robot:=<name>]        # the robot of the map when robot:= is not given
```

The next two sections are independent: a robot can run without any object, and the data can be generated without a robot.

## Add a robot and run it

Example: `ur3e`

### Prepare the robot (once per robot)

1. Prepare the robot URDF and meshes.
2. Make a folder with the robot name in `assets/robot/` (grippers are in `assets/grippper/<name>/` with the same urdf / meshes layout):
   ```bash
   cd assets/robot
   mkdir ur3e
   cd ../..
   ```
3. Put the URDF and meshes in it. The URDF file must be named after the folder, and the meshes
   are found relative to it (or as `package://<pkg>/...` in it or a folder above it, as xacro writes them):
   ```
   assets/robot/ur3e/urdf/ur3e.urdf
   assets/robot/ur3e/meshes/...
   ```
4. Generate the robot config (robot yaml in `isaacsim/configs/robot/`, MoveIt files in `bringup/ur3e/`):
   ```bash
   conda activate sim
   cd isaacsim
   python3 make_robot_config.py robot:=ur3e
   ```

### Run it (3 terminals)

You need three terminals. Keep all three open while the robot runs, and start them in this order.

**Terminal 1: Isaac Sim**
```bash
conda activate sim
cd isaacsim
~/IsaacLab/isaaclab.sh -p robot_sim.py robot:=ur3e        # = lab robot_sim.py robot:=ur3e   (add map:=<map> for a map)
```
Wait until the Isaac Sim window is open and shows the robot. The first run of a new robot converts the
URDF to USD and caches it in `isaacsim/assets_cache/`, which takes a few minutes. It is not frozen,
do not close it. Later runs start much faster.

**Terminal 2: MoveIt and control nodes**
```bash
source /opt/ros/jazzy/setup.bash
cd bringup/integration/launch
ros2 launch robot.launch.py robot:=ur3e        # the same map:=<map> as Isaac Sim, if any: MoveIt then has the robot where the map puts it
```
Start it after the robot appears in Isaac Sim.

**Terminal 3: send a target**
```bash
source /opt/ros/jazzy/setup.bash
cd control
python3 ee_6d.py
```
Type the TCP target in the robot base frame:
- `x y z`: position in m, orientation defaults to roll pitch yaw = 180 0 0 (tool pointing down)
- `x y z roll pitch yaw`: orientation in degrees
- `x y z qx qy qz qw`: orientation as a quaternion

### What you should see

The robot moves to the target (IK + MoveIt). If the target is outside the robot's reach, the motion fails.
Check the robot's reach first.

To run the whole pick & place sequence instead (approach, straight-line insert, gripper, home, place),
start Terminal 2 with a sequence (`isaacsim/configs/sequence/<name>.yaml`: the task on a map, what differs for one robot under `robots:`);
`sequence:=true` = the sequence of the map (`scene.sequence`). With a sequence the cycles of the machines of the map run too (`machines:=false` = not):
```bash
ros2 launch robot.launch.py map:=SIDE_PLATE_MAP robot:=ur3e sequence:=side_plate
```
The robot then moves to its home pose (its start pose, `joint_pos` of `isaacsim/configs/robot/<robot>.yaml`) when the launch has
started, and every target you type runs one sequence: it picks the part, lets go of it at `poses.place`, returns to the home pose and
waits for the next target. Of the map the sequence reads only its robot; the speed of the arm is the robot's (`velocity_scaling` /
`acceleration_scaling` of its yaml).

<!-- add a screenshot or gif here -->

## Add a map

A map is one yaml in `isaacsim/configs/map/` (e.g. `Orange.yaml`), a map of its own (no other map is read):
```yaml
scene:
  robot: r2000ic_125l                         # the robot of this map (robot:= on the command line wins)
  robot_pose: [0, 0, 0, 0, 0, 0]              # where the robot base stands [x, y, z, roll, pitch, yaw] (default 0)
  map: null                                   # an environment USD of assets/map/ (made in the Isaac Sim GUI), null = only the floor
  pallets: {...}                              # crates (a USD of assets/object/usd/ or a box of configs/object/) with or without parts
  machines:                                   # configs/machine/<name>.yaml + where it stands here
    - {name: laser_marker, pos_m: [x, y, z], rpy_deg: [0, 0, yaw]}
```
Start everything with the same `map:=<map>`: `lab robot_sim.py map:=<map>`, `ros2 launch robot.launch.py map:=<map>`,
`python3 control/ee_6d.py map:=<map>`, `python3 isaacsim/check_reach.py map:=<map>` (the reach of the robot to the top layer of the pallets).

## Add an object and generate data

Example: `SIDE-PLATE.stl`

1. Put the STL file in `assets/object/stl/`:
   ```
   assets/object/stl/SIDE-PLATE.stl
   ```
2. Make the part config. It is written to `isaacsim/configs/object/side_plate.yaml`
   (unit mm, mass from the mesh volume, steel-like metal; the size, mass and an estimate of how many parts fit in the bin are printed):
   ```bash
   conda activate sim
   cd isaacsim
   python3 make_object_config.py obj:=SIDE-PLATE
   ```
   Check the printed size and mass. Options: `--units`, `--mass`, `--density`, `--color R G B`.

3. Generate the data:
   ```bash
   lab generate_isaacsim.py obj:=SIDE-PLATE obj_num:=100 scenes:=2 views:=30 out:=~/Downloads/side_plate_data
   ```
   - `obj:=` the part (name of its yaml in `configs/object/`)
   - `obj_num:=` parts per bin (default: the estimate written in the part yaml)
   - `scenes:=` number of piles (one pile = one camera cycle)
   - `views:=` images per pile, so `scenes x views` images in all (here 2 x 30 = 60)
   - `out:=` output folder (default: `run.output_dir` in `configs/map/map.yaml`)

   `lab` is `~/IsaacLab/isaaclab.sh -p`. For long runs use `generate_supervisor.py` with the same words:
   it restarts Isaac Sim if it hangs and resumes when run again.

   Every few views the pile is shaken and the highest parts are taken out, so each pile goes from full to
   about 5-10 % of its parts over its views (`loading.pick.end_left_pct` in `map.yaml`).

### Output

```
<out>/shard_00/
    train_pbr/<scene>/    rgb/  depth/  depth_vis/  depth_relief/  mask_visib_inst/  poses/
                          scene_camera.json  scene_gt.json  scene_gt_info.json  scene_gt_coco.json
    viz/                  annotated bin images (the picture below)
    models/               part models (obj_000001.ply, models_info.json)
```
`depth/` is the sensor depth (16 bit, 0.1 mm per step). `depth_vis/` is a preview like the PhoXi viewer;
`depth_relief/` is a local-contrast preview that makes small height differences (the relief of a part) visible.

<img width="2928" height="1880" alt="000023_00_bin" src="https://github.com/user-attachments/assets/fd2e6a29-8e03-424f-b15f-1892936c80f3" />
