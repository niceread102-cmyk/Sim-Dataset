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
| `assets/` | robot URDF and meshes (one folder per robot) and the objects (`assets/objects/stl/`: STL files, `assets/objects/usd/`: USD assets made in the Isaac Sim GUI) |
| `isaacsim/` | Isaac Sim scripts and configs |
| `bringup/` | MoveIt 2 config and launch, one folder per robot |
| `control/` | robot control nodes (target pose input, pick & place sequence) |

Scripts in `isaacsim/`:

- `robot_sim.py`: starts a robot in Isaac Sim, controlled over ROS 2.
- `generate_isaacsim.py`: generates the simulated vision data (bin-picking dataset).
- `generate_supervisor.py`: runs the generator shard by shard and restarts it when Isaac Sim hangs or
  crashes, so data collection keeps going even on a machine with little memory. Run the same command
  again to resume.
- `make_robot_config.py`: reads a robot URDF from `assets/robot/` and writes the robot yaml and the MoveIt
  bringup folder.
- `make_object_config.py`: reads an object STL from `assets/objects/stl/` and writes its part config yaml.

## Shortcuts used below

The commands use two shell aliases. Put them in `~/.bashrc`, or type the long form:

```bash
alias env="conda activate sim"                   # enter the conda environment
alias lab="~/IsaacLab/isaaclab.sh -p"            # run a script with Isaac Lab's python
```

The steps below are written with the long form, so they work without the aliases.

## Settings (`isaacsim/configs/`)

All the settings are yaml files here, so the robot, the objects and the scene are changed without touching the code.

| folder / file | what you can change |
|---|---|
| `robots/<name>.yaml` | one file per robot (made by `make_robot_config.py`): base position, start pose (`joint_pos`), joint drive `stiffness` / `damping`, `speed`, `physics_hz`, gripper, controller name |
| `objects/<name>.yaml` | one file per object (made by `make_object_config.py`): `mass_kg`, `friction`, `restitution`, collision shape (`collision`: convexDecomposition / convexHull / sdf, `max_convex_hulls`), `color`, `roughness`, `metallic`, `units`, symmetry, grasp points |
| `objects/bin.yaml` | the bin: mesh, size, colour, collision |
| `map/map.yaml` | the scene and the data generation: which objects (`parts:`), the bin, how the pile is made (`loading:`), taking parts out while shooting (`loading.pick`), lights (`randomization:`), output folders (`outputs:`), `run:` (number of scenes, views, output folder) |
| `cameras/*.yaml` | camera presets: intrinsics, mount height, depth sensor noise (`camera.noise`) |
| `sequence/*.yaml` | pick & place sequence: approach distance, delays, tolerances |

Terms: an **object** is the thing itself (the STL in `assets/objects/stl/`); its **part config** is the yaml in
`configs/objects/` that tells the simulator how it behaves (rigid body, collision, colour).

Any value can also be changed for one run on the command line with `key=value`, for example
`lab generate_isaacsim.py obj:=SIDE-PLATE camera.noise.enabled=false`.

Only the robot and the floor, without the bin and objects of `map.yaml`:
```bash
lab robot_sim.py robot:=<name> --robot-only
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
   are found relative to it:
   ```
   assets/robot/ur3e/urdf/ur3e.urdf
   assets/robot/ur3e/meshes/...
   ```
4. Generate the robot config (robot yaml in `isaacsim/configs/robots/`, MoveIt files in `bringup/ur3e/`):
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
~/IsaacLab/isaaclab.sh -p robot_sim.py robot:=ur3e        # = lab robot_sim.py robot:=ur3e
```
Wait until the Isaac Sim window is open and shows the robot. The first run of a new robot converts the
URDF to USD and caches it in `isaacsim/assets_cache/`, which takes a few minutes. It is not frozen,
do not close it. Later runs start much faster.

**Terminal 2: MoveIt and control nodes**
```bash
source /opt/ros/jazzy/setup.bash
cd bringup/integration/launch
ros2 launch robot.launch.py robot:=ur3e
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
start Terminal 2 with the name of a sequence of this robot (`isaacsim/configs/sequence/<name>.yaml`, one per robot, it names its
robot and has its poses and trajectory):
```bash
ros2 launch robot.launch.py robot:=ur3e sequence:=ur3e
```
The robot then moves to its home pose (its start pose, `joint_pos` of `isaacsim/configs/robots/<robot>.yaml`) when the launch has
started, and every target you type runs one sequence: it picks the part, lets go of it at `poses.place`, returns to the home pose and
waits for the next target. The sequence uses only the robot's own files, nothing of the scene.

<!-- add a screenshot or gif here -->

## Add an object and generate data

Example: `SIDE-PLATE.stl`

1. Put the STL file in `assets/objects/stl/`:
   ```
   assets/objects/stl/SIDE-PLATE.stl
   ```
2. Make the part config. It is written to `isaacsim/configs/objects/side_plate.yaml`
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
   - `obj:=` the part (name of its yaml in `configs/objects/`)
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
