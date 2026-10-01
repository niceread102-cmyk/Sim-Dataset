Sim-Dataset 실행 정리 (2026-10-01 기준)
==========================================

작업 폴더: ~/sim-dataset-repo/Sim-Dataset


0. 공통 환경
------------
- 가상환경: conda env `sim`  ->  env   (~/.bashrc alias, = conda activate sim)
- Isaac Lab 실행: lab                  (~/.bashrc alias, = ~/IsaacLab/isaaclab.sh -p)
- ROS 2 Jazzy: ~/.bashrc에서 자동 source, ROS_DOMAIN_ID=23
  (ROS 통신하는 터미널은 전부 같은 ROS_DOMAIN_ID여야 함)


1. 데이터셋 생성 (bin picking 합성 데이터)
-------------------------------------------
    cd ~/sim-dataset-repo/Sim-Dataset
    env
    lab scripts/run_supervised.py --config configs/default.yaml --images 100000

- 설정: configs/default.yaml (부품: configs/parts/side_plate_bp.yaml, 카메라, 빈, 출력 등)
- 출력: ~/IsaacSim/captures/bin_ds_side_plate_bp/part_01 ~ part_04/shard_XX
  (run.output_dir, run.num_parts = 4)
- --images: 파트 하나당 이미지 수. num_parts개 파트 각각 이만큼 만듦.
  생략하면 run.num_scenes x run.views_per_scene
- --scenes-per-process (기본 30): Isaac 프로세스 하나가 30 scene 만들면 새 프로세스로 교체,
  그때마다 shard_XX 폴더가 하나씩 늘어남. 0 = 끄기 (파트당 프로세스 하나)
- 멈춤 감시: 5분(--stall-min) 동안 scene이 안 나오면 스레드 스택을 저장하고 프로세스를 죽인 뒤
  다음 shard로 이어감. 같은 명령을 다시 실행하면 끝난 데까지 건너뛰고 이어서 함.
- 로그: <출력>/part_XX/logs/shard_XX.log, 멈춤 기록 hang_shard_XX_*.txt


2. 로봇 시뮬레이션 (Isaac Sim + ROS 2 + MoveIt)
-----------------------------------------------
터미널 1 - Isaac (로봇 + 그리퍼 + 물체):
    cd ~/sim-dataset-repo/Sim-Dataset
    env
    lab scripts/robot_sim.py                 # 설정: configs/robots/robot_sim.yaml

터미널 2 - MoveIt + RViz + IK 노드:
    source ~/fanuc_ws/install/setup.bash
    ros2 launch m20_moveit_config demo.launch.py

터미널 3 - TCP 목표 입력 (그리퍼 끝 위치):
    cd ~/lite6_ws/src/my_lite6_control/my_lite6_control
    python3 ee_6d.py
    입력: x y z roll pitch yaw   (m, deg, base_link 기준)   예: 1.2 0.0 0.5 180 0 0

그리퍼 열고 닫기 (0 = 닫힘, 1 = 열림, 사이값 = 일부 열림):
    ros2 topic pub --once /gripper/command std_msgs/msg/Float64 "{data: 1}"
    ros2 topic pub --once /gripper/command std_msgs/msg/Float64 "{data: 0}"


3. 현재 구성
------------
로봇:   FANUC M-20iD/35 (m20_35_18d)
        robot_assets/urdf/m20_35_18d.urdf + robot_assets/meshes/m20_35_18d/
        (FANUC 공식 fanuc_m20_description xacro를 푼 것과 링크/조인트 동일)
        베이스 = world 원점
그리퍼: JEGG2008, 일자 끝단, 핑거 300 mm 연장 (jegg2008_straight_300mm)
        gripper_assets/urdf/jegg2008_straight_300mm.urdf + meshes/, 설명은 gripper_assets/docs/
        로봇 flange에 고정 (어댑터 8 mm 포함)
        개폐: gripper_left_joint 0 ~ 0.004 m, 오른쪽은 왼쪽을 따라감 -> 핑거 내부 간격 0 ~ 8 mm
        TCP: 플랜지 면에서 469.316 mm (어댑터 8 + 모터 뒤~핑거 끝 461.316)
        이전 버전 jegg2008_300mm (휜 끝단)도 gripper_assets에 남아 있음
        질량/관성/파지력은 추정값 (gripper_assets/docs 참고)
물체:   side_plate_bp 1개, 중심 (1.2, 0, 0.02) m
        (configs/robots/robot_sim.yaml의 objects:, 부품 설정은 데이터셋과 같은 configs/parts/)


4. ROS 인터페이스
-----------------
robot_sim.py (Isaac):
  /m20_traj_controller/follow_joint_trajectory  control_msgs/FollowJointTrajectory  액션 (MoveIt 실행)
  /m20_traj_controller/joint_trajectory         trajectory_msgs/JointTrajectory      마지막 점으로 이동
  /gripper/command                              std_msgs/Float64                     0 닫힘 ~ 1 열림
  /joint_states                                 sensor_msgs/JointState               J1~J6 + 그리퍼 조인트
ik_node_m20.py (demo.launch.py가 같이 띄움):
  /target_pose (geometry_msgs/Pose, base_link 기준)를 받아 /compute_ik -> /move_action (계획 + 실행)
  - 위치 = 그리퍼 끝(TCP), 자세 = fanuc_flange 기준 (180 0 0 = 툴이 아래를 향함)
  - IK는 ready 자세(J5 = -90도)에서 출발해서 팔꿈치가 위로 가는 해를 고름
  - MoveIt 플래닝 씬에 바닥(z = 0) 추가


5. robot_sim.yaml 설정 항목
---------------------------
  name           로봇 이름 (robot_assets/urdf/<name>.urdf)
  gripper        그리퍼 이름 (gripper_assets/urdf/<gripper>.urdf), none = 그리퍼 없음
  gripper_topic  그리퍼 명령 토픽 (기본 /gripper/command)
  pos_m, rpy_deg 로봇 베이스 위치/자세
  joint_pos      시작 자세 (rad), 적지 않은 조인트는 0
  stiffness, damping  조인트 드라이브 (회전 N*m/deg, 직선 N/m)
  speed          최대 조인트 속도 = speed x URDF 속도 한계 (현재 0.3)
  controller     ROS 액션/토픽 이름 앞부분 (현재 /m20_traj_controller)
  objects        물체 목록, 한 줄에 하나: - {part: side_plate_bp, pos_m: [x, y, z], rpy_deg: [r, p, y]}

명령행 옵션: --name, --gripper none, --force-convert (메시만 바꿨을 때), --headless
첫 실행 / URDF 변경 시 USD로 변환해서 assets_cache/robot_<name>_<gripper>/ 에 저장 (그 뒤로는 재사용)


6. 관련 파일
------------
Sim-Dataset/scripts/robot_sim.py                       로봇 시뮬레이션 + ROS 브리지
Sim-Dataset/configs/robots/robot_sim.yaml              로봇 시뮬레이션 설정
Sim-Dataset/robot_assets/, gripper_assets/             URDF + 메시
~/fanuc_ws/src/m20_moveit_config/                      MoveIt 설정 (FANUC 공식 드라이버엔 M-20용 없음)
  launch/demo.launch.py, config/*.yaml, config/m20_35_18d.srdf, scripts/ik_node_m20.py
  수정 후: cd ~/fanuc_ws && colcon build --packages-select m20_moveit_config
~/lite6_ws/src/my_lite6_control/my_lite6_control/ee_6d.py   TCP 목표 입력


7. 주의사항
-----------
- RViz Plan & Execute와 ee_6d를 동시에 쓰지 말 것. MoveIt 실행이 겹치면 move_group이 망가지고
  segfault로 죽음. IK 노드는 다른 실행이 진행 중이면 목표를 무시하고 경고만 띄움.
  move_group이 죽으면 demo.launch.py를 다시 띄움.
- 시뮬레이션 로봇은 URDF 속도의 0.3배로 움직임. RViz Velocity Scaling을 0.3보다 크게 주면
  MoveIt이 TIMED_OUT으로 중단할 수 있음 (기본 0.1은 괜찮음).
- MoveIt 모델에는 그리퍼/물체 형상이 없음 (플래닝 씬에는 바닥만 있음). 위에서 내려가는 동작은
  괜찮지만, 기울여서 낮게 접근하면 그리퍼 몸체 충돌은 검사하지 않음.
- 그리퍼는 최대 8 mm까지만 벌어짐. side_plate_bp (55.6 x 55.6 x 20.9 mm)는 바깥에서 집을 수 없음.
- robot_sim.py는 isaacsim.exp.base.kit로 실행돼서 GUI에 물리 편집 메뉴(Add > Physics)가 없음.
  필요하면 (아직 실행 확인 안 함):
      lab scripts/robot_sim.py --kit_args "--enable omni.physx.bundle"
