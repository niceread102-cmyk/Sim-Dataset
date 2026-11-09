"""path_planner.py: the shape of the arm paths in TCP poses (pan around the base, height first, tucked turns, approach and insert, carry, place, crate moves, other ways when a path is not possible); ik_node.py turns them into joints.

Knows no map and no robot joints: the coordinates come from the sequence (robot_sequence.py), the settings from control/config/path_planner.yaml.
"""
import copy
import math
from pathlib import Path

import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from scipy.spatial.transform import Rotation

from ik_node import joints, line, move, offset_pose, xyzrpy_to_pose

SETTINGS_FILE = Path(__file__).resolve().parent / 'config' / 'path_planner.yaml'
TOOL_AXIS = [0.0, 0.0, 1.0]
MOTION_KINDS = ('line', 'move', 'joints', 'ik')


def load_settings(path=SETTINGS_FILE):
    return yaml.safe_load(open(path)) or {}


def tool(label, on, settle=0.0):
    return {'label': label, 'kind': 'tool', 'value': bool(on), 'search': False, 'settle': settle}


def via_steps(label, start, end, max_step, max_angle):
    a, b = start.position, end.position
    qa, qb = start.orientation, end.orientation
    ra, rb = Rotation.from_quat([qa.x, qa.y, qa.z, qa.w]), Rotation.from_quat([qb.x, qb.y, qb.z, qb.w])
    turn = ra.inv() * rb
    n = max(1, math.ceil(max(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) / max_step, math.degrees(turn.magnitude()) / max_angle)))
    steps = []
    for i in range(1, n):
        q = (ra * Rotation.from_rotvec(turn.as_rotvec() * i / n)).as_quat()
        pose = Pose(position=Point(x=a.x + (b.x - a.x) * i / n, y=a.y + (b.y - a.y) * i / n, z=a.z + (b.z - a.z) * i / n),
                    orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))
        steps.append({'label': f'{label} {i}/{n}', 'kind': 'ik', 'value': pose, 'search': True})
    return steps


def heading(rot):
    m = rot.as_matrix()
    return math.atan2(m[1, 0], m[0, 0])


def in_band(angle, ref):
    return ref + (angle - ref + math.pi) % (2 * math.pi) - math.pi


def wrist_angle(pose, base_xy):
    q = pose.orientation
    return math.atan2(pose.position.y - base_xy[1], pose.position.x - base_xy[0]) - heading(Rotation.from_quat([q.x, q.y, q.z, q.w]))


def raised(pose, dz):
    p = pose.position
    return Pose(position=Point(x=p.x, y=p.y, z=p.z + dz), orientation=pose.orientation)


def joint_line(label, start, end, pieces):
    a, b = start.position, end.position
    steps = [dict(move(label, Pose(position=Point(x=a.x + (b.x - a.x) * i / pieces, y=a.y + (b.y - a.y) * i / pieces, z=a.z + (b.z - a.z) * i / pieces),
                                   orientation=end.orientation)), near=True) for i in range(1, pieces + 1)]
    return {'label': label, 'kind': 'path', 'steps': steps, 'settle': 0.0, 'search': False}


def tilted(pose, deg, phi_deg):
    a, q = math.radians(phi_deg), pose.orientation
    o = (Rotation.from_rotvec([math.cos(a) * math.radians(deg), math.sin(a) * math.radians(deg), 0.0]) * Rotation.from_quat([q.x, q.y, q.z, q.w])).as_quat()
    return Pose(position=pose.position, orientation=Quaternion(x=float(o[0]), y=float(o[1]), z=float(o[2]), w=float(o[3])))


def pan_steps(label, start, end, base_xy, max_step, gap=None, rise_to=None, rise_angle=0.0, ref=None, wrist=False, tuck=None):
    bx, by = base_xy
    a, b = start.position, end.position
    ta, tb = math.atan2(a.y - by, a.x - bx), math.atan2(b.y - by, b.x - bx)
    dt = (tb - ta + math.pi) % (2 * math.pi) - math.pi
    if gap is not None:
        way = 1.0 if dt >= 0 else -1.0
        if ((gap - ta) * way) % (2 * math.pi) < abs(dt):
            dt -= way * 2 * math.pi
    ra, rb = math.hypot(a.x - bx, a.y - by), math.hypot(b.x - bx, b.y - by)
    qa, qb = start.orientation, end.orientation
    rot_a, rot_b = Rotation.from_quat([qa.x, qa.y, qa.z, qa.w]), Rotation.from_quat([qb.x, qb.y, qb.z, qb.w])
    if ref is None:
        dz = 0.0
    elif wrist:
        dz = dt - (in_band(wrist_angle(end, base_xy), ref) - in_band(wrist_angle(start, base_xy), ref))
    else:
        dz = in_band(heading(rot_b), ref) - in_band(heading(rot_a), ref)
    rest = (rot_a.inv() * Rotation.from_euler('z', -dz) * rot_b).as_rotvec()
    angle = abs(dz) + math.sqrt(sum(v * v for v in rest))
    if abs(dt) < 1e-3:
        rise_to = None
    top = a.z if rise_to is None else rise_to
    rise_angle = min(max(rise_angle, 1e-6), abs(dt) * 0.8)
    length = math.hypot(max(ra, rb) * abs(dt), rb - ra, abs(top - a.z) + abs(b.z - top))
    n = max(1, math.ceil(length / max_step), math.ceil(angle / (math.pi / 2)))
    steps, prev = [], start
    for i in range(1, n + 1):
        f = i / n
        t, r = ta + dt * f, ra + (rb - ra) * f
        if tuck is not None and abs(dt) > math.pi / 2 and r > tuck:
            r = tuck + (r - tuck) * (2 * f - 1) ** 2
        q = (Rotation.from_euler('z', dz * f) * rot_a * Rotation.from_rotvec(rest * f)).as_quat()
        turned = abs(dt) * f
        if rise_to is None:
            z = a.z + (b.z - a.z) * f
        elif turned < rise_angle:
            z = a.z + (rise_to - a.z) * turned / rise_angle
        else:
            z = rise_to + (b.z - rise_to) * (turned - rise_angle) / (abs(dt) - rise_angle)
        pose = Pose(position=Point(x=bx + r * math.cos(t), y=by + r * math.sin(t), z=z),
                    orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))
        steps.append(line(f'{label} {i}/{n}', pose, prev))
        prev = pose
    return steps


def is_pan(cfg, mode):
    return str((cfg.get('trajectory') or {}).get(mode, '')).lower() == 'pan'


def carry_height(cfg):
    cz = cfg['poses'].get('carry_z')
    if cz is None:
        return None
    return float(cfg['poses']['home'][2]) if str(cz).strip().lower() == 'home' else float(cz)


def lift_pose(cfg, approach, carry):
    lz, cz = (cfg.get('trajectory') or {}).get('lift_z'), carry_height(cfg)
    if lz is None or carry is None or cz is None:
        return None
    dz = carry.position.z - (cz - float(lz)) - approach.position.z
    return raised(approach, dz) if dz > 1e-3 else None


def spiral_start(cfg, approach, home):
    traj = cfg.get('trajectory') or {}
    if traj.get('approach_slope_deg') is None:
        return None
    bx, by = cfg['robot']['base_xy']
    p = approach.position
    r = math.hypot(p.x - bx, p.y - by)
    drop = home.position.z - p.z
    if drop <= 1e-3 or r < 1e-3:
        return None
    arc = min(drop / math.tan(math.radians(float(traj['approach_slope_deg']))) / r, math.radians(float(traj.get('approach_arc_max_deg', 90.0))))
    side = (math.atan2(home.position.y - by, home.position.x - bx) - math.atan2(p.y - by, p.x - bx) + math.pi) % (2 * math.pi) - math.pi
    if arc >= abs(side):
        return None
    turn = Rotation.from_euler('z', arc if side >= 0 else -arc)
    v = turn.apply([p.x - bx, p.y - by, 0.0])
    q = (turn * Rotation.from_quat([approach.orientation.x, approach.orientation.y, approach.orientation.z, approach.orientation.w])).as_quat()
    return Pose(position=Point(x=bx + float(v[0]), y=by + float(v[1]), z=home.position.z),
                orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))


def pan(cfg, label, pose, start, rise_to=None, tuck=None):
    robot, traj = cfg['robot'], cfg.get('trajectory') or {}
    turn = str(traj.get('pan_turn', 'heading')).lower()
    if turn not in ('heading', 'wrist'):
        raise ValueError(f'pan_turn must be heading | wrist, got {turn!r}')
    home = xyzrpy_to_pose(*cfg['poses']['home'])
    q = home.orientation
    ref = wrist_angle(home, robot['base_xy']) if turn == 'wrist' else heading(Rotation.from_quat([q.x, q.y, q.z, q.w]))
    return pan_steps(label, start, pose, robot['base_xy'], float(traj.get('pan_step', 0.10)), robot['base_yaw'] + math.pi,
                     rise_to, math.radians(float(traj.get('climb_deg', 60.0))), ref, turn == 'wrist', tuck)


def pan_down(cfg, label, pose, start, high_z):
    forward = pan(cfg, label, start, pose, rise_to=high_z)
    points = [pose] + [st['value'] for st in forward]
    points.reverse()
    n = len(forward)
    return [line(f'{label} {i}/{n}', points[i], points[i - 1]) for i in range(1, n + 1)]


def approach_pan(cfg, approach, home, carry):
    lift = lift_pose(cfg, approach, carry)
    if lift is None:
        start = spiral_start(cfg, approach, home)
        if start is not None:
            return [*pan(cfg, 'approach side', start, home), *pan(cfg, 'approach', approach, start)]
        return pan_down(cfg, 'approach', approach, home, home.position.z)
    return [*pan_down(cfg, 'approach', lift, home, home.position.z), line('approach', approach, lift)]


def joint_steps(steps):
    return [dict(move(st['label'], st['value']), near=True) if st['kind'] == 'line' else st for st in steps]


def crate_pan(cfg, label, new, old):
    traj = cfg.get('trajectory') or {}
    tuck = traj.get('crate_tuck_radius_m')
    tuck = float(tuck) if tuck is not None else None
    shape = joint_steps if traj.get('crate_joint') else (lambda steps: steps)
    if traj.get('crate_z_first') and abs(new.position.z - old.position.z) > 1e-3:
        if new.position.z > old.position.z:
            level = raised(old, new.position.z - old.position.z)
            return [move(f'{label} height', level), *shape(pan(cfg, label, new, level, tuck=tuck))]
        above = raised(new, old.position.z - new.position.z)
        return [*shape(pan(cfg, label, above, old, tuck=tuck)), move(f'{label} height', new)]
    return shape(pan(cfg, label, new, old, tuck=tuck))


def fuse_steps(steps):
    out, group = [], []

    def flush():
        nonlocal group
        real = [st for st in group if st['kind'] != 'ik']
        if len(real) > 1:
            out.append({'label': '+'.join(st['label'] for st in real), 'kind': 'path', 'steps': group,
                        'settle': group[-1].get('settle', 0.0), 'search': False})
        else:
            out.extend(group)
        group = []
    for st in steps:
        if st['kind'] in MOTION_KINDS and not st.get('slow') and not st.get('single'):
            group.append(st)
        else:
            flush()
            out.append(st)
    flush()
    return out


def travel(cfg, name):
    return 'line' if str((cfg.get('trajectory') or {}).get(name, 'joint')).lower() == 'line' else 'joint'


def go(cfg, label, pose, start, mode):
    if mode == 'home' and cfg.get('home_joints'):
        return joints(label, cfg['home_joints'])
    return line(label, pose, start) if travel(cfg, mode) == 'line' else move(label, pose)


def approach_step(cfg, approach, home):
    return go(cfg, 'approach', approach, home, 'approach') if travel(cfg, 'approach') == 'line' else move('approach', approach)


def via_settings(cfg):
    traj = cfg.get('trajectory') or {}
    return float(traj.get('via_step_m', 0.1)), float(traj.get('via_angle_deg', 15.0))


def place_pose(cfg, variant=None):
    place = cfg['poses']['place']
    if not isinstance(place, dict):
        return place
    return place.get(variant) or place.get('standard') or next(iter(place.values()))


def stays_over_standard(cfg, variant):
    place, t = cfg['poses']['place'], cfg.get('transfer') or {}
    return bool(t.get('stay_over_standard')) and isinstance(place, dict) and (variant != 'turn_over' or bool(t.get('stay_over_turn_over')))


def standard_above(cfg):
    poses = cfg['poses']
    chosen = place_pose(cfg, 'standard')
    if poses.get('place_height') is not None:
        z = poses['home'][2] if poses['place_height'] == 'home' else poses['place_height']
    else:
        z = chosen[2] + float(poses.get('place_offset') or 0.0)
    return xyzrpy_to_pose(chosen[0], chosen[1], z, *chosen[3:])


def home_high(cfg):
    poses = cfg['poses']
    home = xyzrpy_to_pose(*poses['home'])
    cz = carry_height(cfg)
    return home, (raised(home, cz - float(poses['home'][2])) if cz is not None and cz > float(poses['home'][2]) + 1e-3 else None)


def high_way_home(cfg, here, here_z, home, home_hi, label):
    up = raised(here, carry_height(cfg) - here_z)
    across = pan(cfg, 'above home', home_hi, up) if is_pan(cfg, 'home') else [line('above home', home_hi, up)]
    return [*([line('up to the carry height', up, here)] if carry_height(cfg) > here_z + 1e-3 else []),
            *across, line(label, home, home_hi)]


def make_sequence(target, cfg, variant=None):
    gripper, poses = cfg['gripper'], cfg['poses']
    release_settle = float(gripper.get('release_settle', 0.0))
    traj = cfg.get('trajectory') or {}
    chosen   = place_pose(cfg, variant)
    place_tcp = poses['home'] if chosen == 'home' else chosen
    release_tcp = poses.get('release')
    if chosen != 'home' and poses.get('place_height') is not None:
        z = poses['home'][2] if poses['place_height'] == 'home' else poses['place_height']
        place_tcp, release_tcp = [*chosen[:2], z, *chosen[3:]], chosen
    elif chosen != 'home' and poses.get('place_offset') is not None:
        place_tcp, release_tcp = [chosen[0], chosen[1], chosen[2] + poses['place_offset'], *chosen[3:]], chosen
    grasp    = target
    approach = offset_pose(grasp, -cfg['approach']['distance'], TOOL_AXIS)
    carry    = None
    cz = carry_height(cfg)
    if cz is not None and cz > target.position.z + cfg['approach']['distance'] + 1e-3:
        carry = Pose(position=Point(x=target.position.x, y=target.position.y, z=cz), orientation=target.orientation)
    home, home_hi = home_high(cfg)
    via      = [xyzrpy_to_pose(*v) for v in (traj.get('via') or [])]
    place    = xyzrpy_to_pose(*place_tcp)
    release  = xyzrpy_to_pose(*release_tcp) if release_tcp else None
    at_home  = chosen == 'home'
    stay     = stays_over_standard(cfg, variant) and not at_home
    to_std   = stay and variant == 'turn_over'
    via_step, via_angle = via_settings(cfg)
    pieces = max(1, int(traj.get('joint_line_pieces', 5)))

    path = [] if at_home else [(f'via {i + 1}', v) for i, v in enumerate(via)] + [('place', place)]
    spiral = carry is not None and not at_home and is_pan(cfg, 'place')
    lift = lift_pose(cfg, approach, carry) if spiral else None
    lift_joint = False
    if spiral and lift is None and traj.get('carry_z_first') and carry.position.z > approach.position.z + 1e-3:
        lift, lift_joint = carry, True
    out, here = [], (lift or approach if spiral else carry or home)
    for label, pose in path:
        if label == 'place' and is_pan(cfg, 'place'):
            out.extend(pan(cfg, label, pose, here, rise_to=carry.position.z if spiral else None))
        else:
            out.append(go(cfg, label, pose, here, 'place' if label == 'place' else 'via'))
        here = pose
    back, there = [], here
    for label, pose in reversed(path[:-1]):
        back.append(go(cfg, 'return ' + label, pose, there, 'via'))
        there = pose
    return [
        tool('close gripper (start)', False),
        *(approach_pan(cfg, approach, home, carry) if is_pan(cfg, 'approach') else
          [*via_steps('approach', home, approach, via_step, via_angle), approach_step(cfg, approach, home)]),
        joint_line('insert', approach, grasp, pieces) if traj.get('insert_mode') == 'joint' else line('insert', grasp, approach),
        tool('open gripper', True),
        joint_line('rise', grasp, approach, pieces) if traj.get('rise_mode') == 'joint' else line('rise', approach, grasp),
        *([move('lift', lift) if lift_joint else line('lift', lift, approach)] if lift else []),
        *([] if spiral else [line('carry', carry, approach)] if carry else [go(cfg, 'home', home, approach, 'home')]),
        *out,
        *([line('lower',       release,  here),
           tool('close gripper', False, settle=release_settle),
           line('lift',        here,     release)]
          if release else [tool('close gripper', False, settle=release_settle)]),
        *([line('wait above standard', standard_above(cfg), here)] if to_std else []),
        *([] if stay else back),
        *([] if (at_home or stay) else (high_way_home(cfg, there, float(place_tcp[2]), home, home_hi, 'return home') if home_hi is not None
                                        else pan(cfg, 'return home', home, there) if is_pan(cfg, 'home') and not cfg.get('home_joints')
                                        else [go(cfg, 'return home', home, there, 'home')])),
    ]


def make_transfer(cfg, place_tcp=None, above=None):
    head, tail = transfer_steps(cfg, place_tcp, above)
    return head + [{'label': 'measure yaw', 'kind': 'align', 'value': None, 'search': False}] + tail


def transfer_steps(cfg, place_tcp=None, above=None):
    gripper, poses, t = cfg['gripper'], cfg['poses'], cfg['transfer']
    release_settle = float(gripper.get('release_settle', 0.0))
    step, angle = via_settings(cfg)
    pick     = xyzrpy_to_pose(*t['pick'])
    approach = offset_pose(pick, -cfg['approach']['distance'], TOOL_AXIS)
    x, y, z, *rest = place_tcp or t['place']
    up       = xyzrpy_to_pose(x, y, z + float(poses.get('place_offset') or cfg['approach']['distance']), *rest)
    down     = xyzrpy_to_pose(x, y, z, *rest)
    home, home_hi = home_high(cfg)

    insert_a = dict(line('insert', pick, approach), follow_part=True)
    insert_b = dict(line('insert', pick, above), follow_part=True) if above is not None else None
    start    = home_hi or home
    head = [
        tool('close gripper (start)', False),
        *([line('above home', home_hi, home)] if home_hi is not None else []),
        *via_steps('approach', start, approach, step, angle),
        approach_step(cfg, approach, start),
        insert_a,
        tool('open gripper', True),
        dict(line('rise', approach, pick), follow_part=True),
    ] if above is None else [
        insert_b,
        tool('open gripper', True),
        dict(line('rise', approach, pick), follow_part=True),
    ]
    tail = [
        *via_steps('place', approach, up, step, angle),
        go(cfg, 'place', up, approach, 'place'),
        line('lower', down, up),
        tool('close gripper', False, settle=release_settle),
        line('lift', up, down),
        *(high_way_home(cfg, up, z + float(poses.get('place_offset') or cfg['approach']['distance']), home, home_hi, 'home') if home_hi is not None
          else [go(cfg, 'home', home, up, 'home')]),
    ]
    return head, tail


def crate_steps(cfg, label, old, old_hi, new, new_hi, final):
    if is_pan(cfg, 'crate'):
        steps = crate_pan(cfg, label, new, old) + final
        return fuse_steps(steps) if (cfg.get('trajectory') or {}).get('fuse_moves') else steps
    if carry_height(cfg) is None:
        return [move(f'{label} scan pose', new)] + final
    up, over = old_hi or old, new_hi or new
    return (([line('up from the crate', up, old)] if old_hi is not None else []) + [line(f'over {label}', over, up)]
            + ([line(f'{label} scan pose', new, over)] if new_hi is not None else []) + final)


def alternatives(cfg):
    traj = cfg.get('trajectory') or {}
    out = [dict(o) for o in traj.get('reach_fallbacks') or []]
    limit = float(traj.get('pick_tilt_max_deg') or 0.0)
    first = float(traj.get('pick_tilt_first_deg', 5.0))
    dirs = traj.get('pick_tilt_dirs_deg') or [0, 90, 180, 270]
    out += [{'tilt_deg': t, 'tilt_dir_deg': float(phi)} for t in sorted({min(first, limit), limit}) if t > 0 for phi in dirs]
    return out


def with_alternative(cfg, target, override):
    override = dict(override)
    tilt, phi = float(override.pop('tilt_deg', 0.0)), float(override.pop('tilt_dir_deg', 0.0))
    alt = copy.copy(cfg)
    alt['trajectory'] = {**(cfg.get('trajectory') or {}), **override}
    text = f'the tool tilted {tilt:g} deg towards {phi:g} deg' if tilt else str(override)
    return (tilted(target, tilt, phi) if tilt else target), alt, text
