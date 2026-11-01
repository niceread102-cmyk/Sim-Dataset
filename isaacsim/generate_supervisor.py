"""generate_supervisor.py: runs generate_isaacsim.py shard after shard until the target number of images exists, restarting it when Isaac Sim hangs or crashes.

    lab generate_supervisor.py [--config configs/map/map.yaml] [--generator generate_isaacsim.py] [--images 25000] [--stall-min 5] [key=value ...]
"""
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from binsim.config import expand_shorthand, load_config

GEN = os.path.join(REPO, "generate_isaacsim.py")
_child = {"proc": None, "stop": False}


def say(msg):
    print(f"[SUP {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shard_dirs(out):
    if not os.path.isdir(out):
        return {}
    return {int(d[6:]): os.path.join(out, d) for d in os.listdir(out) if d.startswith("shard_") and d[6:].isdigit()}


def scene_dirs(shard_dir):
    tp = os.path.join(shard_dir, "train_pbr")
    return [os.path.join(tp, d) for d in sorted(os.listdir(tp)) if d.isdigit()] if os.path.isdir(tp) else []


def complete(scene_dir):
    return os.path.exists(os.path.join(scene_dir, "scene_info.json"))


def scene_images(scene_dir):
    try:
        with open(os.path.join(scene_dir, "scene_info.json")) as f:
            n = json.load(f).get("num_views")
        if n is not None:
            return int(n)
    except (OSError, ValueError):
        pass
    rgb = os.path.join(scene_dir, "rgb")
    return len(os.listdir(rgb)) if os.path.isdir(rgb) else 0


def count_images(out):
    return sum(scene_images(s) for d in shard_dirs(out).values() for s in scene_dirs(d) if complete(s))


def drop_unfinished(shard_dir):
    for s in scene_dirs(shard_dir):
        if not complete(s):
            shutil.rmtree(s, ignore_errors=True)
            say(f"removed unfinished scene {s}")


def dump_stacks(pid, path):
    pyspy = shutil.which("py-spy") or os.path.expanduser("~/.local/bin/py-spy")
    with open(path, "w") as f:
        for cmd in ([pyspy, "dump", "--native", "--pid", str(pid)], [pyspy, "dump", "--pid", str(pid)]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                f.write(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\n")
                if r.returncode == 0:
                    break
            except Exception as e:
                f.write(f"$ {' '.join(cmd)}\nfailed: {e}\n")
    try:
        os.kill(pid, signal.SIGUSR1)
    except OSError:
        pass


def kill_group(proc):
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_shard(args, overrides, out, sid, num_scenes):
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    cmd = [sys.executable, args.generator, "--config", args.config, *overrides, f"run.shard_id={sid}", f"run.num_scenes={num_scenes}"]
    say(f"shard {sid}: {num_scenes} scenes | {' '.join(cmd)}")
    log = open(os.path.join(out, "logs", f"shard_{sid:02d}.log"), "a", buffering=1)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    _child["proc"] = proc
    state = {"last": time.monotonic(), "scenes": 0, "done": False}

    def pump():
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            if line.startswith("[INFO] scene ") and " saved " not in line:
                state["last"], state["scenes"] = time.monotonic(), state["scenes"] + 1
            elif line.startswith("[INFO] scene ") or line.startswith("[INFO] generating"):
                state["last"] = time.monotonic()
            elif line.startswith("[INFO] done:"):
                state["done"] = True

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    result = None
    while result is None:
        try:
            proc.wait(timeout=5)
            result = "done" if state["done"] else f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            limit = (args.stall_min if state["scenes"] else args.startup_min) * 60
            idle = time.monotonic() - state["last"]
            if idle > limit:
                say(f"shard {sid}: no progress for {idle / 60:.1f} min -> saving stacks and restarting")
                stack_file = os.path.join(out, "logs", f"hang_shard_{sid:02d}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
                dump_stacks(proc.pid, stack_file)
                say(f"stacks: {stack_file}")
                kill_group(proc)
                result = "hang"
    reader.join(timeout=10)
    log.close()
    _child["proc"] = None
    return result


def _on_signal(signum, frame):
    _child["stop"] = True
    if _child["proc"] is not None:
        say("stopping ...")
        kill_group(_child["proc"])
    sys.exit(130)


def main():
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "map", "map.yaml"))
    ap.add_argument("--generator", default=GEN, help="script that makes the scenes of one shard (gets --config and the overrides)")
    ap.add_argument("--images", type=int, default=None,
                    help="images per part (all its shards); default run.num_scenes x run.views_per_scene")
    ap.add_argument("--stall-min", type=float, default=5.0, help="restart when no scene finished for this long")
    ap.add_argument("--startup-min", type=float, default=10.0, help="allowance for launch + first scene")
    ap.add_argument("--max-failures", type=int, default=3, help="give up after this many shards in a row with no scene")
    ap.add_argument("--scenes-per-process", type=int, default=30,
                    help="each Isaac Sim process makes at most this many scenes, then a fresh one continues in the next "
                         "shard_XX (0 = no limit: one process per part, a new one only after a hang)")
    ap.add_argument("--dry-run", action="store_true", help="only print what would happen (touches nothing)")
    ap.add_argument("overrides", nargs="*", help="dotted config overrides passed to every shard; the short words obj:=<part> "
                                                 "obj_num:=N out:=DIR work as in generate_isaacsim.py")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.overrides = expand_shorthand(args.overrides, args.config)
    args.generator = os.path.abspath(os.path.expanduser(args.generator))
    if not args.dry_run and not os.path.isfile(args.generator):
        sys.exit(f"generator not found: {args.generator} (put generate_isaacsim.py there or pass --generator)")
    cfg = load_config(args.config, args.overrides)
    root = os.path.expanduser(cfg.run.output_dir)
    V = int(cfg.run.views_per_scene)
    target = int(args.images) if args.images else int(cfg.run.num_scenes) * V
    n_parts = max(int(cfg.run.get("num_parts", 1) or 1), 1)
    offset0 = int(cfg.run.get("scene_offset", 0))
    per_part = math.ceil(target / V)
    parts = [(root, list(args.overrides))] if n_parts == 1 else [
        (os.path.join(root, f"part_{p + 1:02d}"),
         [*args.overrides, f"run.output_dir={os.path.join(root, f'part_{p + 1:02d}')}", f"run.scene_offset={offset0 + p * per_part}"])
        for p in range(n_parts)]

    for p, (out, overrides) in enumerate(parts):
        if n_parts > 1:
            say(f"part {p + 1}/{n_parts}: {out} - {target} images, scene ids from {offset0 + p * per_part:06d}")
        if args.dry_run:
            for sid_, d in sorted(shard_dirs(out).items()):
                sc = scene_dirs(d)
                n_ok = sum(complete(s) for s in sc)
                imgs = sum(scene_images(s) for s in sc if complete(s))
                say(f"shard_{sid_:02d}: {n_ok} finished scenes ({imgs} images) - kept; {len(sc) - n_ok} unfinished - would be deleted")
            have = count_images(out)
            nxt = max(shard_dirs(out), default=-1) + 1
            say(f"{have} / {target} images -> " + (f"next: new folder shard_{nxt:02d} with {math.ceil((target - have) / V)} "
                                                   f"scenes (existing shards are never written)" if have < target else "done, skipped"))
            continue
        supervise(args, out, target, overrides, V, root, offset0 + p * per_part)
    if n_parts > 1 and not args.dry_run:
        say(f"all {n_parts} parts done")


def complete_ids(folder):
    shards = [*shard_dirs(folder).values()]
    for p in (os.path.join(folder, d) for d in os.listdir(folder) if d.startswith("part_")) if os.path.isdir(folder) else []:
        shards += shard_dirs(p).values()
    return {int(os.path.basename(s)) for d in shards for s in scene_dirs(d) if complete(s)}


def supervise(args, out, target, overrides, V, root, part_offset):
    failures = 0
    while True:
        for d in shard_dirs(out).values():
            drop_unfinished(d)
        have = count_images(out)
        say(f"{have} / {target} images in {out}")
        if have >= target:
            say("target reached")
            return
        sid = max(shard_dirs(out), default=-1) + 1
        need = math.ceil((target - have) / V)
        if args.scenes_per_process > 0:
            need = min(need, args.scenes_per_process)
        taken = complete_ids(root)
        base = part_offset
        while clash := [i for i in range(base, base + need) if i in taken]:
            base = max(clash) + 1
        before = have
        result = run_shard(args, [*overrides, f"run.scene_id_base={base}"], out, sid, need)
        d = shard_dirs(out).get(sid)
        if d:
            drop_unfinished(d)
        made = count_images(out) - before
        say(f"shard {sid} ended ({result}), {made} new images")
        failures = 0 if made > 0 else failures + 1
        if failures >= args.max_failures:
            say(f"{failures} shards in a row produced nothing - stopping (see {out}/logs)")
            sys.exit(1)


if __name__ == "__main__":
    main()
