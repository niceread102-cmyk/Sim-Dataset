"""Generate until a target number of images exists, surviving Isaac Sim hangs and crashes.

    lab scripts/run_supervised.py --config configs/default.yaml --images 50000 [--stall-min 5] [key=value ...]

Runs scripts/generate_isaacsim.py one shard at a time. When a shard stops making progress (no new
"[INFO] scene" line for --stall-min minutes) its thread stacks are saved to <output_dir>/logs/hang_shard_XX_*.txt
(py-spy, Python + native frames), the process is killed, the scenes it had not finished writing are deleted and
the next shard continues (new run.shard_id -> new piles; nothing is overwritten). Complete scenes of existing shards
count toward the target, so running the same command again resumes. Logs: <output_dir>/logs/shard_XX.log.
"""
import argparse
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from binsim.config import load_config  # noqa: E402

GEN = os.path.join(REPO, "scripts", "generate_isaacsim.py")
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
    return os.path.exists(os.path.join(scene_dir, "scene_info.json"))  # written last by process_scene


def count_images(out):
    return sum(len(os.listdir(os.path.join(s, "rgb"))) for d in shard_dirs(out).values() for s in scene_dirs(d)
               if complete(s))


def drop_unfinished(shard_dir):
    for s in scene_dirs(shard_dir):
        if not complete(s):
            shutil.rmtree(s, ignore_errors=True)
            say(f"removed unfinished scene {s}")


def dump_stacks(pid, path):
    """Python + native stacks of every thread of the hung process (the generator allows ptrace for this)."""
    pyspy = shutil.which("py-spy") or os.path.expanduser("~/.local/bin/py-spy")
    with open(path, "w") as f:
        for cmd in ([pyspy, "dump", "--native", "--pid", str(pid)], [pyspy, "dump", "--pid", str(pid)]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                f.write(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\n")
                if r.returncode == 0:
                    break
            except Exception as e:  # noqa: BLE001
                f.write(f"$ {' '.join(cmd)}\nfailed: {e}\n")
    try:
        os.kill(pid, signal.SIGUSR1)  # faulthandler: Python stacks into the shard log as well
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
    """Run one shard to completion or until it hangs. Returns 'done' | 'exit <rc>' | 'hang'."""
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    cmd = [sys.executable, GEN, "--config", args.config, *overrides, f"run.shard_id={sid}", f"run.num_scenes={num_scenes}"]
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
    """Ctrl+C / kill: stop the running shard too (it is in its own session) and do not start another one."""
    _child["stop"] = True
    if _child["proc"] is not None:
        say("stopping ...")
        kill_group(_child["proc"])
    sys.exit(130)


def main():
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    ap.add_argument("--images", type=int, required=True, help="stop once this many images exist (all shards)")
    ap.add_argument("--stall-min", type=float, default=5.0, help="restart when no scene finished for this long")
    ap.add_argument("--startup-min", type=float, default=10.0, help="allowance for launch + first scene")
    ap.add_argument("--max-failures", type=int, default=3, help="give up after this many shards in a row with no scene")
    ap.add_argument("--dry-run", action="store_true", help="only print what would happen (touches nothing)")
    ap.add_argument("overrides", nargs="*", help="dotted config overrides passed to every shard")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    cfg = load_config(args.config, args.overrides)
    out = os.path.expanduser(cfg.run.output_dir)
    V = int(cfg.run.views_per_scene)

    if args.dry_run:
        for sid_, d in sorted(shard_dirs(out).items()):
            sc = scene_dirs(d)
            n_ok = sum(complete(s) for s in sc)
            imgs = sum(len(os.listdir(os.path.join(s, "rgb"))) for s in sc if complete(s))
            say(f"shard_{sid_:02d}: {n_ok} finished scenes ({imgs} images) - kept; {len(sc) - n_ok} unfinished - would be deleted")
        have = count_images(out)
        nxt = max(shard_dirs(out), default=-1) + 1
        say(f"{have} / {args.images} images -> next: new folder shard_{nxt:02d} with "
            f"{max(math.ceil((args.images - have) / V), 0)} scenes (existing shards are never written)")
        return

    failures = 0
    while True:
        for d in shard_dirs(out).values():
            drop_unfinished(d)
        have = count_images(out)
        say(f"{have} / {args.images} images in {out}")
        if have >= args.images:
            say("target reached")
            return
        sid = max(shard_dirs(out), default=-1) + 1
        need = math.ceil((args.images - have) / V)
        before = have
        result = run_shard(args, args.overrides, out, sid, need)
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
