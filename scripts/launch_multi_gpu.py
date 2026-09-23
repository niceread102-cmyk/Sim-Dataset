"""Run several generate.py shards in parallel (one or more per GPU) and optionally merge them.

    python scripts/launch_multi_gpu.py --config configs/default.yaml --total-scenes 20000 \
        [--gpus 0,1] [--procs-per-gpu 1] [--python "./isaaclab.sh -p"] [--merge] [key=value ...]

Each shard gets a unique run.shard_id (-> unique scene ids and seeds) and writes to
<output_dir>/shard_XX. Logs go to <output_dir>/logs/shard_XX.log. `--merge` creates
<output_dir>/merged/ (a single BOP dataset of symlinks) when all shards finish.
Within one process, parallelism comes from run.num_envs (cloned bins rendered together).
"""
import argparse
import math
import os
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from binsim.config import load_config  # noqa: E402


def detect_gpus():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return [str(i) for i, line in enumerate(out.strip().splitlines()) if line.startswith("GPU")]
    except Exception:
        return ["0"]


def merge(out_dir, shard_dirs):
    import shutil
    m = os.path.join(out_dir, "merged")
    os.makedirs(os.path.join(m, "train_pbr"), exist_ok=True)
    first = shard_dirs[0]
    for f in ("camera.json", "config_resolved.yaml"):
        if os.path.exists(os.path.join(first, f)):
            shutil.copyfile(os.path.join(first, f), os.path.join(m, f))
    if not os.path.exists(os.path.join(m, "models")):
        shutil.copytree(os.path.join(first, "models"), os.path.join(m, "models"))
    n = 0
    for sd in shard_dirs:
        tp = os.path.join(sd, "train_pbr")
        if not os.path.isdir(tp):
            continue
        for scene in sorted(os.listdir(tp)):
            dst = os.path.join(m, "train_pbr", scene)
            if not os.path.exists(dst):
                os.symlink(os.path.relpath(os.path.join(tp, scene), os.path.dirname(dst)), dst)
                n += 1
    print(f"[merge] {n} scenes linked into {m}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    ap.add_argument("--total-scenes", type=int, required=True)
    ap.add_argument("--gpus", default=None, help="comma-separated GPU indices (default: all)")
    ap.add_argument("--procs-per-gpu", type=int, default=1)
    ap.add_argument("--python", default=sys.executable, help='interpreter command, e.g. "./isaaclab.sh -p"')
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    out_dir = os.path.expanduser(cfg.run.output_dir)
    gpus = a.gpus.split(",") if a.gpus else detect_gpus()
    jobs = [(g, k) for g in gpus for k in range(a.procs_per_gpu)]
    per = math.ceil(a.total_scenes / len(jobs))
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    procs = []
    for shard, (g, _) in enumerate(jobs):
        cmd = shlex.split(a.python) + [os.path.join(REPO, "scripts", "generate.py"), "--config", a.config, *a.overrides,
                                       f"run.shard_id={shard}", f"run.num_scenes={per}", f"sim.device=cuda:{g}",
                                       "sim.headless=true"]
        log = open(os.path.join(out_dir, "logs", f"shard_{shard:02d}.log"), "w")
        print(f"[launch] shard {shard} on GPU {g}: {' '.join(cmd)}")
        procs.append((shard, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(5)  # stagger startup (shader cache / asset conversion)
    failed = []
    for shard, p, log in procs:
        rc = p.wait()
        log.close()
        print(f"[launch] shard {shard} exited with {rc}")
        if rc != 0:
            failed.append(shard)
    if a.merge:
        merge(out_dir, [os.path.join(out_dir, f"shard_{s:02d}") for s, _, _ in procs if s not in failed])
    if failed:
        print(f"[launch] FAILED shards: {failed} — see {out_dir}/logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
