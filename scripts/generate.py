"""Generate a bin-picking dataset shard with Isaac Lab.

    python scripts/generate.py --config configs/default.yaml [key.sub=value ...] [--gui]

Examples:
    # watch 1 bin fill/settle with the GUI, 2 scenes
    python scripts/generate.py --config configs/default.yaml --gui run.num_envs=1 run.num_scenes=2
    # headless production shard
    python scripts/generate.py --config configs/default.yaml run.num_envs=16 run.num_scenes=5000

Use scripts/launch_multi_gpu.py to run several shards in parallel (one per GPU).
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from binsim.config import load_config  # noqa: E402  (pure python, safe before the app starts)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
parser.add_argument("--gui", action="store_true", help="open the Isaac Sim window (overrides sim.headless)")
parser.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. run.num_envs=16")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cfg = load_config(args.config, args.overrides)
if args.gui:
    cfg.sim.headless = False

args.headless = bool(cfg.sim.get("headless", True))
args.enable_cameras = True
args.device = cfg.sim.get("device", "cuda:0")
if cfg.sim.get("experience"):
    args.experience = os.path.expanduser(cfg.sim.experience)
if not args.headless and hasattr(args, "visualizer"):
    args.visualizer = ["kit"]  # Isaac Lab 3.x visualizer selection (as in the legacy script)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from binsim.sim.runner import Generator  # noqa: E402  (needs the running app)


def main():
    gen = Generator(cfg, simulation_app)
    try:
        gen.run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
