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
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from binsim.config import load_config  # noqa: E402  (pure python, safe before the app starts)

# Render settings from Isaac Lab's isaaclab.python.rendering.kit; without them TiledCamera hits CUDA errors when a
# non-Isaac-Lab experience (isaacsim.exp.full.kit) is used on the GB10 machine.
ISAACLAB_RENDER_SETTINGS = {
    "/omni/replicator/asyncRendering": "false",
    "/app/renderer/waitIdle": "true",
    "/app/hydraEngine/waitIdle": "true",
    "/app/updateOrder/checkForHydraRenderComplete": "1000",
    "/exts/omni.replicator.core/Orchestrator/enabled": "false",
    "/rtx/viewTile/limit": "1000000",
    "/renderer/multiGpu/maxGpuCount": "1",
    "/app/useFabricSceneDelegate": "true",
    "/rtx/hydra/readTransformsFromFabricInRenderDelegate": "true",
}

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
    extra = [f"--{k}={v}" for k, v in ISAACLAB_RENDER_SETTINGS.items()]
    args.kit_args = " ".join([getattr(args, "kit_args", "") or ""] + extra).strip()
if not args.headless and hasattr(args, "visualizer"):
    args.visualizer = ["kit"]  # Isaac Lab 3.x visualizer selection (as in the legacy script)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
# let the app finish its async startup before touching the stage (as the legacy script does); touching the
# stage right away intermittently deadlocks with the Isaac Sim Full experience
for _ in range(10):
    simulation_app.update()
if cfg.sim.get("experience"):
    # Isaac Lab's own rendering .kit files set this; a non-Isaac-Lab experience (e.g. isaacsim.exp.full.kit)
    # doesn't, and TiledCamera refuses to initialize without it.
    from isaaclab.app.settings_manager import get_settings_manager

    get_settings_manager().set("/isaaclab/cameras_enabled", True)

from binsim.sim.runner import Generator  # noqa: E402  (needs the running app)


def main():
    gen = Generator(cfg, simulation_app)
    try:
        gen.run()
    except Exception:
        # print before close(): after a CUDA fault close() can hang and the traceback would never appear
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
