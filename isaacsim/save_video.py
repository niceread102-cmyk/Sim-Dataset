#!/usr/bin/env python3
"""save_video.py: records the Perspective view of the running Isaac Sim window to video/<n>.mp4 (n = 1, 2, 3, ...) until Ctrl+C or until Isaac Sim closes.

    python3 save_video.py [--full]        (--full = the whole Isaac Sim window instead of the viewport only)

Prints one line and exits when no Isaac Sim window is open. The file is written in pieces, so it stays playable if Isaac Sim is killed during the recording.
The window must not be minimized (X11 cannot read a minimized window); other windows in front of it are fine on a compositing desktop.
"""
import argparse
import glob
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent / 'video'
WINDOW_CLASS = '^Isaac Sim'
VIEWPORT = (0.022, 0.022, 0.695, 0.685)
FPS = 30
DISPLAY = ':0'


def find_window(pattern=WINDOW_CLASS):
    out = subprocess.run(['xdotool', 'search', '--class', pattern], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def window_size(window):
    info = subprocess.run(['xwininfo', '-id', str(window)], capture_output=True, text=True)
    if info.returncode != 0:
        return None
    return int(re.search(r'Width: (\d+)', info.stdout).group(1)), int(re.search(r'Height: (\d+)', info.stdout).group(1))


def find_ffmpeg():
    found = shutil.which('ffmpeg')
    if found:
        return found
    hits = sorted(glob.glob(str(Path.home() / 'miniconda3/envs/*/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*')))
    return hits[0] if hits else None


def next_name():
    VIDEO_DIR.mkdir(exist_ok=True)
    numbers = [int(p.stem) for p in VIDEO_DIR.glob('*.mp4') if p.stem.isdigit()]
    return VIDEO_DIR / f'{max(numbers, default=0) + 1}.mp4'


def crop_filter(width, height, full):
    if full:
        return f'crop={width // 2 * 2}:{height // 2 * 2}:0:0'
    x0, y0, x1, y1 = VIEWPORT
    return f'crop={int((x1 - x0) * width) // 2 * 2}:{int((y1 - y0) * height) // 2 * 2}:{int(x0 * width)}:{int(y0 * height)}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--full', action='store_true')
    cli = parser.parse_args()

    window = find_window()
    size = window_size(window) if window else None
    if size is None:
        print('Isaac Sim is not running (no Isaac Sim window found).')
        return
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print('ffmpeg not found (install ffmpeg or imageio_ffmpeg).')
        return

    path = next_name()
    proc = subprocess.Popen(
        [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'x11grab', '-window_id', str(window), '-framerate', str(FPS), '-draw_mouse', '0', '-i', DISPLAY,
         '-vf', crop_filter(*size, cli.full), '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p', '-g', str(FPS),
         '-movflags', '+frag_keyframe+empty_moov+default_base_moof', str(path)],
        stdin=subprocess.PIPE)
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.append(1))
    start = time.time()
    print(f'recording {path} (Ctrl+C or closing Isaac Sim stops it)')
    while not stop and proc.poll() is None and window_size(window) is not None:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            proc.stdin.write(b'q')
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait()
    print(f'saved {path} ({time.time() - start:.0f} s)')


if __name__ == '__main__':
    sys.exit(main())
