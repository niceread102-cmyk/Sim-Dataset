"""Background post-processing so the GPU keeps simulating while PNGs/JSON are written.
numpy/PIL/zlib release the GIL for the heavy parts, so a thread pool is enough."""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor


class AsyncWriter:
    def __init__(self, threads: int = 8, max_pending: int | None = None):
        self.pool = ThreadPoolExecutor(max_workers=max(threads, 1), thread_name_prefix="binsim-io")
        # bound memory: block the producer when too many scenes are in flight
        self.sem = threading.BoundedSemaphore(max_pending or threads * 2)
        self.errors: list[str] = []
        self.done = 0

    def submit(self, fn, *args, **kwargs) -> Future:
        self.sem.acquire()

        def run():
            try:
                return fn(*args, **kwargs)
            except Exception:
                self.errors.append(traceback.format_exc())
                print(f"[ERROR] writer task failed:\n{self.errors[-1]}", flush=True)
            finally:
                self.done += 1
                self.sem.release()

        return self.pool.submit(run)

    def close(self):
        self.pool.shutdown(wait=True)
        if self.errors:
            print(f"[ERROR] {len(self.errors)} writer task(s) failed (first error above)", flush=True)
