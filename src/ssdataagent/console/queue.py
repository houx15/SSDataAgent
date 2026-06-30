"""In-process, subprocess-backed job queue. Sequential by default.

Each job shells out to scripts/run_experiment.py; the runner CLI writes the
flags (done.flag/failed.flag) on disk, and the queue mirrors a coarse status
into the SQLite index with source='console' until sync reconciles it.
"""
from __future__ import annotations

import queue as _queue
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from ssdataagent.config import REPO_ROOT

Runner = Callable[[str, Path], int]

_SENTINEL = object()


def _default_runner(name: str, log_path: Path) -> int:
    """Sentinel identity — the real subprocess is created inside the worker."""
    raise RuntimeError(  # pragma: no cover
        "_default_runner must not be called directly; "
        "the worker branches on identity and spawns the Popen itself."
    )


class JobQueue:
    def __init__(self, conn: sqlite3.Connection, results_root: Path, *,
                 concurrency: int = 1, runner: Runner | None = None):
        self._conn = conn
        self._lock = threading.Lock()
        self._root = Path(results_root)
        self._concurrency = max(1, concurrency)
        self._runner = runner if runner is not None else _default_runner
        self._q: _queue.Queue = _queue.Queue()
        self._threads: list[threading.Thread] = []
        self._running: dict[str, subprocess.Popen] = {}
        self._stop = threading.Event()

    # --- status writes (serialized) ---
    def _set_status(self, name: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO experiments(name, status, source)
                   VALUES (?, ?, 'console')
                   ON CONFLICT(name) DO UPDATE SET status=excluded.status,
                                                   source='console'""",
                (name, status),
            )
            self._conn.commit()

    def enqueue(self, name: str) -> None:
        self._set_status(name, "queued")
        self._q.put(name)

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for _ in range(self._concurrency):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._q.put(_SENTINEL)
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []

    def cancel(self, name: str) -> bool:
        with self._lock:
            proc = self._running.get(name)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return True
        return False

    def wait_idle(self, timeout: float = 10.0) -> None:
        done = threading.Event()

        def _waiter() -> None:
            self._q.join()
            done.set()

        t = threading.Thread(target=_waiter, daemon=True)
        t.start()
        done.wait(timeout=timeout)

    def _worker(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            try:
                if item is _SENTINEL:
                    return
                name = item
                self._set_status(name, "running")
                log_path = self._root / name / "run.log"
                code = 1
                try:
                    if self._runner is _default_runner:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open("w") as log:
                            proc = subprocess.Popen(
                                [sys.executable,
                                 str(REPO_ROOT / "scripts" / "run_experiment.py"),
                                 "--experiment", name],
                                stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(REPO_ROOT),
                            )
                            with self._lock:
                                self._running[name] = proc
                            code = proc.wait()
                    else:
                        code = self._runner(name, log_path)
                except Exception:
                    code = 1
                finally:
                    with self._lock:
                        self._running.pop(name, None)
                self._set_status(name, "done" if code == 0 else "failed")
            finally:
                self._q.task_done()
