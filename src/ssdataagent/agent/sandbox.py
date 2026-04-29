from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    timed_out: bool


class Sandbox:
    """Run Python code blocks in a fresh subprocess with a shared workspace dir.

    Each ``run()`` writes step_NN.py to the workspace and shells out to
    ``python step_NN.py``. State persists across calls *only* via files written
    to the workspace, not via Python interpreter state.
    """

    def __init__(self, workspace_root: Path | None = None, timeout: int = 60):
        parent = workspace_root or Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(tempfile.mkdtemp(prefix="ssdataagent_", dir=parent))
        self.timeout = timeout
        self._step = 0

    def stage_file(self, name: str, content: bytes | str) -> Path:
        path = self.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        return path

    def run(self, code: str) -> SandboxResult:
        self._step += 1
        script = self.workspace / f"step_{self._step:03d}.py"
        script.write_text(code)
        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [sys.executable, script.name],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = _decode(e.stdout)
            stderr = _decode(e.stderr)
            rc = -1
        duration = time.monotonic() - start
        return SandboxResult(
            stdout=stdout, stderr=stderr, exit_code=rc,
            duration_s=duration, timed_out=timed_out,
        )

    def close(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


def _decode(x: bytes | str | None) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    return x
