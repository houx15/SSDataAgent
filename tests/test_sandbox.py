import pytest

from ssdataagent.agent.sandbox import Sandbox, SandboxResult


def test_execute_simple_code(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=10)
    try:
        r = sb.run("print(1 + 1)")
    finally:
        sb.close()
    assert isinstance(r, SandboxResult)
    assert r.exit_code == 0
    assert r.stdout.strip() == "2"
    assert r.timed_out is False


def test_pandas_available(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=15)
    try:
        r = sb.run("import pandas as pd; print(pd.DataFrame({'x':[1,2]}).x.mean())")
    finally:
        sb.close()
    assert r.exit_code == 0
    assert "1.5" in r.stdout


def test_timeout(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=2)
    try:
        r = sb.run("while True: pass")
    finally:
        sb.close()
    assert r.timed_out is True
    assert r.exit_code != 0


def test_error_capture(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=10)
    try:
        r = sb.run("undefined_name")
    finally:
        sb.close()
    assert r.exit_code != 0
    assert "NameError" in r.stderr


def test_multi_step_via_files(tmp_path):
    """Stateless model: state persists by writing files to the shared workspace."""
    sb = Sandbox(workspace_root=tmp_path, timeout=15)
    try:
        r1 = sb.run("import json; json.dump({'k': 42}, open('state.json','w'))")
        r2 = sb.run("import json; print(json.load(open('state.json'))['k'])")
    finally:
        sb.close()
    assert r1.exit_code == 0
    assert r2.exit_code == 0
    assert r2.stdout.strip() == "42"


def test_stage_file(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=15)
    try:
        sb.stage_file("greet.txt", "hello\n")
        r = sb.run("print(open('greet.txt').read().strip())")
    finally:
        sb.close()
    assert r.stdout.strip() == "hello"


def test_close_removes_workspace(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=10)
    workspace = sb.workspace
    assert workspace.exists()
    sb.close()
    assert not workspace.exists()


def test_steps_are_numbered(tmp_path):
    sb = Sandbox(workspace_root=tmp_path, timeout=10)
    try:
        sb.run("print('a')")
        sb.run("print('b')")
        scripts = sorted(p.name for p in sb.workspace.glob("step_*.py"))
    finally:
        sb.close()
    assert scripts == ["step_001.py", "step_002.py"]
