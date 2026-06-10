"""Thin wrapper around the Vul4J CLI, executed inside the alldeps Docker image.

The host only needs Docker. Java, Maven, and the right JDK versions all live
inside ``tuhhsse/vul4j:alldeps``. Override with VUL4J_IMAGE if needed.

Each verify() call gets its own host tempdir, which we bind-mount into the
container at /work. Vul4J writes the project tree and ``VUL4J/testing_results.json``
back into that tempdir, so we can read results from the host after the container
exits.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_IMAGE = os.environ.get("VUL4J_IMAGE", "tuhhsse/vul4j:alldeps")
CONTAINER_WORK = "/work"
PROJECT_SUBDIR = "project"  # /work/project inside the container
RESULTS_RELPATH = Path("VUL4J") / "testing_results.json"
INFO_RELPATH = Path("vulnerability_info.json")


@dataclass
class CmdResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def render(self) -> str:
        return (
            f"$ {' '.join(shlex.quote(c) for c in self.cmd)}\n"
            f"[exit {self.returncode}]\n"
            f"--- stdout ---\n{self.stdout}\n"
            f"--- stderr ---\n{self.stderr}\n"
        )


@contextmanager
def ephemeral_workdir(prefix: str = "vul4j-") -> Iterator[Path]:
    """Tempdir that lives for one verify() call, cleaned up after."""
    d = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _docker_run(workdir: Path, vul4j_args: list[str], timeout: int) -> CmdResult:
    """Run a single `vul4j ...` invocation inside the alldeps image.

    The host workdir is mounted at /work and the container CWD is /work, so any
    relative paths the caller passes (e.g. ``-d project``) line up on both sides.
    """
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workdir}:{CONTAINER_WORK}",
        "-w", CONTAINER_WORK,
        DEFAULT_IMAGE,
        "vul4j", *vul4j_args,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as e:
        return CmdResult(
            cmd=cmd, returncode=124,
            stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or ""),
            stderr=f"TIMEOUT after {timeout}s",
        )
    return CmdResult(cmd=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def checkout(workdir: Path, vuln_id: str, timeout: int = 600) -> CmdResult:
    """`vul4j checkout -i <id> -d project` — populates {workdir}/project."""
    return _docker_run(workdir, ["checkout", "-i", vuln_id, "-d", PROJECT_SUBDIR], timeout)


def apply(workdir: Path, version: str, timeout: int = 120) -> CmdResult:
    """`vul4j apply -d project -v <version>` (e.g. ``human_patch`` or ``vulnerable``)."""
    return _docker_run(workdir, ["apply", "-d", PROJECT_SUBDIR, "-v", version], timeout)


def compile_(workdir: Path, timeout: int = 1800) -> CmdResult:
    """`vul4j compile -d project`."""
    return _docker_run(workdir, ["compile", "-d", PROJECT_SUBDIR], timeout)


def test(workdir: Path, batch_type: str | None = None, timeout: int = 3600) -> CmdResult:
    """`vul4j test -d project [-b <batch_type>]`.

    batch_type values supported by vul4j: ``povs`` (PoV/exploit tests only)
    and ``all`` (PoV + regression). Omit to use the project default.
    """
    args = ["test", "-d", PROJECT_SUBDIR]
    if batch_type:
        args += ["-b", batch_type]
    return _docker_run(workdir, args, timeout)


def read_testing_results(workdir: Path) -> dict | None:
    """Load VUL4J/testing_results.json from the host side of the bind mount."""
    p = workdir / PROJECT_SUBDIR / RESULTS_RELPATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def read_vulnerability_info(workdir: Path) -> dict | None:
    """Load vulnerability_info.json from the checked-out project root.

    Contains ``modified_files`` (list of relative file paths the patch must
    target) and the PoV/regression test class lists.
    """
    p = workdir / PROJECT_SUBDIR / INFO_RELPATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def target_file_path(workdir: Path, info: dict) -> Path:
    """Resolve the (single) file the patch must replace, on the host side.

    Vul4J's ``modified_files`` is a list; the verifier currently supports the
    common single-file case. Multi-file patches will raise so we notice early
    instead of silently patching just the first.
    """
    modified = info.get("modified_files") or []
    if not modified:
        raise RuntimeError("vulnerability_info.json has no modified_files")
    if len(modified) > 1:
        raise NotImplementedError(
            f"multi-file patches not yet supported (got {modified}); "
            "extend verify() to accept a {path: code} mapping",
        )
    return workdir / PROJECT_SUBDIR / modified[0]


def write_patched_file(workdir: Path, info: dict, patched_code: str) -> Path:
    """Overwrite the target file with the candidate patch; return its path."""
    target = target_file_path(workdir, info)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(patched_code)
    return target


def docker_available() -> bool:
    return shutil.which("docker") is not None
