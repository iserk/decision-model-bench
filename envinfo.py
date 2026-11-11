"""What a result was measured on: machine, OS and the package versions in each virtualenv.

Stored in every results file, so a published number can be traced to its software and hardware.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
HOME = str(Path.home())
# Each system's environment, relative to ROOT; `.` is the main project.
ENVS = {"strands": ".", "decider": "systems/decider", "openjev": "systems/open-jev",
        "jevbench": "systems/jevbench"}

# Run inside an environment's interpreter: every installed distribution, with the commit of
# any installed from git.
_LIST_PACKAGES = r"""
import importlib.metadata as m, json
out = {}
for d in m.distributions():
    v = d.version
    try:
        info = json.loads(d.read_text("direct_url.json") or "{}")
        commit = info.get("vcs_info", {}).get("commit_id")
        if commit:
            v = f"{v} ({info['url']}@{commit})"
    except (ValueError, KeyError):
        pass
    out[d.metadata["Name"].lower()] = v
print(json.dumps(dict(sorted(out.items()))))
"""


def _run(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _linux() -> dict:
    def field(path: str, key: str, sep: str = ":") -> str | None:
        try:
            for line in Path(path).read_text().splitlines():
                if line.startswith(key):
                    return line.split(sep, 1)[1].strip().strip('"')
        except OSError:
            pass
        return None

    mem_kb = field("/proc/meminfo", "MemTotal")
    # ARM kernels leave "model name" out of /proc/cpuinfo; lscpu decodes the part number.
    chip = field("/proc/cpuinfo", "model name") or next(
        (line.split(":", 1)[1].strip() for line in (_run("lscpu") or "").splitlines()
         if line.strip().startswith("Model name")), None)
    return {"os": field("/etc/os-release", "PRETTY_NAME", "=") or platform.platform(),
            "chip": chip, "model": None,
            "memory_gb": round(int(mem_kb.split()[0]) / 2**20) if mem_kb else None,
            "cpu_cores": os.cpu_count()}


def machine() -> dict:
    if sys.platform == "darwin":
        mem = _run("sysctl", "-n", "hw.memsize")
        info = {"os": f"macOS {platform.mac_ver()[0]}",
                "chip": _run("sysctl", "-n", "machdep.cpu.brand_string"),
                "model": _run("sysctl", "-n", "hw.model"),
                "memory_gb": round(int(mem) / 2**30) if mem else None,
                "cpu_cores": _run("sysctl", "-n", "hw.ncpu")}
    else:
        info = _linux()
    # One line per NVIDIA GPU: name, memory, driver.
    gpus = _run("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader")
    return {**info, "gpus": gpus.splitlines() if gpus else None, "kernel": platform.release(),
            "arch": platform.machine(), "python": platform.python_version()}


def packages(env: str) -> dict | str:
    python = ROOT / ENVS[env] / ".venv" / "bin" / "python"
    if not python.exists():
        return "not installed"
    try:
        out = subprocess.run([str(python), "-c", _LIST_PACKAGES], capture_output=True, text=True,
                             check=True, timeout=60).stdout
        return json.loads(out)
    except (subprocess.SubprocessError, ValueError) as exc:
        return f"unavailable: {exc}"


def environment(envs: list[str]) -> dict:
    return {"machine": machine(), "packages": {e: packages(e) for e in dict.fromkeys(envs)}}


def scrub(obj):
    """Replace the home directory with `~` in every string, so results don't reveal the user name."""
    if isinstance(obj, str):
        return obj.replace(HOME, "~")
    if isinstance(obj, dict):
        return {scrub(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def scrub_file(path: Path) -> None:
    path.write_text(path.read_text(errors="replace").replace(HOME, "~"))
