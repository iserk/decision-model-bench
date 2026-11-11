"""What a result was measured on: machine, OS and the package versions in each virtualenv.

Stored in every results file, so a published number can be traced to its software and hardware.
"""

from __future__ import annotations

import json
import platform
import subprocess
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


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine() -> dict:
    mem = _sysctl("hw.memsize")
    return {"os": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
            "kernel": platform.release(), "arch": platform.machine(),
            "chip": _sysctl("machdep.cpu.brand_string"), "model": _sysctl("hw.model"),
            "memory_gb": round(int(mem) / 2**30) if mem else None,
            "cpu_cores": _sysctl("hw.ncpu"), "python": platform.python_version()}


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
