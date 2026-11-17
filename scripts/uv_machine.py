"""Select a GPU-specific uv project and keep its environment at the repository root."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

ENVIRONMENTS = {"py39-jax0425": "3.9", "py311-jax062": "3.11"}
MACHINES = {"rtx4070ti": "py39-jax0425", "rtx5060ti": "py311-jax062"}
ROOT = Path(__file__).resolve().parents[1]


def detect_machine():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot detect GPU; select --environment py39-jax0425 or py311-jax062.") from error
    matches = set()
    for name in result.stdout.splitlines():
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        for machine in MACHINES:
            if machine in normalized:
                matches.add(machine)
    if len(matches) != 1:
        raise ValueError("GPU selection is ambiguous; select --environment py39-jax0425 or py311-jax062.")
    return matches.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--environment", choices=ENVIRONMENTS, help="Select a Python/JAX environment explicitly.")
    selection.add_argument("--machine", choices=MACHINES, help="Select the environment for a GPU explicitly.")
    parser.add_argument("uv_args", nargs=argparse.REMAINDER, help="uv command and arguments, e.g. sync --locked")
    args = parser.parse_args()
    if not args.uv_args:
        parser.error("provide a uv command, for example: sync --locked")
    try:
        profile = args.environment or MACHINES[args.machine or detect_machine()]
    except ValueError as error:
        parser.error(str(error))
    executable = shutil.which("uv")
    if executable is None:
        parser.error("uv is not installed or is missing from PATH")
    project = ROOT / "environments" / profile
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv")
    print(f"{profile}: Python {ENVIRONMENTS[profile]}, project {project.relative_to(ROOT)}, environment .venv", flush=True)
    os.chdir(ROOT)
    os.execvpe(executable, [executable, "--project", str(project), *args.uv_args], env)


if __name__ == "__main__":
    main()
