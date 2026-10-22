"""Exclusive execution claims persist after success or failure; retries use new directories."""

import os
import subprocess
from pathlib import Path
from .common import dump


def claim_task(path, command=None):
    path = Path(path)
    dump(
        path / "execution_claim.json",
        {
            "pid": os.getpid(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "command": command,
        },
    )


def run_claimed(path, command):
    path = Path(path).resolve()
    claim_task(path, command)
    with (
        (path / "driver.out").open("x") as stdout,
        (path / "driver.err").open("x") as stderr,
    ):
        result = subprocess.run(command, cwd=path, stdout=stdout, stderr=stderr)
    dump(path / "exit_status.json", {"returncode": result.returncode})
    result.check_returncode()
