"""Runtime provenance and execution environment capture for promptdiff."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

from promptdiff import __version__
from promptdiff.core.models import RunProvenance


def get_git_info() -> tuple[str | None, str | None, bool | None]:
    """Safely obtain git commit hash, branch, and dirty status without throwing."""
    try:
        res_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        commit = res_commit.stdout.strip() if res_commit.returncode == 0 and res_commit.stdout.strip() else None

        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        branch = res_branch.stdout.strip() if res_branch.returncode == 0 and res_branch.stdout.strip() else None

        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        dirty = bool(res_status.stdout.strip()) if res_status.returncode == 0 else None

        return commit, branch, dirty
    except Exception:
        return None, None, None


def capture_provenance(extra_env_keys: list[str] | None = None) -> RunProvenance:
    """Capture runtime provenance metadata safely without leaking secrets."""
    commit, branch, dirty = get_git_info()

    # Capture safe, relevant environment variables (never capture API keys/secrets)
    safe_keys = ["CI", "GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_REF"]
    if extra_env_keys:
        safe_keys.extend(extra_env_keys)

    import platform

    from promptdiff.pricing import PRICING_TABLE_VERSION

    env_dict = {k: os.environ[k] for k in safe_keys if k in os.environ}
    command_str = " ".join(sys.argv) if sys.argv else None
    now_iso = datetime.now(timezone.utc).isoformat()

    return RunProvenance(
        framework_version=__version__,
        runner_version=__version__,
        python_version=platform.python_version(),
        platform=sys.platform,
        pricing_version=PRICING_TABLE_VERSION,
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        timestamp_utc=now_iso,
        timestamp=now_iso,
        environment=env_dict,
        command=command_str,
    )
