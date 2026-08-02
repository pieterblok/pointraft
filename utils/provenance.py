"""Small helpers for auditable benchmark artifacts."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch_geometric


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str:
    """Return the checked-out Git commit.

    Parameters
    ----------
    project_root
        Repository worktree root.

    Returns
    -------
    str
        Full commit hash, or ``unknown`` when Git is unavailable.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_source_dirty(project_root: Path) -> bool | None:
    """Report whether tracked or untracked source files differ from ``HEAD``.

    Generated data and benchmark artifacts are excluded because paper
    worktrees intentionally replace the public dataset with a private symlink.

    Parameters
    ----------
    project_root
        Repository worktree root.

    Returns
    -------
    bool | None
        Dirty state, or ``None`` when Git is unavailable.
    """
    try:
        output = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)data/3DPotatoTwin",
                ":(exclude)weights",
                ":(exclude)results",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_versions() -> dict[str, str]:
    """Return runtime versions relevant to benchmark reproduction.

    Returns
    -------
    dict[str, str]
        PyTorch, PyG, CUDA, and cuDNN version strings.
    """
    return {
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON provenance record with stable formatting.

    Parameters
    ----------
    path
        Destination path.
    payload
        JSON-compatible record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
