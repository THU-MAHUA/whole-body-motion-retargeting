"""Resolve resources from this editable checkout, never from the working directory."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "robots"
MOTIONS = ROOT / "motions"


def body_model_dir(value=None):
    """Body model weights are user supplied and are not distributed here."""
    value = value or os.environ.get("SMPLX_MODEL_DIR")
    if not value:
        raise ValueError("Provide --body_model_dir or set SMPLX_MODEL_DIR to your licensed body-model directory.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Body-model directory does not exist: {path}")
    return path
