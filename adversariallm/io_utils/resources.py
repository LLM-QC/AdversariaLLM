"""Helpers for accessing packaged non-code resources."""

from importlib.resources import files
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def packaged_conf_dir():
    """Return the packaged Hydra config directory."""
    path = files("adversariallm").joinpath("resources", "conf")
    if path.is_dir():
        return path
    return _project_root() / "conf"


def packaged_chat_templates_dir():
    """Return the packaged chat templates directory."""
    path = files("adversariallm").joinpath("resources", "chat_templates", "chat_templates")
    if path.is_dir():
        return path
    return _project_root() / "chat_templates" / "chat_templates"
