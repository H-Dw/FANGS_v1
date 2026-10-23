"""Resolve external tool locations without machine-specific defaults."""

from __future__ import annotations

import os
import shutil
from typing import Optional, Sequence


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = os.path.expanduser(str(value).strip())
    return text or None


def _bin_dir(path: str) -> str:
    if os.path.isdir(path):
        directory = path
    elif os.path.isfile(path):
        directory = os.path.dirname(path)
    else:
        raise FileNotFoundError(path)
    return directory if directory.endswith(os.sep) else directory + os.sep


def _directory_has(directory: str, names: Sequence[str]) -> bool:
    return all(os.path.isfile(os.path.join(directory, name)) for name in names)


def resolve_tool_prefix(
    explicit: Optional[str],
    env_name: str,
    executables: Sequence[str],
) -> str:
    """Return a bin directory prefix ending in a path separator.

    ``explicit`` wins, then ``env_name``, then a single ``PATH`` directory that
    contains every name in ``executables``. The prefix is concatenated with the
    executable name by the existing search commands.
    """
    names = ", ".join(executables)
    given = _clean(explicit)
    if given is not None:
        try:
            prefix = _bin_dir(given)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{given} does not exist. Pass a directory that contains {names}, "
                f"or set {env_name}."
            ) from None
        if not _directory_has(prefix, executables):
            raise FileNotFoundError(
                f"{prefix} does not contain {names}. Set {env_name} to that directory."
            )
        return prefix

    from_env = _clean(os.environ.get(env_name))
    if from_env is not None:
        return resolve_tool_prefix(from_env, env_name, executables)

    parents = []
    for name in executables:
        found = shutil.which(name)
        if not found:
            parents = []
            break
        parents.append(os.path.dirname(os.path.abspath(found)))
    if parents and len(set(parents)) == 1 and _directory_has(parents[0], executables):
        return _bin_dir(parents[0])

    raise FileNotFoundError(
        f"Could not find {names} in one directory. Export {env_name} to that bin directory, "
        "or add it to PATH."
    )


def resolve_directory(explicit: Optional[str], env_name: str, label: str) -> str:
    """Return an existing directory from an argument or environment variable."""
    given = _clean(explicit)
    source = "argument"
    if given is None:
        given = _clean(os.environ.get(env_name))
        source = env_name
    if given is None:
        raise FileNotFoundError(
            f"{label} is unset. Export {env_name} to the {label} directory."
        )
    if not os.path.isdir(given):
        raise FileNotFoundError(f"{label} directory from {source} does not exist: {given}")
    return os.path.abspath(given)
