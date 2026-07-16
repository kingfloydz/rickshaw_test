"""Low-level hashing helpers shared by artifacts and provenance."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file's exact bytes."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["sha256_file"]
