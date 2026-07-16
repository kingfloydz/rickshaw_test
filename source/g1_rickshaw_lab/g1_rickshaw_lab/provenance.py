"""Checkpoint provenance, validation, and atomic persistence.

The implementation guide makes provenance part of the policy ABI.  A
checkpoint without this metadata, with a different joint order, with modified
configuration files, or built against another RSL-RL revision is rejected
before any state dictionary is consumed.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any

from ._hashing import sha256_file
from .configuration import FIXED_G1_JOINT_ORDER, validate_joint_order


PROVENANCE_SCHEMA_VERSION = 1
CHECKPOINT_METADATA_KEY = "g1_rickshaw_provenance"
RSL_RL_VERSION = "v5.0.1"
RSL_RL_COMMIT = "3ac56acd3376f2952eb636a133f4b5aa30142552"
PINNED_RSL_RL_VERSION = RSL_RL_VERSION
PINNED_RSL_RL_COMMIT = RSL_RL_COMMIT
CUDA_NOT_AVAILABLE = "none"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceError(ValueError):
    """Raised when checkpoint provenance is absent, malformed, or mismatched."""


class ProvenanceDependencyError(RuntimeError):
    """Raised when checkpoint IO dependencies are unavailable."""


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ProvenanceDependencyError(
            "PyTorch is required for G1 rickshaw checkpoint IO and runtime provenance; "
            "install the PyTorch build required by Isaac Lab."
        ) from exc
    return torch


def _normalize_config_files(
    config_files: Mapping[str, str | Path] | Sequence[str | Path],
) -> dict[str, Path]:
    if isinstance(config_files, Mapping):
        if not config_files:
            raise ProvenanceError("at least one configuration file is required")
        result: dict[str, Path] = {}
        for label, path in config_files.items():
            if not isinstance(label, str) or not label:
                raise ProvenanceError("configuration hash labels must be non-empty strings")
            result[label] = Path(path)
        return result
    if isinstance(config_files, (str, bytes)) or not isinstance(config_files, Sequence):
        raise TypeError("config_files must be a label-to-path mapping or path sequence")
    if not config_files:
        raise ProvenanceError("at least one configuration file is required")
    result = {}
    for value in config_files:
        path = Path(value)
        # Preserve the caller's spelling as the stable label.  Callers that
        # need relocatable checkpoints should pass an explicit label mapping.
        label = os.fspath(value)
        if label in result:
            raise ProvenanceError(f"duplicate configuration path label {label!r}")
        result[label] = path
    return result


def hash_config_files(
    config_files: Mapping[str, str | Path] | Sequence[str | Path],
) -> Mapping[str, str]:
    """Hash every runtime configuration file under stable caller labels."""

    normalized = _normalize_config_files(config_files)
    hashes: dict[str, str] = {}
    for label, path in normalized.items():
        if not path.is_file():
            raise FileNotFoundError(f"configuration file does not exist: {path}")
        hashes[label] = sha256_file(path)
    return MappingProxyType(hashes)


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceError(f"{path} must be a non-empty string")
    return value.strip()


def _commit(value: Any, path: str) -> str:
    commit = _required_text(value, path).lower()
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ProvenanceError(f"{path} must be a full 40-character hexadecimal git commit")
    return commit


def _hashes(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ProvenanceError("config_sha256 must be a non-empty label-to-SHA256 mapping")
    result: dict[str, str] = {}
    for label, digest in value.items():
        if not isinstance(label, str) or not label:
            raise ProvenanceError("config_sha256 labels must be non-empty strings")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest.lower()) is None:
            raise ProvenanceError(f"config_sha256[{label!r}] must be a 64-character SHA256")
        result[label] = digest.lower()
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """The immutable policy ABI persisted under ``CHECKPOINT_METADATA_KEY``."""

    isaac_sim_version: str
    isaaclab_commit: str
    pytorch_version: str
    cuda_version: str
    config_sha256: Mapping[str, str]
    joint_order: tuple[str, ...] = FIXED_G1_JOINT_ORDER
    rsl_rl_version: str = RSL_RL_VERSION
    rsl_rl_commit: str = RSL_RL_COMMIT
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PROVENANCE_SCHEMA_VERSION
        ):
            raise ProvenanceError(
                f"unsupported provenance schema_version={self.schema_version!r}; "
                f"expected {PROVENANCE_SCHEMA_VERSION}"
            )
        isaac_sim_version = _required_text(self.isaac_sim_version, "isaac_sim_version")
        isaaclab_commit = _commit(self.isaaclab_commit, "isaaclab_commit")
        pytorch_version = _required_text(self.pytorch_version, "pytorch_version")
        cuda_version = _required_text(self.cuda_version, "cuda_version")
        rsl_rl_version = _required_text(self.rsl_rl_version, "rsl_rl_version")
        rsl_rl_commit = _commit(self.rsl_rl_commit, "rsl_rl_commit")
        if rsl_rl_version != RSL_RL_VERSION:
            raise ProvenanceError(
                f"RSL-RL version mismatch: checkpoint has {rsl_rl_version!r}, "
                f"project requires {RSL_RL_VERSION!r}"
            )
        if rsl_rl_commit != RSL_RL_COMMIT:
            raise ProvenanceError(
                f"RSL-RL commit mismatch: checkpoint has {rsl_rl_commit}, "
                f"project requires {RSL_RL_COMMIT}"
            )
        joint_order = validate_joint_order(self.joint_order, path="checkpoint joint_order")
        config_sha256 = _hashes(self.config_sha256)
        object.__setattr__(self, "isaac_sim_version", isaac_sim_version)
        object.__setattr__(self, "isaaclab_commit", isaaclab_commit)
        object.__setattr__(self, "pytorch_version", pytorch_version)
        object.__setattr__(self, "cuda_version", cuda_version)
        object.__setattr__(self, "rsl_rl_version", rsl_rl_version)
        object.__setattr__(self, "rsl_rl_commit", rsl_rl_commit)
        object.__setattr__(self, "joint_order", joint_order)
        object.__setattr__(self, "config_sha256", config_sha256)

    @property
    def torch_version(self) -> str:
        """Compatibility alias for code that calls PyTorch simply ``torch``."""

        return self.pytorch_version

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointMetadata":
        if not isinstance(value, Mapping):
            raise ProvenanceError("checkpoint provenance metadata must be a mapping")
        expected = {
            "schema_version",
            "isaac_sim_version",
            "isaaclab_commit",
            "rsl_rl_version",
            "rsl_rl_commit",
            "pytorch_version",
            "cuda_version",
            "joint_order",
            "config_sha256",
        }
        actual = set(value)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"unknown={extra}")
            raise ProvenanceError("invalid checkpoint provenance fields: " + ", ".join(details))
        return cls(
            schema_version=value["schema_version"],
            isaac_sim_version=value["isaac_sim_version"],
            isaaclab_commit=value["isaaclab_commit"],
            rsl_rl_version=value["rsl_rl_version"],
            rsl_rl_commit=value["rsl_rl_commit"],
            pytorch_version=value["pytorch_version"],
            cuda_version=value["cuda_version"],
            joint_order=value["joint_order"],
            config_sha256=value["config_sha256"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "isaac_sim_version": self.isaac_sim_version,
            "isaaclab_commit": self.isaaclab_commit,
            "rsl_rl_version": self.rsl_rl_version,
            "rsl_rl_commit": self.rsl_rl_commit,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "joint_order": list(self.joint_order),
            "config_sha256": dict(self.config_sha256),
        }


def _git_commit_at(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    candidate = completed.stdout.strip().lower()
    return candidate if _GIT_COMMIT_RE.fullmatch(candidate) is not None else None


def _module_paths(module_name: str) -> tuple[Path, ...]:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return ()
    if spec is None:
        return ()
    paths: list[Path] = []
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        paths.append(Path(spec.origin).resolve().parent)
    if spec.submodule_search_locations:
        paths.extend(Path(value).resolve() for value in spec.submodule_search_locations)
    return tuple(dict.fromkeys(paths))


def _discover_git_commit(module_name: str, explicit_root: str | Path | None) -> str | None:
    starts = (Path(explicit_root).resolve(),) if explicit_root is not None else _module_paths(module_name)
    visited: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in visited:
                continue
            visited.add(candidate)
            if (candidate / ".git").exists():
                commit = _git_commit_at(candidate)
                if commit is not None:
                    return commit
    # Worktrees and some installations expose git metadata outside the package
    # path; allow git itself one final discovery attempt from each start.
    for start in starts:
        commit = _git_commit_at(start)
        if commit is not None:
            return commit
    return None


def discover_isaac_sim_version() -> str:
    """Discover Isaac Sim's version without importing or starting Kit."""

    environment_value = os.environ.get("ISAAC_SIM_VERSION")
    if environment_value:
        return _required_text(environment_value, "ISAAC_SIM_VERSION")
    for distribution in ("isaacsim", "isaac-sim", "isaac_sim"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if version:
            return version
    raise ProvenanceError(
        "cannot determine Isaac Sim version; pass isaac_sim_version explicitly or set "
        "ISAAC_SIM_VERSION inside the Isaac Sim environment"
    )


def discover_isaaclab_commit(root: str | Path | None = None) -> str:
    environment_value = os.environ.get("ISAACLAB_COMMIT")
    if environment_value:
        return _commit(environment_value, "ISAACLAB_COMMIT")
    commit = _discover_git_commit("isaaclab", root)
    if commit is None:
        raise ProvenanceError(
            "cannot determine Isaac Lab commit; pass isaaclab_commit explicitly, provide "
            "isaaclab_root, or set ISAACLAB_COMMIT"
        )
    return commit


def discover_rsl_rl_commit(root: str | Path | None = None) -> str:
    environment_value = os.environ.get("RSL_RL_COMMIT")
    commit = (
        _commit(environment_value, "RSL_RL_COMMIT")
        if environment_value
        else _discover_git_commit("rsl_rl", root)
    )
    if commit is None:
        raise ProvenanceError(
            "cannot determine RSL-RL commit; install the pinned source checkout, pass "
            "rsl_rl_commit explicitly, provide rsl_rl_root, or set RSL_RL_COMMIT"
        )
    if commit != RSL_RL_COMMIT:
        raise ProvenanceError(
            f"installed RSL-RL commit is {commit}; required {RSL_RL_COMMIT} ({RSL_RL_VERSION})"
        )
    return commit


def torch_cuda_versions() -> tuple[str, str]:
    torch = _require_torch()
    torch_version = _required_text(torch.__version__, "torch.__version__")
    cuda_value = getattr(torch.version, "cuda", None)
    cuda_version = CUDA_NOT_AVAILABLE if cuda_value is None else _required_text(cuda_value, "torch.version.cuda")
    return torch_version, cuda_version


def collect_checkpoint_metadata(
    config_files: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    isaac_sim_version: str | None = None,
    isaaclab_commit: str | None = None,
    rsl_rl_commit: str | None = None,
    isaaclab_root: str | Path | None = None,
    rsl_rl_root: str | Path | None = None,
    joint_order: Sequence[str] = FIXED_G1_JOINT_ORDER,
) -> CheckpointMetadata:
    """Collect a complete checkpoint ABI, failing if any source is unknown."""

    torch_version, cuda_version = torch_cuda_versions()
    resolved_isaac_sim = (
        discover_isaac_sim_version()
        if isaac_sim_version is None
        else _required_text(isaac_sim_version, "isaac_sim_version")
    )
    resolved_isaaclab = (
        discover_isaaclab_commit(isaaclab_root)
        if isaaclab_commit is None
        else _commit(isaaclab_commit, "isaaclab_commit")
    )
    resolved_rsl_rl = (
        discover_rsl_rl_commit(rsl_rl_root)
        if rsl_rl_commit is None
        else _commit(rsl_rl_commit, "rsl_rl_commit")
    )
    return CheckpointMetadata(
        isaac_sim_version=resolved_isaac_sim,
        isaaclab_commit=resolved_isaaclab,
        rsl_rl_commit=resolved_rsl_rl,
        pytorch_version=torch_version,
        cuda_version=cuda_version,
        joint_order=tuple(joint_order),
        config_sha256=hash_config_files(config_files),
    )


def attach_checkpoint_metadata(
    checkpoint: MutableMapping[str, Any],
    metadata: CheckpointMetadata | Mapping[str, Any],
    *,
    replace: bool = False,
) -> MutableMapping[str, Any]:
    """Attach canonical metadata to a mutable checkpoint mapping in place."""

    if not isinstance(checkpoint, MutableMapping):
        raise TypeError("checkpoint must be a mutable mapping")
    parsed = metadata if isinstance(metadata, CheckpointMetadata) else CheckpointMetadata.from_mapping(metadata)
    if CHECKPOINT_METADATA_KEY in checkpoint and not replace:
        current = CheckpointMetadata.from_mapping(checkpoint[CHECKPOINT_METADATA_KEY])
        if current != parsed:
            raise ProvenanceError(
                f"checkpoint already contains different metadata under {CHECKPOINT_METADATA_KEY!r}"
            )
    checkpoint[CHECKPOINT_METADATA_KEY] = parsed.to_mapping()
    return checkpoint


def extract_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> CheckpointMetadata:
    if not isinstance(checkpoint, Mapping):
        raise ProvenanceError("loaded checkpoint must be a mapping")
    if CHECKPOINT_METADATA_KEY not in checkpoint:
        raise ProvenanceError(
            f"checkpoint is missing required metadata key {CHECKPOINT_METADATA_KEY!r}"
        )
    return CheckpointMetadata.from_mapping(checkpoint[CHECKPOINT_METADATA_KEY])


def _assert_equal(field: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ProvenanceError(f"{field} mismatch: checkpoint={actual!r}, runtime={expected!r}")


def validate_checkpoint_metadata(
    metadata: CheckpointMetadata | Mapping[str, Any],
    *,
    expected: CheckpointMetadata | Mapping[str, Any] | None = None,
    config_files: Mapping[str, str | Path] | Sequence[str | Path] | None = None,
    joint_order: Sequence[str] = FIXED_G1_JOINT_ORDER,
    isaac_sim_version: str | None = None,
    isaaclab_commit: str | None = None,
    pytorch_version: str | None = None,
    cuda_version: str | None = None,
    validate_torch_runtime: bool = False,
) -> CheckpointMetadata:
    """Validate checkpoint metadata against explicit runtime authorities.

    Structural validation, the pinned RSL-RL revision, and the fixed 29-joint
    order are always enforced.  Isaac versions/commits are compared when passed
    explicitly or through ``expected``.  Set ``validate_torch_runtime`` to also
    compare the currently imported PyTorch/CUDA build.
    """

    parsed = metadata if isinstance(metadata, CheckpointMetadata) else CheckpointMetadata.from_mapping(metadata)
    runtime_order = validate_joint_order(joint_order, path="runtime joint_order")
    if parsed.joint_order != runtime_order:
        raise ProvenanceError("checkpoint joint_order differs from runtime joint_order")

    expected_parsed = None
    if expected is not None:
        expected_parsed = (
            expected if isinstance(expected, CheckpointMetadata) else CheckpointMetadata.from_mapping(expected)
        )
        if parsed.to_mapping() != expected_parsed.to_mapping():
            fields = (
                "isaac_sim_version",
                "isaaclab_commit",
                "rsl_rl_version",
                "rsl_rl_commit",
                "pytorch_version",
                "cuda_version",
                "joint_order",
                "config_sha256",
            )
            differing = [
                field for field in fields if getattr(parsed, field) != getattr(expected_parsed, field)
            ]
            raise ProvenanceError(f"checkpoint metadata differs from expected fields: {differing}")

    if config_files is not None:
        current_hashes = hash_config_files(config_files)
        if dict(parsed.config_sha256) != dict(current_hashes):
            missing = sorted(set(parsed.config_sha256) - set(current_hashes))
            extra = sorted(set(current_hashes) - set(parsed.config_sha256))
            changed = sorted(
                key
                for key in set(parsed.config_sha256) & set(current_hashes)
                if parsed.config_sha256[key] != current_hashes[key]
            )
            raise ProvenanceError(
                f"configuration SHA256 mismatch: missing={missing}, unknown={extra}, changed={changed}"
            )

    if isaac_sim_version is not None:
        _assert_equal(
            "Isaac Sim version",
            parsed.isaac_sim_version,
            _required_text(isaac_sim_version, "isaac_sim_version"),
        )
    if isaaclab_commit is not None:
        _assert_equal(
            "Isaac Lab commit",
            parsed.isaaclab_commit,
            _commit(isaaclab_commit, "isaaclab_commit"),
        )
    if validate_torch_runtime:
        runtime_torch, runtime_cuda = torch_cuda_versions()
        pytorch_version = runtime_torch if pytorch_version is None else pytorch_version
        cuda_version = runtime_cuda if cuda_version is None else cuda_version
    if pytorch_version is not None:
        _assert_equal(
            "PyTorch version",
            parsed.pytorch_version,
            _required_text(pytorch_version, "pytorch_version"),
        )
    if cuda_version is not None:
        _assert_equal(
            "CUDA version", parsed.cuda_version, _required_text(cuda_version, "cuda_version")
        )
    return parsed


def validate_checkpoint(
    checkpoint: Mapping[str, Any], **validation_kwargs: Any
) -> CheckpointMetadata:
    """Extract and validate metadata before a caller consumes model weights."""

    return validate_checkpoint_metadata(extract_checkpoint_metadata(checkpoint), **validation_kwargs)


def atomic_torch_save(value: Any, path: str | Path) -> Path:
    """Durably save with a same-directory temporary file and atomic replace."""

    torch = _require_torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def save_checkpoint_atomic(
    checkpoint: Mapping[str, Any],
    path: str | Path,
    *,
    metadata: CheckpointMetadata | Mapping[str, Any] | None = None,
) -> Path:
    """Validate/attach provenance, then atomically persist a checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    payload = dict(checkpoint)
    if metadata is None:
        extract_checkpoint_metadata(payload)
    else:
        attach_checkpoint_metadata(payload, metadata, replace=False)
    return atomic_torch_save(payload, path)


def load_checkpoint_with_validation(
    path: str | Path,
    *,
    map_location: str | Any = "cpu",
    **validation_kwargs: Any,
) -> Mapping[str, Any]:
    """Load a trusted local checkpoint and validate provenance immediately."""

    torch = _require_torch()
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    validate_checkpoint(checkpoint, **validation_kwargs)
    return checkpoint


# Short aliases used by training scripts and downstream deployment wrappers.
atomic_save = save_checkpoint_atomic
load_checkpoint = load_checkpoint_with_validation


__all__ = [
    "CHECKPOINT_METADATA_KEY",
    "CUDA_NOT_AVAILABLE",
    "CheckpointMetadata",
    "PINNED_RSL_RL_COMMIT",
    "PINNED_RSL_RL_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceDependencyError",
    "ProvenanceError",
    "RSL_RL_COMMIT",
    "RSL_RL_VERSION",
    "atomic_save",
    "atomic_torch_save",
    "attach_checkpoint_metadata",
    "collect_checkpoint_metadata",
    "discover_isaac_sim_version",
    "discover_isaaclab_commit",
    "discover_rsl_rl_commit",
    "extract_checkpoint_metadata",
    "hash_config_files",
    "load_checkpoint",
    "load_checkpoint_with_validation",
    "save_checkpoint_atomic",
    "sha256_file",
    "torch_cuda_versions",
    "validate_checkpoint",
    "validate_checkpoint_metadata",
]
