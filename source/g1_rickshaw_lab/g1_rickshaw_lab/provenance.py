"""Checkpoint runtime metadata, validation, and atomic persistence.

Only runtime versions and the policy joint order are part of the checkpoint
ABI. Source and configuration files are not fingerprinted, so implementation
edits do not make an otherwise compatible checkpoint unloadable.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import importlib
import importlib.metadata
import os
from pathlib import Path
import tempfile
from typing import Any

from .configuration import FIXED_G1_JOINT_ORDER, validate_joint_order


PROVENANCE_SCHEMA_VERSION = 2
CHECKPOINT_METADATA_KEY = "g1_rickshaw_provenance"
RSL_RL_VERSION = "5.0.1"
CUDA_NOT_AVAILABLE = "none"


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


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceError(f"{path} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """The immutable policy ABI persisted under ``CHECKPOINT_METADATA_KEY``."""

    isaac_sim_version: str
    isaaclab_version: str
    pytorch_version: str
    cuda_version: str
    joint_order: tuple[str, ...] = FIXED_G1_JOINT_ORDER
    rsl_rl_version: str = RSL_RL_VERSION
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
        isaaclab_version = _required_text(self.isaaclab_version, "isaaclab_version")
        pytorch_version = _required_text(self.pytorch_version, "pytorch_version")
        cuda_version = _required_text(self.cuda_version, "cuda_version")
        rsl_rl_version = _required_text(self.rsl_rl_version, "rsl_rl_version")
        if rsl_rl_version != RSL_RL_VERSION:
            raise ProvenanceError(
                f"RSL-RL version mismatch: checkpoint has {rsl_rl_version!r}, "
                f"project requires {RSL_RL_VERSION!r}"
            )
        joint_order = validate_joint_order(self.joint_order, path="checkpoint joint_order")
        object.__setattr__(self, "isaac_sim_version", isaac_sim_version)
        object.__setattr__(self, "isaaclab_version", isaaclab_version)
        object.__setattr__(self, "pytorch_version", pytorch_version)
        object.__setattr__(self, "cuda_version", cuda_version)
        object.__setattr__(self, "rsl_rl_version", rsl_rl_version)
        object.__setattr__(self, "joint_order", joint_order)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointMetadata":
        if not isinstance(value, Mapping):
            raise ProvenanceError("checkpoint provenance metadata must be a mapping")
        expected = {
            "schema_version",
            "isaac_sim_version",
            "isaaclab_version",
            "rsl_rl_version",
            "pytorch_version",
            "cuda_version",
            "joint_order",
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
            isaaclab_version=value["isaaclab_version"],
            rsl_rl_version=value["rsl_rl_version"],
            pytorch_version=value["pytorch_version"],
            cuda_version=value["cuda_version"],
            joint_order=value["joint_order"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "isaac_sim_version": self.isaac_sim_version,
            "isaaclab_version": self.isaaclab_version,
            "rsl_rl_version": self.rsl_rl_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "joint_order": list(self.joint_order),
        }


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


def _discover_package_version(distribution: str, module_name: str) -> str | None:
    try:
        value = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        value = None
    if value:
        return value
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    return getattr(module, "__version__", None)


def discover_isaaclab_version() -> str:
    """Discover the installed Isaac Lab release without inspecting Git state."""

    environment_value = os.environ.get("ISAACLAB_VERSION")
    if environment_value:
        return _required_text(environment_value, "ISAACLAB_VERSION")
    version = _discover_package_version("isaaclab", "isaaclab")
    if version is None:
        raise ProvenanceError(
            "cannot determine Isaac Lab version; pass isaaclab_version explicitly or set "
            "ISAACLAB_VERSION inside the Isaac Lab environment"
        )
    return _required_text(version, "Isaac Lab version")


def discover_rsl_rl_version() -> str:
    """Discover and enforce the RSL-RL release used by the policy ABI."""

    environment_value = os.environ.get("RSL_RL_VERSION")
    version = (
        _required_text(environment_value, "RSL_RL_VERSION")
        if environment_value
        else _discover_package_version("rsl-rl-lib", "rsl_rl")
    )
    if version is None:
        raise ProvenanceError(
            "cannot determine RSL-RL version; install the required release or set "
            "RSL_RL_VERSION"
        )
    normalized = _required_text(version, "RSL-RL version").removeprefix("v")
    if normalized != RSL_RL_VERSION:
        raise ProvenanceError(
            f"installed RSL-RL version is {normalized}; required {RSL_RL_VERSION}"
        )
    return normalized


def torch_cuda_versions() -> tuple[str, str]:
    torch = _require_torch()
    torch_version = _required_text(torch.__version__, "torch.__version__")
    cuda_value = getattr(torch.version, "cuda", None)
    cuda_version = CUDA_NOT_AVAILABLE if cuda_value is None else _required_text(cuda_value, "torch.version.cuda")
    return torch_version, cuda_version


def collect_checkpoint_metadata(
    *,
    isaac_sim_version: str | None = None,
    isaaclab_version: str | None = None,
    rsl_rl_version: str | None = None,
    joint_order: Sequence[str] = FIXED_G1_JOINT_ORDER,
) -> CheckpointMetadata:
    """Collect the runtime versions and joint order needed to reload a policy."""

    torch_version, cuda_version = torch_cuda_versions()
    resolved_isaac_sim = (
        discover_isaac_sim_version()
        if isaac_sim_version is None
        else _required_text(isaac_sim_version, "isaac_sim_version")
    )
    resolved_isaaclab_version = (
        discover_isaaclab_version()
        if isaaclab_version is None
        else _required_text(isaaclab_version, "isaaclab_version")
    )
    resolved_rsl_rl_version = (
        discover_rsl_rl_version()
        if rsl_rl_version is None
        else _required_text(rsl_rl_version, "rsl_rl_version").removeprefix("v")
    )
    return CheckpointMetadata(
        isaac_sim_version=resolved_isaac_sim,
        isaaclab_version=resolved_isaaclab_version,
        rsl_rl_version=resolved_rsl_rl_version,
        pytorch_version=torch_version,
        cuda_version=cuda_version,
        joint_order=tuple(joint_order),
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
    joint_order: Sequence[str] = FIXED_G1_JOINT_ORDER,
    isaac_sim_version: str | None = None,
    isaaclab_version: str | None = None,
    pytorch_version: str | None = None,
    cuda_version: str | None = None,
    validate_torch_runtime: bool = False,
) -> CheckpointMetadata:
    """Validate checkpoint metadata against explicit runtime authorities.

    Structural validation, the pinned RSL-RL release, and the fixed 29-joint
    order are always enforced.  Runtime versions are compared when passed
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
                "isaaclab_version",
                "rsl_rl_version",
                "pytorch_version",
                "cuda_version",
                "joint_order",
            )
            differing = [
                field for field in fields if getattr(parsed, field) != getattr(expected_parsed, field)
            ]
            raise ProvenanceError(f"checkpoint metadata differs from expected fields: {differing}")

    if isaac_sim_version is not None:
        _assert_equal(
            "Isaac Sim version",
            parsed.isaac_sim_version,
            _required_text(isaac_sim_version, "isaac_sim_version"),
        )
    if isaaclab_version is not None:
        _assert_equal(
            "Isaac Lab version",
            parsed.isaaclab_version,
            _required_text(isaaclab_version, "isaaclab_version"),
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


__all__ = [
    "CHECKPOINT_METADATA_KEY",
    "CUDA_NOT_AVAILABLE",
    "CheckpointMetadata",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceDependencyError",
    "ProvenanceError",
    "RSL_RL_VERSION",
    "atomic_torch_save",
    "attach_checkpoint_metadata",
    "collect_checkpoint_metadata",
    "discover_isaac_sim_version",
    "discover_isaaclab_version",
    "discover_rsl_rl_version",
    "extract_checkpoint_metadata",
    "load_checkpoint_with_validation",
    "save_checkpoint_atomic",
    "torch_cuda_versions",
    "validate_checkpoint",
    "validate_checkpoint_metadata",
]
