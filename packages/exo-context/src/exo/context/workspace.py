"""Workspace — persistent artifact storage with versioning and observer notifications.

Provides a local-filesystem-backed artifact store where agents can read/write
files during execution.  Supports artifact type classification, version history
with revert, and an observer callback pattern for create/update/delete events.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable, Coroutine
from enum import StrEnum
from pathlib import Path
from typing import Any

from exo.types import ExoError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class WorkspaceError(ExoError):
    """Raised for workspace operation errors."""


# ── Artifact type ─────────────────────────────────────────────────────


class ArtifactType(StrEnum):
    """Classification of stored artifacts."""

    CODE = "code"
    CSV = "csv"
    IMAGE = "image"
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


# ── Artifact version entry ───────────────────────────────────────────


class ArtifactVersion:
    """Immutable snapshot of an artifact at a point in time."""

    __slots__ = ("_content", "_timestamp")

    def __init__(self, content: str, timestamp: float | None = None) -> None:
        self._content = content
        self._timestamp = timestamp or time.time()

    @property
    def content(self) -> str:
        return self._content

    @property
    def timestamp(self) -> float:
        return self._timestamp

    def __repr__(self) -> str:
        return f"ArtifactVersion(len={len(self._content)}, ts={self._timestamp:.0f})"


# ── Artifact ─────────────────────────────────────────────────────────


class Artifact:
    """A named artifact with type, content, and version history."""

    __slots__ = ("_artifact_type", "_name", "_versions")

    def __init__(
        self, name: str, content: str, artifact_type: ArtifactType = ArtifactType.TEXT
    ) -> None:
        self._name = name
        self._artifact_type = artifact_type
        self._versions: list[ArtifactVersion] = [ArtifactVersion(content)]

    @property
    def name(self) -> str:
        return self._name

    @property
    def artifact_type(self) -> ArtifactType:
        return self._artifact_type

    @property
    def content(self) -> str:
        """Current (latest) content."""
        return self._versions[-1].content

    @property
    def version_count(self) -> int:
        return len(self._versions)

    @property
    def versions(self) -> list[ArtifactVersion]:
        """All versions (oldest first)."""
        return list(self._versions)

    def _add_version(self, content: str) -> None:
        self._versions.append(ArtifactVersion(content))

    def __repr__(self) -> str:
        return (
            f"Artifact(name={self._name!r}, type={self._artifact_type.value!r}, "
            f"versions={self.version_count})"
        )


# ── Observer callback types ──────────────────────────────────────────

ObserverCallback = Callable[[str, Artifact], Coroutine[Any, Any, None]]
"""async callback(event_name, artifact)."""


# ── Workspace ────────────────────────────────────────────────────────


class Workspace:
    """Persistent artifact storage with versioning and observer notifications.

    Parameters
    ----------
    workspace_id:
        Unique workspace identifier.
    storage_path:
        Root directory for artifact files.  Created on first write if needed.
    """

    __slots__ = ("_artifacts", "_knowledge_store", "_observers", "_storage_path", "_workspace_id")

    def __init__(
        self,
        workspace_id: str,
        *,
        storage_path: str | Path | None = None,
        knowledge_store: Any | None = None,
    ) -> None:
        if not workspace_id:
            msg = "workspace_id is required and must be non-empty"
            raise WorkspaceError(msg)
        self._workspace_id = workspace_id
        self._storage_path = Path(storage_path) if storage_path else None
        self._artifacts: dict[str, Artifact] = {}
        self._observers: dict[str, list[ObserverCallback]] = {}
        self._knowledge_store = knowledge_store

    # ── Properties ────────────────────────────────────────────────────

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    @property
    def storage_path(self) -> Path | None:
        return self._storage_path

    @property
    def knowledge_store(self) -> Any | None:
        """Attached :class:`KnowledgeStore` for auto-indexing, or ``None``."""
        return self._knowledge_store

    # ── Observer management ───────────────────────────────────────────

    def on(self, event: str, callback: ObserverCallback) -> Workspace:
        """Register an observer callback for an event (on_create, on_update, on_delete).

        Returns self for chaining.
        """
        self._observers.setdefault(event, []).append(callback)
        return self

    async def _notify(self, event: str, artifact: Artifact) -> None:
        for cb in self._observers.get(event, []):
            try:
                await cb(event, artifact)
            except Exception:
                logger.error(
                    "observer callback failed for event %r on artifact %r",
                    event,
                    artifact.name,
                    exc_info=True,
                )
                raise

    # ── CRUD ─────────────────────────────────────────────────────────

    async def write(
        self,
        name: str,
        content: str,
        *,
        artifact_type: ArtifactType = ArtifactType.TEXT,
    ) -> Artifact:
        """Write or update an artifact.

        If *name* exists, a new version is appended and ``on_update`` fires.
        Otherwise a new artifact is created and ``on_create`` fires.
        Persists to filesystem if *storage_path* is set.
        """
        if not name:
            msg = "artifact name is required"
            raise WorkspaceError(msg)

        existing = self._artifacts.get(name)
        if existing is not None:
            existing._add_version(content)
            self._persist(existing)
            self._index_artifact(existing)
            logger.debug("Artifact stored: id=%s size=%d bytes", name, len(content))
            await self._notify("on_update", existing)
            return existing

        artifact = Artifact(name, content, artifact_type)
        self._artifacts[name] = artifact
        self._persist(artifact)
        self._index_artifact(artifact)
        logger.debug("Artifact stored: id=%s size=%d bytes", name, len(content))
        await self._notify("on_create", artifact)
        return artifact

    def read(self, name: str) -> str | None:
        """Read current content of an artifact by name.  Returns ``None`` if missing."""
        artifact = self._artifacts.get(name)
        if artifact is not None:
            logger.debug("Artifact retrieved: id=%s", name)
        return artifact.content if artifact else None

    def get(self, name: str) -> Artifact | None:
        """Get the full Artifact object.  Returns ``None`` if missing."""
        return self._artifacts.get(name)

    def list(self, *, artifact_type: ArtifactType | None = None) -> list[Artifact]:
        """List all artifacts, optionally filtered by type."""
        arts = list(self._artifacts.values())
        if artifact_type is not None:
            arts = [a for a in arts if a.artifact_type == artifact_type]
        return arts

    async def delete(self, name: str) -> bool:
        """Delete an artifact by name.  Returns ``True`` if deleted, ``False`` if missing."""
        artifact = self._artifacts.pop(name, None)
        if artifact is None:
            return False
        self._remove_persisted(artifact)
        self._deindex_artifact(artifact)
        logger.debug("deleted artifact %r", name)
        await self._notify("on_delete", artifact)
        return True

    # ── Versioning ───────────────────────────────────────────────────

    def version_history(self, name: str) -> list[ArtifactVersion]:
        """Return the version history of an artifact.  Empty list if missing."""
        artifact = self._artifacts.get(name)
        return artifact.versions if artifact else []

    def revert_to_version(self, name: str, version: int) -> Artifact:
        """Revert an artifact to a previous version (0-indexed).

        Creates a new version whose content matches the target version.
        Raises :class:`WorkspaceError` if name or version is invalid.
        """
        artifact = self._artifacts.get(name)
        if artifact is None:
            msg = f"artifact {name!r} not found"
            raise WorkspaceError(msg)
        if version < 0 or version >= artifact.version_count:
            msg = f"version {version} out of range (0..{artifact.version_count - 1})"
            logger.warning("revert failed for artifact %r: %s", name, msg)
            raise WorkspaceError(msg)
        old_content = artifact.versions[version].content
        artifact._add_version(old_content)
        self._persist(artifact)
        logger.debug("reverted artifact %r to version %d", name, version)
        return artifact

    # ── Knowledge store integration ────────────────────────────────

    def _index_artifact(self, artifact: Artifact) -> None:
        """Auto-index artifact content in the knowledge store if attached."""
        if self._knowledge_store is not None:
            self._knowledge_store.add(artifact.name, artifact.content)

    def _deindex_artifact(self, artifact: Artifact) -> None:
        """Remove artifact from the knowledge store index if attached."""
        if self._knowledge_store is not None:
            self._knowledge_store.remove(artifact.name)

    # ── Filesystem persistence ───────────────────────────────────────

    def _persist(self, artifact: Artifact) -> None:
        """Write artifact content to filesystem if storage_path is set."""
        if self._storage_path is None:
            return
        # Validate artifact name to prevent path traversal
        resolved = (self._storage_path / artifact.name).resolve()
        if not resolved.is_relative_to(self._storage_path.resolve()):
            logger.error(
                "path traversal blocked in _persist: artifact name %r escapes storage_path",
                artifact.name,
            )
            raise WorkspaceError(
                f"artifact name {artifact.name!r} resolves outside storage directory"
            )
        artifact_dir = resolved
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "content").write_text(artifact.content, encoding="utf-8")
        meta = {
            "name": artifact.name,
            "artifact_type": artifact.artifact_type.value,
            "version_count": artifact.version_count,
        }
        (artifact_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def _remove_persisted(self, artifact: Artifact) -> None:
        """Remove artifact directory from filesystem if storage_path is set."""
        if self._storage_path is None:
            return
        # Validate artifact name to prevent path traversal
        resolved = (self._storage_path / artifact.name).resolve()
        if not resolved.is_relative_to(self._storage_path.resolve()):
            logger.error(
                "path traversal blocked in _remove_persisted: artifact name %r escapes storage_path",
                artifact.name,
            )
            raise WorkspaceError(
                f"artifact name {artifact.name!r} resolves outside storage directory"
            )
        if resolved.exists():
            shutil.rmtree(resolved)

    # ── Representation ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._artifacts)

    def __repr__(self) -> str:
        return f"Workspace(id={self._workspace_id!r}, artifacts={len(self)})"
