"""Tests for the workspace export/import route module.

Covers:
- POST /api/v1/settings/export — triggers export and returns metadata
- GET /api/v1/settings/export/{export_id}/download — IDOR fix: user cannot
  download another user's export (returns 404 in both missing and wrong-owner cases)
- POST /api/v1/settings/import — validates ZIP content, rejects non-zip/empty/invalid
- Auth: unauthenticated requests return 401
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.workspace_export import router

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_USER = {"id": "user-001", "email": "a@example.com", "role": "developer"}
OTHER_USER_ID = "user-999"


def _auth_override(user: dict[str, Any] = FAKE_USER) -> FastAPI:
    from exo_web.routes.auth import get_current_user

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: user
    return app


def _make_zip(manifest: dict[str, Any] | None = None, include_agents: bool = False) -> bytes:
    """Build a minimal valid ZIP suitable for download/import tests."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if manifest is not None:
            zf.writestr("manifest.json", json.dumps(manifest))
        if include_agents:
            zf.writestr("agents.json", json.dumps([]))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /api/v1/settings/export
# ---------------------------------------------------------------------------


class TestTriggerExport:
    def test_returns_export_metadata(self, tmp_path: Path):
        app = _auth_override()
        fake_result = {
            "export_id": "abc123",
            "filename": "exo-export-20250101T000000-abc123.zip",
            "path": str(tmp_path / "exo-export-20250101T000000-abc123.zip"),
            "entity_counts": {"agents": 0},
            "exported_at": "2025-01-01T00:00:00+00:00",
        }

        with (
            patch(
                "exo_web.routes.workspace_export.create_export", AsyncMock(return_value=fake_result)
            ),
            patch("exo_web.routes.workspace_export.audit_log", AsyncMock()),
        ):
            client = TestClient(app)
            resp = client.post("/api/v1/settings/export")

        assert resp.status_code == 200
        data = resp.json()
        assert data["export_id"] == "abc123"
        assert "filename" in data
        assert "entity_counts" in data

    def test_unauthenticated_returns_401(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post("/api/v1/settings/export")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/settings/export/{export_id}/download
# ---------------------------------------------------------------------------


class TestDownloadExport:
    def test_download_own_export_succeeds(self, tmp_path: Path):
        """Owner of the export ZIP can download it."""
        manifest = {
            "export_id": "abc123",
            "user_id": "user-001",
            "exported_at": "2025-01-01T00:00:00+00:00",
        }
        zip_bytes = _make_zip(manifest)
        zip_path = tmp_path / "exo-export-20250101T000000-abc123.zip"
        zip_path.write_bytes(zip_bytes)

        app = _auth_override()
        with patch("exo_web.routes.workspace_export.get_export_path", return_value=zip_path):
            client = TestClient(app)
            resp = client.get("/api/v1/settings/export/abc123/download")

        assert resp.status_code == 200
        assert resp.headers["content-type"] in (
            "application/zip",
            "application/zip; charset=utf-8",
        )

    def test_different_user_cannot_download_returns_404(self, tmp_path: Path):
        """IDOR fix: a different user gets 404, not 200 or 403."""
        manifest = {
            "export_id": "abc123",
            "user_id": "user-001",  # owned by user-001
            "exported_at": "2025-01-01T00:00:00+00:00",
        }
        zip_bytes = _make_zip(manifest)
        zip_path = tmp_path / "exo-export-20250101T000000-abc123.zip"
        zip_path.write_bytes(zip_bytes)

        # Authenticate as a different user
        app = _auth_override(user={**FAKE_USER, "id": OTHER_USER_ID})
        with patch("exo_web.routes.workspace_export.get_export_path", return_value=zip_path):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/settings/export/abc123/download")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Export not found"

    def test_missing_export_returns_404(self):
        """When no file exists, the endpoint returns 404."""
        app = _auth_override()
        with patch("exo_web.routes.workspace_export.get_export_path", return_value=None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/settings/export/missing/download")

        assert resp.status_code == 404

    def test_bad_zip_returns_404(self, tmp_path: Path):
        """A corrupt ZIP file results in 404 rather than 500."""
        bad_path = tmp_path / "corrupt.zip"
        bad_path.write_bytes(b"not-a-zip")

        app = _auth_override()
        with patch("exo_web.routes.workspace_export.get_export_path", return_value=bad_path):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/settings/export/corrupt/download")

        assert resp.status_code == 404

    def test_zip_without_manifest_returns_404(self, tmp_path: Path):
        """A ZIP with no manifest.json is rejected with 404 (ownership unverifiable)."""
        zip_bytes = _make_zip(manifest=None)  # no manifest
        zip_path = tmp_path / "no-manifest.zip"
        zip_path.write_bytes(zip_bytes)

        app = _auth_override()
        with patch("exo_web.routes.workspace_export.get_export_path", return_value=zip_path):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/settings/export/no-manifest/download")

        assert resp.status_code == 404

    def test_unauthenticated_returns_401(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.get("/api/v1/settings/export/abc/download")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/settings/import
# ---------------------------------------------------------------------------


class TestTriggerImport:
    def test_import_valid_zip_succeeds(self):
        manifest = {"export_id": "abc", "user_id": "user-001", "exported_at": "2025-01-01T00:00:00"}
        zip_bytes = _make_zip(manifest, include_agents=True)

        app = _auth_override()
        with (
            patch(
                "exo_web.routes.workspace_export.import_workspace",
                AsyncMock(return_value={"imported": {"agents": 0}}),
            ),
            patch("exo_web.routes.workspace_export.audit_log", AsyncMock()),
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/settings/import",
                files={"file": ("workspace.zip", zip_bytes, "application/zip")},
            )

        assert resp.status_code == 200
        assert "imported" in resp.json()

    def test_non_zip_file_returns_422(self):
        app = _auth_override()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/settings/import",
            files={"file": ("data.json", b'{"key": "val"}', "application/json")},
        )
        assert resp.status_code == 422
        assert "zip" in resp.json()["detail"].lower()

    def test_empty_zip_file_returns_422(self):
        app = _auth_override()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/settings/import",
            files={"file": ("empty.zip", b"", "application/zip")},
        )
        assert resp.status_code == 422

    def test_invalid_zip_content_returns_422(self):
        """A .zip filename but invalid binary content triggers ValueError from import_workspace."""
        app = _auth_override()
        with patch(
            "exo_web.routes.workspace_export.import_workspace",
            AsyncMock(side_effect=ValueError("Invalid ZIP file")),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/settings/import",
                files={"file": ("bad.zip", b"NOTAZIP", "application/zip")},
            )
        assert resp.status_code == 422
        assert "Invalid ZIP" in resp.json()["detail"]

    def test_unauthenticated_returns_401(self):
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/settings/import",
            files={"file": ("data.zip", b"PK", "application/zip")},
        )
        assert resp.status_code == 401
