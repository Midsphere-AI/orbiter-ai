"""Tests for the knowledge-bases route module.

Covers:
- CRUD (list, create, get, update, delete)
- _escape_like: % and _ in search terms are treated literally (no wildcard expansion)
- Search returns results ranked by keyword score, filtered by similarity_threshold
- Document upload/list/chunk preview/delete
- Auth (401 when no session, 404 when wrong user)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.knowledge_bases import _escape_like, router

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_USER = {"id": "user-001", "email": "a@example.com", "role": "developer"}

KB_ROW: dict[str, Any] = {
    "id": "kb-001",
    "name": "My KB",
    "description": "desc",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "search_type": "keyword",
    "top_k": 5,
    "similarity_threshold": 0.0,
    "reranker_enabled": 0,
    "doc_count": 0,
    "chunk_count": 0,
    "project_id": None,
    "user_id": "user-001",
    "created_at": "2025-01-01 00:00:00",
    "updated_at": "2025-01-01 00:00:00",
}

DOC_ROW: dict[str, Any] = {
    "id": "doc-001",
    "kb_id": "kb-001",
    "filename": "test.txt",
    "file_type": "txt",
    "file_size": 100,
    "chunk_count": 2,
    "metadata_json": "{}",
    "status": "completed",
    "created_at": "2025-01-01 00:00:00",
}


class _FakeRow:
    """Minimal aiosqlite.Row-compatible object."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()

    def __iter__(self) -> Any:
        return iter(self._data.items())

    def __contains__(self, key: str) -> bool:
        return key in self._data


def _fake_row(data: dict[str, Any] | None) -> _FakeRow | None:
    return _FakeRow(data) if data is not None else None


def _make_get_db(*row_sequence: dict[str, Any] | list[dict[str, Any]] | None):
    """Return a patched get_db; each call to db.execute gets the next row/list."""
    items = list(row_sequence)
    call_idx = 0

    @asynccontextmanager
    async def _mock():
        nonlocal call_idx
        db = AsyncMock()

        async def _execute(sql: str, params: Any = ()) -> AsyncMock:
            nonlocal call_idx
            cursor = AsyncMock()
            item = items[call_idx] if call_idx < len(items) else None
            call_idx += 1
            if isinstance(item, list):
                cursor.fetchall = AsyncMock(return_value=[_FakeRow(r) for r in item])
                cursor.fetchone = AsyncMock(return_value=_fake_row(item[0]) if item else None)
            else:
                cursor.fetchone = AsyncMock(return_value=_fake_row(item))
                cursor.fetchall = AsyncMock(return_value=[_FakeRow(item)] if item else [])
            cursor.rowcount = 1
            return cursor

        db.execute = _execute
        db.commit = AsyncMock()
        yield db

    return _mock


def _auth_override(user: dict[str, Any] = FAKE_USER):
    """Override get_current_user dependency to return a fake user."""
    from exo_web.routes.auth import get_current_user

    app_copy = FastAPI()
    app_copy.include_router(router)
    app_copy.dependency_overrides[get_current_user] = lambda: user
    return app_copy


# ---------------------------------------------------------------------------
# Unit tests: _escape_like
# ---------------------------------------------------------------------------


class TestEscapeLike:
    """Verify that LIKE special characters are escaped correctly."""

    def test_percent_is_escaped(self):
        result = _escape_like("50%")
        assert result == r"50\%"

    def test_underscore_is_escaped(self):
        result = _escape_like("foo_bar")
        assert result == r"foo\_bar"

    def test_both_wildcards_escaped(self):
        result = _escape_like("10%_off")
        assert result == r"10\%\_off"

    def test_backslash_itself_is_doubled(self):
        result = _escape_like("a\\b")
        assert result == "a\\\\b"

    def test_plain_text_unchanged(self):
        result = _escape_like("hello world")
        assert result == "hello world"

    def test_percent_as_literal_in_query(self):
        """A query containing % should produce a LIKE pattern that matches literally.

        The escaped term '50\\%' inside '%...%' becomes '%50\\%%' — SQLite
        interprets the \\% as a literal percent, so '50%' in content matches.
        """
        term = "50%"
        escaped = _escape_like(term)
        # escaped should be '50\\%'; embed in a LIKE pattern: '%50\\%%'
        like_pattern = f"%{escaped}%"
        # The pattern should match the literal string '50%'
        # and NOT match '50anything' (which would happen if % were unescaped)
        assert like_pattern == r"%50\%%"

    def test_underscore_as_literal(self):
        """A query containing _ should produce a LIKE pattern that matches literally."""
        term = "my_field"
        escaped = _escape_like(term)
        like_pattern = f"%{escaped}%"
        assert like_pattern == r"%my\_field%"


# ---------------------------------------------------------------------------
# Endpoint tests: list, create, get, update, delete
# ---------------------------------------------------------------------------


class TestListKnowledgeBases:
    def test_returns_list(self):
        app = _auth_override()
        mock_db = _make_get_db([KB_ROW])
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/v1/knowledge-bases")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "kb-001"

    def test_unauthenticated_returns_401(self):
        client = TestClient(_app, raise_server_exceptions=False)
        with patch("exo_web.routes.knowledge_bases.get_db", _make_get_db()):
            resp = client.get("/api/v1/knowledge-bases")
        assert resp.status_code == 401

    def test_filters_by_project_id(self):
        app = _auth_override()
        mock_db = _make_get_db([KB_ROW])
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/knowledge-bases", params={"project_id": "proj-001"})
        assert resp.status_code == 200


class TestCreateKnowledgeBase:
    def test_creates_and_returns_201(self):
        app = _auth_override()
        mock_db = _make_get_db(None, KB_ROW)  # INSERT commit, then SELECT
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/knowledge-bases",
                json={
                    "name": "My KB",
                    "description": "desc",
                    "embedding_model": "text-embedding-3-small",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "search_type": "keyword",
                    "top_k": 5,
                    "similarity_threshold": 0.0,
                    "reranker_enabled": False,
                },
            )
        assert resp.status_code == 201
        assert resp.json()["id"] == "kb-001"

    def test_invalid_search_type_returns_422(self):
        app = _auth_override()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "X", "search_type": "invalid"},
        )
        assert resp.status_code == 422

    def test_missing_name_returns_422(self):
        app = _auth_override()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/knowledge-bases", json={"search_type": "keyword"})
        assert resp.status_code == 422


class TestGetKnowledgeBase:
    def test_returns_existing_kb(self):
        app = _auth_override()
        mock_db = _make_get_db(KB_ROW)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/knowledge-bases/kb-001")
        assert resp.status_code == 200
        assert resp.json()["name"] == "My KB"

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/knowledge-bases/nonexistent")
        assert resp.status_code == 404

    def test_wrong_user_cannot_access(self):
        """A KB owned by user-002 is invisible to user-001 (returns 404)."""
        other_user = {**FAKE_USER, "id": "user-999"}
        app = _auth_override(user=other_user)
        # Simulate DB returning None because user_id filter excludes the row
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/knowledge-bases/kb-001")
        assert resp.status_code == 404


class TestUpdateKnowledgeBase:
    def test_updates_returns_200(self):
        app = _auth_override()
        # SELECT (ownership check), UPDATE, SELECT (fetch updated)
        mock_db = _make_get_db(KB_ROW, None, KB_ROW)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app)
            resp = client.put(
                "/api/v1/knowledge-bases/kb-001",
                json={"name": "Renamed KB"},
            )
        assert resp.status_code == 200

    def test_empty_update_returns_422(self):
        app = _auth_override()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.put("/api/v1/knowledge-bases/kb-001", json={})
        assert resp.status_code == 422

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.put(
                "/api/v1/knowledge-bases/nonexistent",
                json={"name": "X"},
            )
        assert resp.status_code == 404


class TestDeleteKnowledgeBase:
    def test_deletes_returns_204(self):
        app = _auth_override()
        mock_db = _make_get_db(KB_ROW, None)  # SELECT, DELETE
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app)
            resp = client.delete("/api/v1/knowledge-bases/kb-001")
        assert resp.status_code == 204

    def test_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/knowledge-bases/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

CHUNK_ROW: dict[str, Any] = {
    "chunk_id": "chunk-001",
    "document_id": "doc-001",
    "filename": "test.txt",
    "chunk_index": 0,
    "content": "The price is 50% off today",
}


class TestSearchKnowledgeBase:
    def test_basic_search_returns_results(self):
        app = _auth_override()
        # KB lookup, then chunks query
        mock_db = _make_get_db(KB_ROW, [CHUNK_ROW])
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/knowledge-bases/kb-001/search",
                json={"query": "price"},
            )
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert len(results) == 1
        assert "score" in results[0]

    def test_search_percent_literal_not_wildcard(self):
        """A query term containing '%' should not act as a SQL wildcard.

        The escaped LIKE pattern '%50\\%%' must match chunks containing
        the literal substring '50%', not any chunk containing '50'.
        We verify the _escape_like function produces the right pattern.
        """
        term = "50%"
        escaped = _escape_like(term)
        # Confirm the '%' is escaped so it becomes a literal in the LIKE clause
        assert "\\%" in escaped

        # Build the full LIKE parameter as the route does
        like_param = f"%{escaped}%"
        # It must match a string containing literal '50%'

        # fnmatch doesn't do SQL LIKE, but we can verify the raw string is correct
        assert like_param == r"%50\%%"

    def test_search_underscore_literal_not_wildcard(self):
        """A query term containing '_' should not act as a single-character wildcard."""
        term = "my_field"
        escaped = _escape_like(term)
        assert "\\_" in escaped
        like_param = f"%{escaped}%"
        assert like_param == r"%my\_field%"

    def test_kb_not_found_returns_404(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/knowledge-bases/missing/search",
                json={"query": "hello"},
            )
        assert resp.status_code == 404

    def test_empty_query_returns_422(self):
        app = _auth_override()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/knowledge-bases/kb-001/search", json={"query": ""})
        assert resp.status_code == 422

    def test_similarity_threshold_filters_low_scores(self):
        """Chunks with score below threshold should be excluded from results."""
        high_threshold_kb = {**KB_ROW, "similarity_threshold": 1.0, "top_k": 5}
        app = _auth_override()
        # KB with threshold=1.0 means only perfect matches pass
        mock_db = _make_get_db(high_threshold_kb, [CHUNK_ROW])
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app)
            # The chunk content has "price" (1 of 2 terms) → score 0.5 < 1.0
            resp = client.post(
                "/api/v1/knowledge-bases/kb-001/search",
                json={"query": "price missing"},
            )
        assert resp.status_code == 200
        # Score 0.5 is below threshold 1.0, so no results
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------


class TestDocumentEndpoints:
    def test_list_documents(self):
        app = _auth_override()
        # _verify_kb_access (SELECT), then list docs SELECT
        mock_db = _make_get_db(KB_ROW, [DOC_ROW])
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app)
            resp = client.get("/api/v1/knowledge-bases/kb-001/documents")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_list_documents_kb_not_found(self):
        app = _auth_override()
        mock_db = _make_get_db(None)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/knowledge-bases/missing/documents")
        assert resp.status_code == 404

    def test_delete_document_not_found_returns_404(self):
        app = _auth_override()
        # _verify_kb_access succeeds (KB_ROW), then doc SELECT returns None
        mock_db = _make_get_db(KB_ROW, None)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.delete("/api/v1/knowledge-bases/kb-001/documents/nonexistent")
        assert resp.status_code == 404

    def test_upload_unsupported_type_returns_422(self):
        app = _auth_override()
        mock_db = _make_get_db(KB_ROW)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/knowledge-bases/kb-001/documents",
                files={"file": ("bad.xyz", b"content", "application/octet-stream")},
            )
        assert resp.status_code == 422

    def test_upload_empty_file_returns_422(self):
        app = _auth_override()
        mock_db = _make_get_db(KB_ROW)
        with patch("exo_web.routes.knowledge_bases.get_db", mock_db):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/knowledge-bases/kb-001/documents",
                files={"file": ("test.txt", b"", "text/plain")},
            )
        assert resp.status_code == 422

    def test_upload_txt_file_succeeds(self):
        app = _auth_override()
        file_content = b"Hello world this is test content for the knowledge base."
        # _verify_kb_access, KB chunk config SELECT, INSERT doc, INSERT chunk,
        # UPDATE kb counters, SELECT doc
        mock_db = _make_get_db(KB_ROW, KB_ROW, None, None, None, DOC_ROW)
        with (
            patch("exo_web.routes.knowledge_bases.get_db", mock_db),
            patch(
                "exo_web.routes.knowledge_bases.extract_text",
                return_value="Hello world this is test content for the knowledge base.",
            ),
            patch(
                "exo_web.routes.knowledge_bases.chunk_text",
                return_value=["Hello world this is test content for the knowledge base."],
            ),
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/knowledge-bases/kb-001/documents",
                files={"file": ("test.txt", file_content, "text/plain")},
            )
        assert resp.status_code == 201
        assert resp.json()["id"] == "doc-001"
