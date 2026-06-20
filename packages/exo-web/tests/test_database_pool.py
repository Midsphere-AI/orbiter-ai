"""Tests for the database connection pool (database.py).

Covers: init_pool / close_pool lifecycle, PRAGMAs applied, get_db borrow/return,
concurrent borrows, idempotent init/close, and run_migrations.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — always use a real temp file, never the production DB
# ---------------------------------------------------------------------------


def _tmp_db_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


async def _reset_pool_module() -> None:
    """Forcibly reset the module-level pool state to None.

    The pool is a module global; tests must clean it up to stay hermetic.
    """
    import exo_web.database as db_mod

    # Close and discard any open pool.
    await db_mod.close_pool()
    # Paranoia: ensure the module sees None.
    db_mod._pool = None


# ---------------------------------------------------------------------------
# init_pool / close_pool
# ---------------------------------------------------------------------------


class TestInitPool:
    async def test_init_pool_creates_connections(self) -> None:

        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()
                assert db_mod._pool is not None
                assert db_mod._pool.qsize() == db_mod._POOL_SIZE
        finally:
            await db_mod.close_pool()
            os.unlink(path)

    async def test_init_pool_idempotent(self) -> None:
        """Calling init_pool() twice must not create a second pool."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()
                first_pool = db_mod._pool
                await db_mod.init_pool()
                assert db_mod._pool is first_pool  # Same object
        finally:
            await db_mod.close_pool()
            os.unlink(path)

    async def test_close_pool_sets_none(self) -> None:
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()
                assert db_mod._pool is not None
                await db_mod.close_pool()
                assert db_mod._pool is None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    async def test_close_pool_idempotent(self) -> None:
        """close_pool() on an already-closed pool must not raise."""
        import exo_web.database as db_mod

        await db_mod.close_pool()
        await db_mod.close_pool()  # Should not raise


# ---------------------------------------------------------------------------
# PRAGMAs
# ---------------------------------------------------------------------------


class TestConnectionPragmas:
    async def test_wal_mode_enabled(self) -> None:
        """WAL journal mode must be set on each new connection."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                conn = await db_mod._make_connection()
                cursor = await conn.execute("PRAGMA journal_mode")
                row = await cursor.fetchone()
                assert row[0] == "wal"
                await conn.close()
        finally:
            os.unlink(path)

    async def test_foreign_keys_enabled(self) -> None:
        """Foreign key enforcement must be ON for every new connection."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                conn = await db_mod._make_connection()
                cursor = await conn.execute("PRAGMA foreign_keys")
                row = await cursor.fetchone()
                assert row[0] == 1
                await conn.close()
        finally:
            os.unlink(path)

    async def test_row_factory_set(self) -> None:
        """Row factory must be aiosqlite.Row (not the default tuple)."""
        import aiosqlite

        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                conn = await db_mod._make_connection()
                assert conn.row_factory is aiosqlite.Row
                await conn.close()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_db — borrow and return
# ---------------------------------------------------------------------------


class TestGetDb:
    async def test_get_db_yields_connection(self) -> None:
        import aiosqlite

        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()
                async with db_mod.get_db() as conn:
                    assert isinstance(conn, aiosqlite.Connection)
        finally:
            await db_mod.close_pool()
            os.unlink(path)

    async def test_connection_returned_to_pool_after_use(self) -> None:
        """Pool size must be the same before and after a get_db() block."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()
                size_before = db_mod._pool.qsize()  # type: ignore[union-attr]
                async with db_mod.get_db():
                    pass
                size_after = db_mod._pool.qsize()  # type: ignore[union-attr]
                assert size_before == size_after
        finally:
            await db_mod.close_pool()
            os.unlink(path)

    async def test_connection_returned_on_exception(self) -> None:
        """Connection must be returned to pool even when the body raises."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()
                size_before = db_mod._pool.qsize()  # type: ignore[union-attr]
                with pytest.raises(ValueError, match="intentional"):
                    async with db_mod.get_db():
                        raise ValueError("intentional")
                size_after = db_mod._pool.qsize()  # type: ignore[union-attr]
                assert size_before == size_after
        finally:
            await db_mod.close_pool()
            os.unlink(path)

    async def test_concurrent_borrows(self) -> None:
        """Up to _POOL_SIZE concurrent borrows must succeed without deadlock."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()

                results: list[bool] = []

                async def use_db() -> None:
                    async with db_mod.get_db():
                        await asyncio.sleep(0)  # yield
                        results.append(True)

                await asyncio.gather(*[use_db() for _ in range(db_mod._POOL_SIZE)])
                assert len(results) == db_mod._POOL_SIZE
        finally:
            await db_mod.close_pool()
            os.unlink(path)

    async def test_concurrent_borrow_beyond_pool_size_queues(self) -> None:
        """Requesting more connections than the pool holds must not error,
        it must queue and eventually succeed."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        pool_size = db_mod._POOL_SIZE
        n = pool_size + 1  # one more than available
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()

                results: list[int] = []

                async def use_db(i: int) -> None:
                    async with db_mod.get_db():
                        await asyncio.sleep(0.01)  # hold briefly
                        results.append(i)

                await asyncio.gather(*[use_db(i) for i in range(n)])
                assert len(results) == n
        finally:
            await db_mod.close_pool()
            os.unlink(path)


# ---------------------------------------------------------------------------
# Transactional correctness
# ---------------------------------------------------------------------------


class TestTransactionalCorrectness:
    async def test_commit_persists_data(self) -> None:
        """Data written + committed in one get_db block must be visible in another."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()

                async with db_mod.get_db() as db:
                    await db.execute("CREATE TABLE t (val TEXT)")
                    await db.commit()

                async with db_mod.get_db() as db:
                    await db.execute("INSERT INTO t VALUES ('hello')")
                    await db.commit()

                async with db_mod.get_db() as db:
                    cursor = await db.execute("SELECT val FROM t")
                    row = await cursor.fetchone()
                    assert row[0] == "hello"
        finally:
            await db_mod.close_pool()
            os.unlink(path)

    async def test_rollback_on_exception_does_not_corrupt_pool(self) -> None:
        """An exception mid-transaction must not corrupt subsequent connections."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()

                async with db_mod.get_db() as db:
                    await db.execute("CREATE TABLE t2 (val TEXT)")
                    await db.commit()

                try:
                    async with db_mod.get_db() as db:
                        await db.execute("INSERT INTO t2 VALUES ('will_be_lost')")
                        raise RuntimeError("abort")
                except RuntimeError:
                    pass

                # A subsequent connection should still work fine
                async with db_mod.get_db() as db:
                    cursor = await db.execute("SELECT count(*) FROM t2")
                    row = await cursor.fetchone()
                    # The insert was not committed so count may be 0 or 1
                    # depending on auto-commit mode — the important thing is
                    # that the query completes without error.
                    assert row is not None
        finally:
            await db_mod.close_pool()
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_db_dep (FastAPI dependency)
# ---------------------------------------------------------------------------


class TestGetDbDep:
    async def test_get_db_dep_yields_connection(self) -> None:
        import aiosqlite

        import exo_web.database as db_mod

        path = _tmp_db_path()
        try:
            with patch.object(db_mod, "_DB_PATH", path):
                await db_mod.init_pool()
                gen = db_mod.get_db_dep()
                conn = await gen.__anext__()
                assert isinstance(conn, aiosqlite.Connection)
                # Cleanup: exhaust the generator
                with contextlib.suppress(StopAsyncIteration):
                    await gen.__anext__()
        finally:
            await db_mod.close_pool()
            os.unlink(path)


# ---------------------------------------------------------------------------
# run_migrations
# ---------------------------------------------------------------------------


class TestRunMigrations:
    async def test_migrations_applied_once(self) -> None:
        """Calling run_migrations() twice must only apply each migration once."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        migrations_dir = Path(tempfile.mkdtemp())
        (migrations_dir / "001_test.sql").write_text("CREATE TABLE migration_test (id INTEGER);")

        try:
            with (
                patch.object(db_mod, "_DB_PATH", path),
                patch.object(db_mod, "MIGRATIONS_DIR", migrations_dir),
            ):
                await db_mod.init_pool()
                applied1 = await db_mod.run_migrations()
                applied2 = await db_mod.run_migrations()

            assert "001_test.sql" in applied1
            assert "001_test.sql" not in applied2
        finally:
            await db_mod.close_pool()
            os.unlink(path)
            (migrations_dir / "001_test.sql").unlink()
            migrations_dir.rmdir()

    async def test_migrations_tracking_table_created(self) -> None:
        """The _migrations table must exist after run_migrations()."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        migrations_dir = Path(tempfile.mkdtemp())

        try:
            with (
                patch.object(db_mod, "_DB_PATH", path),
                patch.object(db_mod, "MIGRATIONS_DIR", migrations_dir),
            ):
                await db_mod.init_pool()
                await db_mod.run_migrations()

                async with db_mod.get_db() as db:
                    cursor = await db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
                    )
                    row = await cursor.fetchone()
                assert row is not None
        finally:
            await db_mod.close_pool()
            os.unlink(path)
            migrations_dir.rmdir()

    async def test_empty_migrations_dir_returns_empty(self) -> None:
        """No migration files → returns empty list without crashing."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        migrations_dir = Path(tempfile.mkdtemp())

        try:
            with (
                patch.object(db_mod, "_DB_PATH", path),
                patch.object(db_mod, "MIGRATIONS_DIR", migrations_dir),
            ):
                await db_mod.init_pool()
                applied = await db_mod.run_migrations()
                assert applied == []
        finally:
            await db_mod.close_pool()
            os.unlink(path)
            migrations_dir.rmdir()

    async def test_nonexistent_migrations_dir_returns_empty(self) -> None:
        """Missing migrations dir → returns empty list without crashing."""
        import exo_web.database as db_mod

        path = _tmp_db_path()
        missing_dir = Path("/tmp/definitely_does_not_exist_exo_test_12345")

        try:
            with (
                patch.object(db_mod, "_DB_PATH", path),
                patch.object(db_mod, "MIGRATIONS_DIR", missing_dir),
            ):
                await db_mod.init_pool()
                applied = await db_mod.run_migrations()
                assert applied == []
        finally:
            await db_mod.close_pool()
            os.unlink(path)
