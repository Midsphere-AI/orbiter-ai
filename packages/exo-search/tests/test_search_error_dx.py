"""Tests for exo-search error DX and resilience (Wave C)."""

from __future__ import annotations

import pytest

from exo.search import SearchConfigError, SearchError
from exo.search.config import SearchConfig, SearchModels, SearchSources
from exo.search.pipeline import _normalize_mode
from exo.search.tools.searxng import search_and_collect
from exo.types import ExoError

# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class TestErrorTaxonomy:
    """SearchError and SearchConfigError are ExoError subclasses."""

    def test_search_error_is_exo_error(self) -> None:
        err = SearchError("pipeline failed")
        assert isinstance(err, ExoError)

    def test_search_config_error_is_exo_error(self) -> None:
        err = SearchConfigError("bad config")
        assert isinstance(err, ExoError)

    def test_search_error_carries_context_and_hint(self) -> None:
        err = SearchError(
            "Classifier failed.",
            context={"stage": "classifier", "mode": "balanced", "query": "test"},
            hint="Check credentials.",
        )
        msg = str(err)
        assert "Classifier failed" in msg
        assert "stage" in msg
        assert "Check credentials" in msg

    def test_search_config_error_carries_context_and_hint(self) -> None:
        err = SearchConfigError(
            "Conflicting fields.",
            context={"conflicting_fields": ["model"]},
            hint="Pass one or the other.",
        )
        msg = str(err)
        assert "Conflicting fields" in msg
        assert "conflicting_fields" in msg


# ---------------------------------------------------------------------------
# SearchConfig raises SearchConfigError (not plain ValueError)
# ---------------------------------------------------------------------------


class TestSearchConfigErrors:
    """SearchConfig.__post_init__ raises SearchConfigError on conflicts."""

    def test_models_conflict_raises_search_config_error(self) -> None:
        with pytest.raises(SearchConfigError, match="Cannot combine models_config"):
            SearchConfig(
                model="openai:gpt-4o",
                models_config=SearchModels(model="openai:gpt-4o-mini"),
            )

    def test_sources_conflict_raises_search_config_error(self) -> None:
        with pytest.raises(SearchConfigError, match="Cannot combine sources_config"):
            SearchConfig(
                searxng_url="http://localhost:8888",
                sources_config=SearchSources(searxng_url="http://other:9999"),
            )

    def test_unknown_source_raises_search_config_error(self) -> None:
        with pytest.raises(SearchConfigError, match="Unknown search source"):
            SearchConfig(sources=["web", "not_a_real_source"])

    def test_conflict_error_has_context(self) -> None:
        with pytest.raises(SearchConfigError) as exc_info:
            SearchConfig(
                model="openai:gpt-4o",
                models_config=SearchModels(model="openai:gpt-4o-mini"),
            )
        err = exc_info.value
        assert "conflicting_fields" in err.context
        assert err.hint is not None

    def test_unknown_source_error_has_hint(self) -> None:
        with pytest.raises(SearchConfigError) as exc_info:
            SearchConfig(sources=["badSource"])
        err = exc_info.value
        assert err.hint is not None
        assert "web" in err.hint  # valid source listed in hint


# ---------------------------------------------------------------------------
# _normalize_mode raises SearchError (not plain ValueError)
# ---------------------------------------------------------------------------


class TestNormalizeMode:
    """_normalize_mode raises SearchError on unknown modes."""

    def test_valid_modes_pass(self) -> None:
        for mode in ("speed", "balanced", "quality", "deep"):
            assert _normalize_mode(mode) == mode

    def test_case_insensitive(self) -> None:
        assert _normalize_mode("SPEED") == "speed"
        assert _normalize_mode("Balanced") == "balanced"

    def test_unknown_mode_raises_search_error(self) -> None:
        with pytest.raises(SearchError, match="Unknown search mode"):
            _normalize_mode("fasst")

    def test_unknown_mode_error_has_context_and_hint(self) -> None:
        with pytest.raises(SearchError) as exc_info:
            _normalize_mode("turbo")
        err = exc_info.value
        assert "mode" in err.context
        assert err.hint is not None
        # hint should list valid options
        assert "balanced" in err.hint


# ---------------------------------------------------------------------------
# search_and_collect partial-failure resilience
# ---------------------------------------------------------------------------


class TestSearchAndCollect:
    """search_and_collect tolerates partial query failures."""

    async def test_partial_failures_do_not_propagate(self) -> None:
        """When some query threads raise, results from successful ones are returned."""
        call_count = 0

        def _flaky_search(q: str, *args: object, **kwargs: object) -> list[dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated search failure")
            return [{"url": f"http://example.com/{q}", "title": q, "content": ""}]

        import exo.search.tools.searxng as searxng_mod

        original = searxng_mod._search
        searxng_mod._search = _flaky_search  # type: ignore[assignment]
        try:
            results = await search_and_collect(["query1", "query2"])
        finally:
            searxng_mod._search = original

        # Should not raise; should have partial results from the successful query
        assert isinstance(results, list)
        # query2 succeeded, so we should have at least one result
        assert len(results) >= 1

    async def test_all_failures_returns_empty(self) -> None:
        """If every query fails, return empty list rather than raising."""

        def _always_fails(q: str, *args: object, **kwargs: object) -> list[dict]:
            raise RuntimeError("all queries fail")

        import exo.search.tools.searxng as searxng_mod

        original = searxng_mod._search
        searxng_mod._search = _always_fails  # type: ignore[assignment]
        try:
            results = await search_and_collect(["q1", "q2", "q3"])
        finally:
            searxng_mod._search = original

        assert results == []
