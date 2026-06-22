"""Tests for SearchConfig grouped namespace configs (Tier-3 DX)."""

from __future__ import annotations

import pytest

from exo.search import SearchConfigError
from exo.search.config import (
    ResearchConfig,
    RevisionConfig,
    SearchConfig,
    SearchModels,
    SearchSources,
    VerificationConfig,
    WriterConfig,
)


class TestSearchModels:
    """SearchModels namespace populates flat fields."""

    def test_models_config_populates_flat_fields(self) -> None:
        cfg = SearchConfig(
            models_config=SearchModels(
                model="openai:gpt-4o",
                fast_model="openai:gpt-4o-mini",
                embedding_model="text-embedding-3-small",
                context_window_tokens=64_000,
            )
        )
        assert cfg.model == "openai:gpt-4o"
        assert cfg.fast_model == "openai:gpt-4o-mini"
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.context_window_tokens == 64_000

    def test_flat_model_fields_still_work(self) -> None:
        cfg = SearchConfig(model="openai:gpt-4o-mini", fast_model="openai:gpt-4o-mini")
        assert cfg.model == "openai:gpt-4o-mini"
        assert cfg.fast_model == "openai:gpt-4o-mini"

    def test_models_config_conflict_raises(self) -> None:
        with pytest.raises(SearchConfigError, match="Cannot combine models_config"):
            SearchConfig(
                model="openai:gpt-4o",
                models_config=SearchModels(model="openai:gpt-4o-mini"),
            )

    def test_model_config_accessor_returns_view(self) -> None:
        cfg = SearchConfig(model="openai:gpt-4o", context_window_tokens=128_000)
        view = cfg.model_config
        assert isinstance(view, SearchModels)
        assert view.model == "openai:gpt-4o"
        assert view.context_window_tokens == 128_000


class TestSearchSources:
    """SearchSources namespace populates flat fields."""

    def test_sources_config_populates_flat_fields(self) -> None:
        cfg = SearchConfig(
            sources_config=SearchSources(
                searxng_url="http://example.com:9999",
                searxng_timeout=30,
                sources=["web", "academic"],
                max_results=20,
                jina_api_key="jina-key",
            )
        )
        assert cfg.searxng_url == "http://example.com:9999"
        assert cfg.searxng_timeout == 30
        assert "web" in cfg.sources
        assert "academic" in cfg.sources
        assert cfg.max_results == 20
        assert cfg.jina_api_key == "jina-key"

    def test_flat_sources_field_still_works(self) -> None:
        cfg = SearchConfig(sources=["web", "academic"])
        assert "web" in cfg.sources
        assert "academic" in cfg.sources

    def test_sources_config_conflict_raises(self) -> None:
        with pytest.raises(SearchConfigError, match="Cannot combine sources_config"):
            SearchConfig(
                searxng_url="http://localhost:8888",
                sources_config=SearchSources(searxng_url="http://other:9999"),
            )

    def test_source_config_accessor_returns_view(self) -> None:
        cfg = SearchConfig(max_results=25)
        view = cfg.source_config
        assert isinstance(view, SearchSources)
        assert view.max_results == 25


class TestResearchConfig:
    """ResearchConfig namespace populates flat fields."""

    def test_research_config_populates_flat_fields(self) -> None:
        cfg = SearchConfig(
            research=ResearchConfig(
                research_mode="quality",
                max_iterations=10,
                max_deep_research_steps=12,
                deep_research_enrich_per_step=8,
                max_content_chars=20_000,
            )
        )
        assert cfg.research_mode == "quality"
        assert cfg.max_iterations == 10
        assert cfg.max_deep_research_steps == 12
        assert cfg.deep_research_enrich_per_step == 8
        assert cfg.max_content_chars == 20_000

    def test_flat_research_fields_still_work(self) -> None:
        cfg = SearchConfig(research_mode="speed", max_iterations=3)
        assert cfg.research_mode == "speed"
        assert cfg.max_iterations == 3

    def test_research_conflict_raises(self) -> None:
        with pytest.raises(SearchConfigError, match="Cannot combine research"):
            SearchConfig(
                research_mode="speed",
                research=ResearchConfig(research_mode="quality"),
            )

    def test_research_config_accessor_returns_view(self) -> None:
        cfg = SearchConfig(research_mode="quality", max_deep_research_steps=15)
        view = cfg.research_config
        assert isinstance(view, ResearchConfig)
        assert view.research_mode == "quality"
        assert view.max_deep_research_steps == 15


class TestWriterConfig:
    """WriterConfig namespace populates flat fields."""

    def test_writer_config_populates_flat_fields(self) -> None:
        cfg = SearchConfig(
            writer=WriterConfig(
                system_instructions="Be concise.",
                max_writer_words=1500,
                max_writer_sources=10,
                claim_first_writing=False,
                use_reasoning_preamble=True,
            )
        )
        assert cfg.system_instructions == "Be concise."
        assert cfg.max_writer_words == 1500
        assert cfg.max_writer_sources == 10
        assert cfg.claim_first_writing is False
        assert cfg.use_reasoning_preamble is True

    def test_flat_writer_fields_still_work(self) -> None:
        cfg = SearchConfig(system_instructions="Hello", max_writer_words=2000)
        assert cfg.system_instructions == "Hello"
        assert cfg.max_writer_words == 2000

    def test_writer_conflict_raises(self) -> None:
        with pytest.raises(SearchConfigError, match="Cannot combine writer"):
            SearchConfig(
                system_instructions="Hello",
                writer=WriterConfig(system_instructions="World"),
            )

    def test_writer_config_accessor_returns_view(self) -> None:
        cfg = SearchConfig(max_writer_sources=8, claim_first_writing=False)
        view = cfg.writer_config
        assert isinstance(view, WriterConfig)
        assert view.max_writer_sources == 8
        assert view.claim_first_writing is False


class TestVerificationConfig:
    """VerificationConfig namespace populates flat fields."""

    def test_verification_config_populates_flat_fields(self) -> None:
        cfg = SearchConfig(
            verification=VerificationConfig(
                llm_verification=True,
                llm_verify_source_chars=8000,
            )
        )
        assert cfg.llm_verification is True
        assert cfg.llm_verify_source_chars == 8000

    def test_flat_verification_fields_still_work(self) -> None:
        cfg = SearchConfig(llm_verification=True, llm_verify_source_chars=5000)
        assert cfg.llm_verification is True
        assert cfg.llm_verify_source_chars == 5000

    def test_verification_conflict_raises(self) -> None:
        with pytest.raises(SearchConfigError, match="Cannot combine verification"):
            SearchConfig(
                llm_verification=True,
                verification=VerificationConfig(llm_verification=True),
            )

    def test_verification_config_accessor_returns_view(self) -> None:
        cfg = SearchConfig(llm_verification=True)
        view = cfg.verification_config
        assert isinstance(view, VerificationConfig)
        assert view.llm_verification is True


class TestRevisionConfig:
    """RevisionConfig namespace populates flat fields."""

    def test_revision_config_populates_flat_fields(self) -> None:
        cfg = SearchConfig(
            revision=RevisionConfig(
                max_revision_rounds=5,
                revision_threshold=0.5,
            )
        )
        assert cfg.max_revision_rounds == 5
        assert cfg.revision_threshold == 0.5

    def test_flat_revision_fields_still_work(self) -> None:
        cfg = SearchConfig(max_revision_rounds=3, revision_threshold=0.4)
        assert cfg.max_revision_rounds == 3
        assert cfg.revision_threshold == 0.4

    def test_revision_conflict_raises(self) -> None:
        with pytest.raises(SearchConfigError, match="Cannot combine revision"):
            SearchConfig(
                max_revision_rounds=3,
                revision=RevisionConfig(max_revision_rounds=5),
            )

    def test_revision_config_accessor_returns_view(self) -> None:
        cfg = SearchConfig(max_revision_rounds=4, revision_threshold=0.2)
        view = cfg.revision_config
        assert isinstance(view, RevisionConfig)
        assert view.max_revision_rounds == 4
        assert view.revision_threshold == 0.2


class TestCompositeNamespaceConfigs:
    """Multiple namespace configs can be combined."""

    def test_multiple_namespaces_combined(self) -> None:
        cfg = SearchConfig(
            models_config=SearchModels(model="openai:gpt-4o"),
            research=ResearchConfig(research_mode="quality", max_iterations=20),
            verification=VerificationConfig(llm_verification=True),
        )
        assert cfg.model == "openai:gpt-4o"
        assert cfg.research_mode == "quality"
        assert cfg.max_iterations == 20
        assert cfg.llm_verification is True

    def test_public_exports(self) -> None:
        from exo.search import (
            ResearchConfig,
            RevisionConfig,
            SearchConfig,
            SearchModels,
            SearchSources,
            VerificationConfig,
            WriterConfig,
        )

        assert SearchConfig is not None
        assert SearchModels is not None
        assert SearchSources is not None
        assert ResearchConfig is not None
        assert WriterConfig is not None
        assert VerificationConfig is not None
        assert RevisionConfig is not None
