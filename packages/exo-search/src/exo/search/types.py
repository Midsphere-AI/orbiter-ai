"""Pydantic data models for the Exo Search search engine."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ResearchMode(StrEnum):
    """Research quality modes matching Exo Search."""

    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"
    DEEP = "deep"


class SearchSource(StrEnum):
    """Known source identifiers for ``SearchConfig.sources``.

    Pass these (or their string equivalents) to ``SearchConfig.sources``::

        cfg = SearchConfig(sources=[SearchSource.WEB, SearchSource.ACADEMIC])
        # or equivalently:
        cfg = SearchConfig(sources=["web", "academic"])

    Unknown strings raise ``ValueError`` at config construction time.
    """

    WEB = "web"
    ACADEMIC = "academic"
    DISCUSSIONS = "discussions"


class Classification(BaseModel):
    """Exo Search's classifier output labels."""

    skip_search: bool = Field(default=False, alias="skipSearch")
    personal_search: bool = Field(default=False, alias="personalSearch")
    academic_search: bool = Field(default=False, alias="academicSearch")
    discussion_search: bool = Field(default=False, alias="discussionSearch")
    show_weather_widget: bool = Field(default=False, alias="showWeatherWidget")
    show_stock_widget: bool = Field(default=False, alias="showStockWidget")
    show_calculation_widget: bool = Field(default=False, alias="showCalculationWidget")
    requires_sequential_research: bool = Field(default=False, alias="requiresSequentialResearch")
    estimated_complexity: str = Field(default="moderate", alias="estimatedComplexity")

    model_config = {"populate_by_name": True}


class ClassifierOutput(BaseModel):
    """Full classifier response."""

    classification: Classification
    standalone_follow_up: str = Field(default="", alias="standaloneFollowUp")
    sub_questions: list[str] = Field(default_factory=list, alias="subQuestions")

    model_config = {"populate_by_name": True}

    @property
    def requires_sequential_research(self) -> bool:
        """Convenience accessor for the sequential research flag."""
        return self.classification.requires_sequential_research

    @property
    def estimated_complexity(self) -> str:
        """Convenience accessor for the estimated complexity level."""
        return self.classification.estimated_complexity


class SearchResult(BaseModel):
    """A single search result from SearXNG."""

    title: str = ""
    url: str = ""
    content: str = ""
    enriched: bool = False  # True when content is full page text (e.g., from Jina Search)


class Source(BaseModel):
    """A cited source in the final response."""

    title: str
    url: str
    content: str = ""


class QueryPlan(BaseModel):
    """Structured output for adaptive query generation."""

    queries: list[str] = Field(default_factory=list)
    sufficient: bool = Field(default=False)


class SuggestionOutput(BaseModel):
    """Structured suggestion output."""

    suggestions: list[str] = Field(default_factory=list)


class PipelineEvent(BaseModel):
    """Pipeline stage transition event."""

    type: str = "pipeline"
    stage: str  # "classifier", "researcher", "writer", "suggestions"
    status: str  # "started", "completed"
    message: str = ""


class ResearchStep(BaseModel):
    """A single step in a sequential deep research plan."""

    step_id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    extraction_goal: str


class ResearchPlan(BaseModel):
    """LLM-generated plan for sequential deep research."""

    steps: list[ResearchStep]
    reasoning: str = ""


class StepExtraction(BaseModel):
    """Extracted findings from a single research step."""

    step_id: str
    extracted_info: str
    found: bool = True


class ExtractedClaim(BaseModel):
    """A structured claim extracted from source content for claim-first writing."""

    claim: str
    source_index: int  # 1-based
    verbatim_quote: str = ""


class VerificationResult(BaseModel):
    """Result of LLM-based claim verification against a source."""

    supported: bool
    reason: str = ""


class CitationVerification(BaseModel):
    """Stats from post-hoc citation verification."""

    total_citations: int = 0
    verified: int = 0
    removed: int = 0
    flagged: int = 0
    llm_verified: int = 0
    revision_count: int = 0
    failed_claims: list[tuple[str, int]] = Field(default_factory=list, exclude=True)


class FactualClaim(BaseModel):
    """A single factual claim extracted from an answer."""

    claim_text: str
    cited_sources: list[int] = Field(default_factory=list)


class Contradiction(BaseModel):
    """A detected contradiction between sources."""

    claim_text: str
    position_a: str
    position_b: str
    source_indices_a: list[int] = Field(default_factory=list)
    source_indices_b: list[int] = Field(default_factory=list)
    severity: str = "moderate"  # "minor", "moderate", "major"


class ContradictionReport(BaseModel):
    """Report of contradictions found across sources."""

    contradictions: list[Contradiction] = Field(default_factory=list)
    claims_checked: int = 0
    has_contradictions: bool = False


class SearchResponse(BaseModel):
    """Final structured search response."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    query: str = ""
    mode: str = ""
    verification: CitationVerification | None = None
    contradictions: ContradictionReport | None = None
    confidence: float | None = None
    confidence_breakdown: dict | None = None
