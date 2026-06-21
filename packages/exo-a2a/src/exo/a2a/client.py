"""A2A client — HTTP client and RemoteAgent for calling remote A2A agents."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from exo.a2a.types import (  # pyright: ignore[reportMissingImports]
    AgentCard,
    ClientConfig,
)
from exo.types import AgentOutput, ExoError, Usage  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class A2AClientError(ExoError):
    """Raised for A2A client-level errors."""


# ---------------------------------------------------------------------------
# A2A HTTP client
# ---------------------------------------------------------------------------


class A2AClient:
    """HTTP client for communicating with a remote A2A agent.

    Resolves agent cards from URLs or local files, sends tasks, and
    optionally streams responses.

    Args:
        agent_card: An ``AgentCard`` instance, a URL string pointing to a
            ``/.well-known/agent-card`` endpoint, or a local file path.
        config: Client configuration (timeout, streaming prefs, etc.).
    """

    __slots__ = ("_agent_card", "_config", "_http", "_source")

    def __init__(
        self,
        agent_card: AgentCard | str,
        config: ClientConfig | None = None,
    ) -> None:
        self._config = config or ClientConfig()
        self._source: str | None = None
        self._agent_card: AgentCard | None = None

        if isinstance(agent_card, AgentCard):
            self._agent_card = agent_card
        elif isinstance(agent_card, str):
            if not agent_card.strip():
                raise A2AClientError("agent_card string cannot be empty")
            self._source = agent_card.strip()
        else:
            raise A2AClientError(
                f"agent_card must be AgentCard, URL, or file path, got {type(agent_card).__name__}"
            )
        self._http = httpx.AsyncClient(timeout=self._config.timeout)

    # -- Agent card resolution ------------------------------------------------

    async def resolve_agent_card(self) -> AgentCard:
        """Resolve and cache the agent card.

        Returns:
            The resolved ``AgentCard``.

        Raises:
            A2AClientError: If resolution fails.
        """
        if self._agent_card is not None:
            return self._agent_card

        if self._source is None:
            raise A2AClientError("No agent card source to resolve")

        if self._source.startswith(("http://", "https://")):
            self._agent_card = await self._resolve_from_url(self._source)
        else:
            self._agent_card = self._resolve_from_file(self._source)
        return self._agent_card

    async def _resolve_from_url(self, url: str) -> AgentCard:
        """Fetch agent card JSON from a remote URL."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise A2AClientError(f"Invalid URL: {url}")
        try:
            resp = await self._http.get(url)
            resp.raise_for_status()
            return AgentCard(**resp.json())
        except httpx.HTTPError as exc:
            raise A2AClientError(f"Failed to fetch agent card from {url}: {exc}") from exc

    @staticmethod
    def _resolve_from_file(path_str: str) -> AgentCard:
        """Load agent card JSON from a local file."""
        path = Path(path_str)
        if not path.is_file():
            raise A2AClientError(f"Agent card file not found: {path_str}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return AgentCard(**data)
        except (json.JSONDecodeError, Exception) as exc:
            raise A2AClientError(f"Invalid agent card file {path_str}: {exc}") from exc

    # -- Task execution -------------------------------------------------------

    async def send_task(self, text: str, *, task_id: str | None = None) -> dict[str, Any]:
        """Send a task to the remote agent and return the response.

        Args:
            text: The input text for the task.
            task_id: Optional task identifier. Auto-generated if omitted.

        Returns:
            Response dict from the server.

        Raises:
            A2AClientError: If the request fails.
        """
        card = await self.resolve_agent_card()
        url = card.url.rstrip("/")
        payload: dict[str, Any] = {"text": text}
        if task_id:
            payload["task_id"] = task_id

        try:
            resp = await self._http.post(f"{url}/", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise A2AClientError(f"Task request failed: {exc}") from exc

    async def send_task_collect(
        self,
        text: str,
        *,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Send a task to the streaming endpoint and collect all NDJSON events.

        This method **buffers the full response** before returning — it is
        *not* a true async generator.  It exists for convenience when the
        caller wants a simple list of events rather than incremental delivery.
        The remote server must have streaming enabled (``ServingConfig(streaming=True)``).

        Args:
            text: The input text for the task.
            task_id: Optional task identifier.

        Returns:
            List of parsed NDJSON event dicts received from ``/stream``.

        Raises:
            A2AClientError: If the request fails or streaming is not supported.
        """
        card = await self.resolve_agent_card()
        if not card.capabilities.streaming:
            raise A2AClientError("Remote agent does not support streaming")

        url = card.url.rstrip("/")
        payload: dict[str, Any] = {"text": text}
        if task_id:
            payload["task_id"] = task_id

        try:
            resp = await self._http.post(f"{url}/stream", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise A2AClientError(f"Stream request failed: {exc}") from exc

        events: list[dict[str, Any]] = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events

    # -- Lifecycle ------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    def __repr__(self) -> str:
        name = self._agent_card.name if self._agent_card else self._source or "unresolved"
        return f"A2AClient({name!r})"


# ---------------------------------------------------------------------------
# RemoteAgent — Agent-like wrapper for remote A2A agents
# ---------------------------------------------------------------------------


class RemoteAgent:
    """Agent-compatible wrapper for calling a remote A2A agent.

    Provides the same ``run(input, ...)`` interface as ``exo.agent.Agent``
    so it can be used as a handoff target or standalone caller.

    Args:
        name: Local name for this remote agent.
        agent_card: ``AgentCard``, URL, or file path for the remote agent.
        config: Client configuration.
    """

    __slots__ = ("_client", "name")

    def __init__(
        self,
        *,
        name: str,
        agent_card: AgentCard | str,
        config: ClientConfig | None = None,
    ) -> None:
        self.name = name
        self._client = A2AClient(agent_card, config)

    async def run(self, input: str, **kwargs: Any) -> AgentOutput:
        """Send input to the remote agent and return the parsed output.

        Args:
            input: The user query text.
            **kwargs: Ignored (compatibility with Agent.run signature).

        Returns:
            ``AgentOutput`` with the remote agent's response text.
        """
        logger.debug("RemoteAgent.run: name=%s input=%.80s...", self.name, input)
        resp = await self._client.send_task(input)
        text = _extract_text(resp)
        return AgentOutput(text=text, tool_calls=[], usage=Usage())

    def describe(self) -> dict[str, Any]:
        """Return a description using the cached agent card.

        This method is **synchronous** to match the ``describe()`` signature of
        local ``Agent`` objects and allow duck-typed callers to work correctly.

        If the agent card has already been resolved (i.e. ``RemoteAgent`` was
        constructed with an ``AgentCard`` instance, or ``run()`` has been called
        at least once), the cached card is used immediately.  If the card has
        not yet been fetched, a minimal description is returned with only the
        locally known ``name``; callers that need the full remote card should
        call ``await describe_async()`` instead.
        """
        logger.debug("RemoteAgent.describe: name=%s", self.name)
        card = self._client._agent_card
        if card is not None:
            return {
                "name": self.name,
                "remote_name": card.name,
                "description": card.description,
                "url": card.url,
                "capabilities": card.capabilities.model_dump(),
            }
        # Card not yet resolved — return minimal info from local state only.
        return {
            "name": self.name,
            "remote_name": None,
            "description": "",
            "url": self._client._source or "",
            "capabilities": {},
        }

    async def describe_async(self) -> dict[str, Any]:
        """Return a full description, resolving the agent card if needed.

        Use this when you need the complete remote agent metadata and the card
        may not have been fetched yet.
        """
        logger.debug("RemoteAgent.describe_async: name=%s", self.name)
        card = await self._client.resolve_agent_card()
        return {
            "name": self.name,
            "remote_name": card.name,
            "description": card.description,
            "url": card.url,
            "capabilities": card.capabilities.model_dump(),
        }

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.close()

    def __repr__(self) -> str:
        return f"RemoteAgent(name={self.name!r}, client={self._client!r})"


def _extract_text(resp: dict[str, Any]) -> str:
    """Extract text from a task execution response.

    Checks ``artifact.text``, then ``result``, then falls back to empty string.
    """
    artifact = resp.get("artifact")
    if isinstance(artifact, dict):
        text = artifact.get("text", "")
        if text:
            return text
    result = resp.get("result")
    if isinstance(result, str):
        return result
    return ""
