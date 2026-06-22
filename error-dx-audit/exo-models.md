# exo-models — Error DX & Resilience Audit

## Counts
- raise sites: 22
- error classes total / not inheriting ExoError: 2 / 1
  - `EmbeddingError(Exception)` — `embeddings.py:56` (explicitly noted in charter's known offenders)
  - `ModelError(ExoError)` — `types.py:21` (already inherits ExoError, but bypasses `hint=`/`context=` fields — see below)
- `except Exception` sites: 3 (all in `media_tools.py:67, 119, 184`); swallow-and-pass: 0; drop-cause (no `from`): 0
- CancelledError handlers: 0 (no explicit handling; `except Exception` in media_tools is safe since `CancelledError` is a `BaseException`, not `Exception`, in Python 3.8+)
- I/O call sites lacking timeout/retry: 3 (Google provider `complete()` and `stream()` — no timeout or retry configured on the `genai.Client`; plus `veo_generate_video` polling loop has no timeout guard at all)

---

## Findings (prioritized)

| Priority | Checklist# | File:line | What's wrong | Concrete fix |
|---|---|---|---|---|
| P0 | 8 | `gemini.py:120–141` | `genai.Client` is constructed with no `timeout` or retry config. OpenAI/Anthropic clients accept `max_retries=` and `timeout=` from `ModelConfig`; the Google client does not. A hung Vertex AI call will block forever. | Pass `http_options={"timeout": config.timeout}` and wrap calls with tenacity/manual retry on `genai_errors.APIError` with status 429/5xx, matching the OpenAI/Anthropic pattern. |
| P0 | 8 | `media_tools.py:167–183` | `veo_generate_video` has an unbounded `asyncio.sleep(5)` polling loop with no maximum iteration count or overall timeout. A stuck Veo operation will poll indefinitely, leaking the `genai.Client` and blocking the agent. | Add `max_polls` / total-timeout guard; raise `ToolError` with a message like "Veo operation timed out after Ns — check Vertex AI console for job status." |
| P0 | 8 | `media_tools.py:57–68` | `AsyncOpenAI()` in `dalle_generate_image` is constructed with no `timeout=` or `max_retries=` override. A transient network hang blocks the calling agent. Also, the genai clients in `imagen_generate_image` and `veo_generate_video` (lines 109, 167) are similarly unconfigured for timeout. | Pass `timeout=30.0, max_retries=2` to `AsyncOpenAI()`, and use `http_options={"timeout": 30}` for `genai.Client` calls. |
| P1 | 1 | `embeddings.py:56` | `EmbeddingError(Exception)` does not inherit `ExoError`. It crosses the package boundary without the structured `hint=`/`context=` fields and cannot be caught by a blanket `except ExoError`. The charter explicitly lists this as a known offender. | Change to `class EmbeddingError(ExoError)` and update `__init__` to call `super().__init__(message, hint=..., context=...)`. Preserve `operation` and `details` as `context` fields. |
| P1 | 2 | `openai.py:352, 399` | `ModelError(str(exc), model=...)` — the raw SDK exception message is pasted verbatim as the error message. OpenAI SDK errors are often multi-line JSON blobs (status code, request ID, body). No actionable hint is ever attached (e.g. "Set OPENAI_API_KEY" on 401, "Reduce prompt size" on 400 context-length errors). | Classify by `exc.status_code`: 401 → hint "Check OPENAI_API_KEY is set correctly"; 429 → "Rate limited — reduce request frequency or add a delay"; 400 with "context" in body → "Reduce prompt length or increase max_tokens"; otherwise just `str(exc)`. |
| P1 | 2 | `anthropic.py:429, 519` | Same pattern as OpenAI above. `ModelError(str(exc), ...)` carries a raw Anthropic SDK message with no actionable routing. 401 gets no "check ANTHROPIC_API_KEY" hint; 529 overload gets no "retry later" hint. | Same fix: classify `anthropic.APIStatusError.status_code` and set `hint=` accordingly. |
| P1 | 2 | `gemini.py:188–190, 241–243` | `ModelError(str(exc), model=...)` on `genai_errors.APIError` carries no hint. Gemini API quota errors (RESOURCE_EXHAUSTED) and auth errors (PERMISSION_DENIED) reach developers with zero guidance on next action. | Inspect `exc.code` (gRPC/HTTP status) and attach hint: PERMISSION_DENIED → "Check GOOGLE_API_KEY or Vertex service account permissions"; RESOURCE_EXHAUSTED → "API quota exceeded — reduce request rate". |
| P1 | 3 | `openai.py:352, 399` `anthropic.py:429, 519` `gemini.py:188–190, 241–243` | `ModelError` bypasses `ExoError`'s `context=` dict. Although `model=` is forwarded to `[model]` prefix, no structured keys (`model`, `provider`, `status_code`) are carried. A caller doing `exc.context["model"]` gets an empty dict. | Use `ExoError.__init__(message, context={"model": ..., "status_code": ...}, hint=...)` instead of the current flat-string approach. Requires updating `ModelError.__init__` to forward kwargs to `super().__init__`. |
| P1 | 2 | `anthropic.py:375–379` | `AnthropicProvider.__init__` raises `ModelError("No API key found...")` — this is good. But it omits a `hint=` and does not use `context=`. Compare: GoogleProvider has the same pattern at `gemini.py:135–139`. | Add `hint="Set ANTHROPIC_API_KEY env var or pass api_key= to get_provider()"` and `context={"model": f"anthropic:{config.model_name}"}`. |
| P1 | 2 | `gemini.py:135–139` | Same as above for the Google provider — missing `hint=` on the "No API key found" path. | Add `hint="Set GOOGLE_API_KEY (or GEMINI_API_KEY) env var or pass api_key= to get_provider()"`. |
| P1 | 2 | `provider.py:124–127` | `ModelError(f"Provider '{provider_name}' not registered. Available: {available}", model=model)` — good first step. But `from None` drops the `RegistryError` cause, and there is no `hint=`. | Keep `from None` (the RegistryError adds no info), but add `hint=f"Check the model string format 'provider:model'. Registered providers: {available}"`. |
| P1 | 2 | `embeddings.py:232–235, 343–347, 460–463` | `EmbeddingError(f"OpenAI/Vertex/HTTP embeddings API error: {status_code}", ...)` — no hint. A 401 gives the developer a status code but no next step. | Add provider-specific hints for 401 ("Check api_key") and 429 ("Reduce batch size or add a delay"). Include the response body snippet in the message for non-401 cases. |
| P1 | 8 | `embeddings.py:222, 334, 450` | `httpx.AsyncClient()` is created fresh per `embed_batch()` call. This is functionally correct (properly closed via `async with`) but creates a new TCP connection on every call. At high call rates this is inefficient, and more importantly there is no retry on transient failures (connection reset, 502, 503). | Add exponential backoff on `httpx.HTTPStatusError` with 429/5xx; for high-throughput users, recommend a shared `httpx.AsyncClient` as a class attribute. |
| P2 | 9 | `openai.py:43` | Uses `_log = logging.getLogger(__name__)` instead of the core internal pattern. But this file is not in `_internal/`, so the standard `logging.getLogger` pattern is correct per CLAUDE.md. No action needed. — Actually correct as-is. | N/A — this is the correct non-internal pattern. |
| P2 | 3 | `embeddings.py:237–241, 349–353, 465–469` | `EmbeddingError(f"...request failed: {exc}", operation="embed")` — no `details=` context with the URL or model. A network failure gives no indication of which endpoint failed. | Add `details={"url": self._base_url, "model": self._model}` (for OpenAI/Vertex) or `details={"url": self._url}` (for HTTPEmbeddings). |
| P2 | 2 | `anthropic.py:111–121` | `ModelError("Anthropic does not support audio/video input; remove ... before calling this provider", model="anthropic")` — has no `hint=`. This is an excellent message already but could use a structured hint for IDEs. | Add `hint="Convert to a text description or use a provider that supports this media type (e.g. gemini:gemini-2.0-flash)."` |
| P2 | 2 | `_google_common.py:125–128` | `ModelError("VideoBlock has neither url nor data...", model="google")` — good message, missing `hint=`. | Add `hint="Set either VideoBlock.url or VideoBlock.data before passing to the Google provider."` |
| P2 | 5 | `provider.py:127` | `from None` deliberately drops the `RegistryError` cause. The charter says always chain or make a deliberate choice. Here it's deliberate (RegistryError just says "key not found") — but worth a comment explaining why. | Add `# RegistryError only says "key not found" — the ModelError message already contains all context.` |
| P2 | 2 | `media_tools.py:68, 120, 185` | `except Exception as exc: raise ToolError(f"... failed: {exc}") from exc` — this is properly chained. But the messages are uninformative (no status code, no provider-specific guidance). DALL-E 401 and Imagen 403 both produce `"DALL-E 3 generation failed: ..."` with no hint. | Narrow the `except` to `openai.APIError`/`genai_errors.APIError` on the fast path and add hints for common codes; keep the broad `except Exception` as a fallback but log it. |

---

## Resilience gaps

| File:line | External system | Gap |
|---|---|---|
| `gemini.py:120–141` (GoogleProvider `__init__`) | Google genai SDK | No `timeout` or `max_retries` passed to `genai.Client`. Neither `complete()` nor `stream()` have a request-level timeout guard. A network stall hangs forever. |
| `media_tools.py:167–185` (veo_generate_video) | Vertex AI Veo | Unbounded polling loop (`while not operation.done`) with `asyncio.sleep(5)` and no max iterations or total-time cap. |
| `media_tools.py:57–68` (dalle_generate_image) | OpenAI Images API | Fresh `AsyncOpenAI()` with no `timeout=` or `max_retries=`. |
| `media_tools.py:109–120` (imagen_generate_image) | Google Imagen (Vertex) | Fresh `genai.Client()` with no timeout configuration. |
| `embeddings.py:222, 334, 450` | OpenAI / Vertex / HTTP embeddings | No retry/backoff on transient 429/5xx. A single failed batch is immediately surfaced as an `EmbeddingError`. Configuring tenacity with `wait_exponential` on those three `except httpx.HTTPStatusError` blocks would harden the hot path significantly. |
| `openai.py:304–310` `anthropic.py:380–385` | OpenAI / Anthropic SDK | Both providers delegate retry/backoff to the SDK (`max_retries=config.max_retries`, `timeout=config.timeout`). This is correct — SDK handles 429/5xx natively. No gap for these two. |

---

## Effort estimate

**M** (medium). The taxonomy fix (`EmbeddingError → ExoError`) is one class change with minor ripple to callers. Modernising `ModelError.__init__` to forward `hint=`/`context=` to `ExoError` unlocks the rest. Adding hint classification for 401/429/5xx on three providers is repetitive but mechanical. The biggest new work is the Google provider timeout/retry (no SDK knob — requires wrapping calls in `asyncio.wait_for` or a tenacity retry) and the Veo polling-loop guard. Overall: ~2–3 focused coding sessions, one per wave (taxonomy → actionable messages → resilience).
