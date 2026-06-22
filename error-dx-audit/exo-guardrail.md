# exo-guardrail — Error DX & Resilience Audit

## Counts
- raise sites: 4
- error classes total / not inheriting ExoError: 1 / 0
  - `GuardrailError(ExoError)` — inherits correctly (`types.py:49`)
  - All `ValueError` raises are at construction/config-validation boundaries (not escaping the package boundary into user-facing async flows)
- `except Exception` sites: 3; swallow-and-pass: 0; drop-cause: 2
  - `llm_backend.py:132` — `except Exception as exc:` — does NOT drop cause but logs and returns a value (two paths, neither re-raises); cause is not chained
  - `llm_backend.py:186` — `except (json.JSONDecodeError, ValueError):` — drops cause entirely, returns safe default (fail-open)
  - `base.py:90` — `except ValueError:` — re-raises immediately as new `ValueError(msg) from None` (intentional, suppresses noisy internal context)
- `CancelledError` handlers: 0 — no explicit handling; `except Exception` does NOT catch `CancelledError` (Python 3.8+), so cancellation propagates correctly
- I/O call sites lacking timeout: 1 (`llm_backend.py:126` — `provider.complete(...)` with no timeout guard)

## Findings (prioritized)

**[P0] | checklist#4,5 | llm_backend.py:186** — `_parse_llm_response` fails **open** on JSON parse failure.
When the LLM returns non-JSON or malformed JSON, the `except (json.JSONDecodeError, ValueError)` block logs a warning and returns `RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)`. This silently treats a garbled/truncated LLM response as "safe" — exactly backwards for a security component. The cause is also dropped with no chaining.
Fix: Return `RiskAssessment(has_risk=True, risk_level=RiskLevel.HIGH, risk_type="parse_failure", details={"raw": content[:200]})` on parse failure (matching the fail-closed contract already documented on `LLMGuardrailBackend`). Add `from exc` on any re-raise variants.

**[P0] | checklist#4 | llm_backend.py:190-191** — Unrecognised `risk_level` string from LLM response silently falls back to `RiskLevel.SAFE`.
`risk_level = RiskLevel(raw_level) if raw_level in _VALID_RISK_LEVELS else RiskLevel.SAFE` — if the model outputs e.g. `"unknown"` or hallucinates a level name, the system treats it as safe rather than blocking. This is a fail-open on a security check.
Fix: Fall back to `RiskLevel.HIGH` (fail-closed) when the level string is not in `_VALID_RISK_LEVELS`, and log a warning with the raw value.

**[P0] | checklist#4 | llm_backend.py:193-194** — `has_risk=False` from LLM response is trusted unconditionally even when `risk_level` is non-SAFE.
If an adversarial or glitched response returns `{"has_risk": false, "risk_level": "critical"}`, the current code returns `RiskAssessment(has_risk=False, ...)` and `BaseGuardrail.detect()` returns `GuardrailResult.safe()`.
Fix: Treat non-SAFE `risk_level` as implying `has_risk=True` regardless of the boolean field.

**[P1] | checklist#1 | base.py:59,73,92** — Three `ValueError` raises at construction/validation time do not inherit `ExoError`.
The charter requires all package-boundary exceptions to be `ExoError` subclasses. `BaseGuardrail(backend=None)` and `BaseGuardrail(events=["bad_hook"])` both raise plain `ValueError`, which callers cannot catch as `ExoError`.
Fix: Raise `GuardrailConfigError(ExoError)` (new subclass) with `hint=` pointing to valid backends / `HookPoint` enum values.

**[P1] | checklist#2,3 | base.py:120-126** — `GuardrailError` raised in the hook body is missing `context=` and `hint=`.
Current message: `"Guardrail blocked at pre_llm_call: prompt_injection"`. It names the hook point and risk type but does not tell the developer which agent/guardrail triggered it, what the matched input contained, or what to do.
Fix: Add `context={"hook": __point.value, "risk_level": result.risk_level, "risk_type": result.risk_type}` and `hint="Inspect result.details for matched patterns; adjust guardrail thresholds or whitelist the input."`.

**[P1] | checklist#3 | types.py:66** — `GuardrailError.__init__` calls `super().__init__(message)` but does not forward `context=`, `hint=`, or `doc=` to `ExoError`.
The `GuardrailError` constructor signature does not accept these fields, so callers cannot attach structured context/hints even if they wanted to.
Fix: Add `context`, `hint`, and `doc` kwargs and pass them to `super().__init__()`.

**[P1] | checklist#5 | llm_backend.py:132** — `except Exception as exc:` in `analyze()` does not chain the cause on any path.
The fail-closed path logs `exc` and returns an assessment; the fail-open path does the same. Neither propagates `exc` on `__cause__`. The cause chain is therefore invisible to callers who catch `GuardrailError` further up.
Note: Because the method returns a value rather than raising, there is no `raise ... from exc` to write — but the cause is completely discarded. If a future refactor raises here, the chaining would be missing.
Fix: Acceptable as-is if returning is intentional; add a comment noting the cause is consumed by the logged warning and encoded in `details["error"]`.

**[P1] | checklist#8 | llm_backend.py:126** — LLM `provider.complete(...)` call has no timeout.
A slow or hung provider backend will block the guardrail hook indefinitely, stalling the agent's entire `pre_llm_call` hook chain with no deadline.
Fix: Wrap with `asyncio.wait_for(provider.complete(...), timeout=self._timeout)` where `_timeout` defaults to e.g. 10 seconds; on `asyncio.TimeoutError`, apply the same fail-closed/fail-open logic as other backend failures.

**[P2] | checklist#2 | llm_backend.py:186-188** — Log message for parse failure only shows `content[:200]` with no model name.
Fix: Include `model=self._model` (not accessible here since it's a module-level function) — move the parse failure log into `analyze()` or thread the model name through.

**[P2] | checklist#9 | llm_backend.py:18** — Logging is correctly `logging.getLogger(__name__)` (non-internal file). No issue.

**[P2] | checklist#2 | base.py:92** — `raise ValueError(msg) from None` suppresses cause chain intentionally to avoid confusing `HookPoint(name)` internals leaking out, which is reasonable, but the raised message could be more actionable for a mistyped event string that slipped through `__init__` validation.
Fix: Minor — add `hint="Use HookPoint.<NAME> for validated hook points."` (requires converting to `GuardrailConfigError`).

## Resilience gaps

**`llm_backend.py:126`** — `await provider.complete(...)` — no timeout. A hanging LLM call (network stall, slow provider) blocks the entire guardrail hook with no deadline. The guardrail is called synchronously in the agent's hook chain, so a hung LLM here hangs the whole agent. Mitigation: `asyncio.wait_for(..., timeout=self._timeout)` with a reasonable default (10s), feeding the `TimeoutError` into the existing fail-closed path.

**`llm_backend.py:132`** — `except Exception` correctly catches most failures but does not explicitly handle `asyncio.TimeoutError` (which is an `Exception` subclass), so timeout would fall into the fail-closed/fail-open branch — that part is actually fine as-is once a timeout wrapper is added.

## Effort estimate

S — The codebase is small (5 source files, ~400 LOC). The critical P0 fixes (fail-open on parse failure and on unrecognised risk level) are 2–3 targeted line changes. The `GuardrailError` context/hint wiring and `GuardrailConfigError` taxonomy fix are each under 20 lines. Timeout wiring adds ~5 lines. Total: half a day's work including tests.
