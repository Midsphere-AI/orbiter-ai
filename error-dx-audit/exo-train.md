# exo-train — Error DX & Resilience Audit

## Counts

- raise sites: 42
- error classes total / not inheriting ExoError: 3 / 10 (offenders below)
  - `raise ValueError` at synthesis.py:43, 46, 112 (SynthesisConfig.__post_init__, split_dataset)
  - `raise ValueError` at evolution.py:70, 73, 201, 204 (EvolutionConfig.__post_init__, GaussianMutationStrategy.__init__)
  - `raise ValueError` at verl.py:105, 108, 111 (VeRLConfig.__post_init__)
- `except Exception` sites: 7; swallow-and-pass: 2 (verl.py:684, verl.py:703 — score = 0.0 silently); drop-cause: 2 (verl.py:425 `from None`, verl.py:493 `from None`)
- CancelledError handlers: 0
- optional-dep ImportErrors made actionable? **partial** — `_check_verl_available()` (verl.py:417) raises `ImportError` with a pip hint, but then both callers at verl.py:281 and verl.py:364 wrap it as `TrainerError(str(exc)) from exc`, losing the structured ExoError fields (no `hint=`, no `context=`). The `ImportError` is also still an `ImportError` at the `_check_verl_available` boundary (not an `ExoError` subclass), so the caller gets a raw exception type. GPU/OOM/CUDA errors are wholly unhandled — if VeRL's `trainer.fit()` raises `RuntimeError: CUDA out of memory` it surfaces as an undecorated RuntimeError wrapped in a generic `TrainerError(str(exc))` with no hint.
- I/O call sites lacking timeout/cleanup: 3 (FileCheckpointStore.save/load at operator_trainer.py:65,83; verl.py:327-328 and 387-388 run_in_executor with no timeout/cancellation guard)

---

## Findings (prioritized)

| Pri | # | File:line | What's wrong | Concrete fix |
|-----|---|-----------|-------------|--------------|
| P0 | 1 | verl.py:417–425 | `_check_verl_available()` raises a plain `ImportError`, not an `ExoError` subclass. Callers wrap it with `TrainerError(str(exc))` but no `hint=` or `context=` fields, so the structured message is never rendered. | Change `raise ImportError(msg) from None` → `raise TrainerError("VeRL is not installed.", hint="Run: pip install exo-train[verl]", context={"extra": "verl>=0.2 required"}) from None`. Remove the re-wrap at verl.py:281,364 (or keep `from exc`). |
| P0 | 2 | verl.py:254–257 | `except Exception as exc: ... raise TrainerError(f"VeRL training failed: {exc}") from exc` — swallows GPU/OOM/CUDA errors as generic strings. `RuntimeError: CUDA out of memory` becomes `"VeRL training failed: CUDA out of memory"` with no hint about what to do. | Detect `RuntimeError` containing "CUDA" / "out of memory" and re-raise with `hint="Reduce batch_size / rollout_batch_size or run on a GPU with more VRAM."`. |
| P0 | 3 | verl.py:493 | `raise TrainerError(msg) from None` — drops the import chain. The ImportError from `verl.trainer.{grpo,ppo}.ray_trainer` that triggered the fallback is lost. | Change `from None` → `from exc` (catch and rebind: `except ImportError as import_exc: ... raise TrainerError(msg) from import_exc`). |
| P1 | 4 | synthesis.py:43,46,112 | `raise ValueError(...)` escapes the package boundary instead of `SynthesisError`. `SynthesisError(ExoError)` already exists at synthesis.py:17. | Change all three `raise ValueError` → `raise SynthesisError(msg, hint="Adjust SynthesisConfig.num_samples / train_ratio to valid range.")`. |
| P1 | 5 | evolution.py:70,73,201,204 | `raise ValueError(...)` in `EvolutionConfig.__post_init__` and `GaussianMutationStrategy.__init__` — should be `EvolutionError`. `EvolutionError(ExoError)` already exists at evolution.py:26. | Replace all four with `raise EvolutionError(msg, hint="See EvolutionConfig / GaussianMutationStrategy docstring for valid ranges.")`. |
| P1 | 6 | verl.py:105,108,111 | `raise ValueError(...)` in `VeRLConfig.__post_init__` — should be `TrainerError` (the only ExoError subclass in scope in verl.py). | Replace all three with `raise TrainerError(msg, context={"config_field": "<field>"}, hint="See VeRLConfig docstring for valid parameter ranges.")`. |
| P1 | 7 | verl.py:281, 364 | `raise TrainerError(str(exc)) from exc` — `str(exc)` re-stringifies the already-good ImportError message but produces a `TrainerError` with no `hint=` or `context=`. | Use `raise TrainerError("VeRL not installed.", hint="Run: pip install exo-train[verl]") from exc` (after fixing P0/finding 1 so `_check_verl_available` itself raises `TrainerError`). |
| P1 | 8 | operator_trainer.py:350–352 | `except Exception: self._state = FAILED; raise` — correct state transition but the re-raised exception is whatever raw error the user's `eval_fn` or `optimizer.backward/step` raised, which may be uncontextualised. | Wrap: `except Exception as exc: self._state = TrainerState.FAILED; raise TrainerError(f"Training failed at epoch {epoch}.", context={"epoch": epoch}, hint="Check eval_fn and optimizer implementations.") from exc`. |
| P1 | 9 | evolution.py:411–414 | `except Exception as exc: ... raise EvolutionError(f"Evolution failed at epoch {epoch_idx}: {exc}") from exc` — message is decent, but no `hint=` and no `context=` dict (epoch, phase active). | Add `context={"epoch": epoch_idx, "state": "running"}` and `hint="Check strategy.synthesise/train/evaluate implementations."`. |
| P1 | 10 | verl.py:327–328, 387–388 | `asyncio.get_event_loop()` is deprecated in Python 3.10+. Use `asyncio.get_running_loop()` instead. Also the `run_in_executor` calls have no timeout — a hung VeRL `trainer.fit()` will block indefinitely. | Replace `get_event_loop()` with `asyncio.get_running_loop()`. Wrap executor call in `asyncio.wait_for(loop.run_in_executor(...), timeout=cfg.extra.get("timeout_seconds"))` with a sensible default (e.g. 24 h or configurable). |
| P1 | 11 | verl.py:162–163 | `check_agent`: `"VeRL training requires a non-None agent"` — no hint about what type of agent is expected or how to supply one. | Add `hint="Pass an Exo Agent (or any object with an instructions attribute) to check_agent()."`. |
| P1 | 12 | operator_trainer.py:214 | `"agent is required"` — bare one-word message, no hint. | Add `hint="Call check_agent(agent) before mark_validated()."` and `context={"trainer": type(self).__name__}`. |
| P2 | 13 | verl.py:149 | `assert isinstance(self._config, VeRLConfig)` — bare assertion in `verl_config` property that becomes `AssertionError` in production (optimised bytecode). | Replace with `if not isinstance(self._config, VeRLConfig): raise TrainerError("Config must be a VeRLConfig.", hint="Construct VeRLTrainer with a VeRLConfig instance.")`. |
| P2 | 14 | operator_trainer.py:65 | `FileCheckpointStore.save()` has no error handling — `path.write_text()` can raise `OSError` / `PermissionError` with no contextual message. | Wrap in `try/except OSError as exc: raise TrainerError(f"Checkpoint save failed for epoch {epoch}.", context={"path": str(path)}, hint="Check disk space and directory permissions.") from exc`. |
| P2 | 15 | operator_trainer.py:83 | `FileCheckpointStore.load()` reads JSON but `json.loads()` can raise `json.JSONDecodeError` if a checkpoint file is corrupt. | Wrap with `except (json.JSONDecodeError, OSError) as exc: raise TrainerError(...) from exc`. |
| P2 | 16 | verl.py:232 | `object.__setattr__(current, "extra", merged_extra)` — bypasses the `frozen=True` dataclass contract on `VeRLConfig`, which is surprising and fragile. | Add a comment flagging this as intentional, or (better) make the merged config immutable by reassigning `self._config = VeRLConfig(**{..., "extra": merged_extra})`. |
| P2 | 17 | verl.py:584–585 | `except ImportError: return base` in `_build_verl_omega_config` — silently falls back to a plain dict when OmegaConf is missing, but VeRL may not accept a plain dict. No log, no warning. | Add `logger.warning("omegaconf not available; passing raw dict to VeRL. Install omegaconf if training fails.")`. |
| P2 | 18 | verl.py:654–655 | `except ImportError: default_score_fn = None` in `_verl_evaluate` — swallowed silently. | Add `logger.debug("verl.utils.reward_score.gsm8k not found; using fallback scoring.")`. |
| P2 | 19 | synthesis.py:281 | `SynthesisError` class defined but `SynthesisPipeline.run()` never raises it — empty-after-filter returns a silent `SynthesisResult` with no items and no warning. | Log a warning (`logger.warning("SynthesisPipeline: all items filtered out; returning empty result.")`) and/or raise `SynthesisError("All source items were filtered out.", hint="Lower min_score or provide more diverse source data.")`. |

---

## Resilience gaps

| File:line | Issue | System |
|-----------|-------|--------|
| verl.py:327–328 | `run_in_executor(None, _run_fit)` with no timeout — hangs indefinitely on crashed GPU job or hung Ray cluster. No cancellation propagation: if the outer task is cancelled, the thread cannot be cancelled and the `await` may silently complete after the cancel. | VeRL/GPU |
| verl.py:387–388 | Same issue in `evaluate()`. | VeRL/GPU |
| operator_trainer.py:319–348 | Long training loop: no timeout, no per-epoch heartbeat/watchdog. If `eval_fn` hangs, the loop stalls indefinitely with no `CancelledError` path. | Operator training |
| operator_trainer.py:65 | `FileCheckpointStore.save()` writes with no atomicity (not write-then-rename). A mid-write crash leaves a corrupt partial JSON that `load()` will fail on. | Disk I/O |
| operator_trainer.py:311–315 | Resume-from-checkpoint path: if checkpoint file exists but is corrupt/partial, `_restore_checkpoint()` silently proceeds with default/empty operator states — checkpoint load failure is not surfaced. | Disk I/O |
| verl.py:303–304 | `reward_spec.resolve()` called in training hot path — if the module referenced by `module_path` is missing, `TrainerError` fires mid-run rather than at validation time. Should be pre-checked in `check_reward`. | Dependency |

---

## Effort estimate

**M** — The experimental/pre-release status lowers urgency, but 10 raw `ValueError` escapes, one P0 ImportError that bypasses ExoError, and two unguarded `run_in_executor` calls in the GPU training path make this a medium lift: roughly a day of targeted fixes with test coverage.
