"""Evaluation framework REST API.

Provides CRUD for evaluation suites, running evaluations against agents,
and retrieving scored results.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.pagination import paginate
from exo_web.routes.auth import get_current_user
from exo_web.sanitize import sanitize_html
from exo_web.services.evaluators import EVALUATORS, run_evaluator
from exo_web.services.safety import (
    DEFAULT_POLICY,
    SAFETY_CATEGORIES,
    generate_redteam_cases,
    judge_safety,
)

router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluations"])
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestCase(BaseModel):
    input: str = Field(..., min_length=1, description="Input text or data")
    expected: str = Field("", description="Expected")
    evaluator: str = Field("exact_match", description="Evaluator")


class EvaluationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    agent_id: str = Field(..., min_length=1, description="Associated agent identifier")
    test_cases: list[TestCase] = Field(default_factory=list, description="Test cases")


class EvaluationUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    agent_id: str | None = Field(None, min_length=1, description="Associated agent identifier")
    test_cases: list[TestCase] | None = Field(None, description="Test cases")


class EvaluationResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    agent_id: str = Field(description="Associated agent identifier")
    test_cases_json: str = Field(description="Test cases json")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class EvalResultResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    evaluation_id: str = Field(description="Associated evaluation identifier")
    run_at: str = Field(description="Run at")
    results_json: str = Field(description="Results json")
    overall_score: float = Field(description="Overall score")
    pass_rate: float = Field(description="Pass rate")


class SafetyRunCreate(BaseModel):
    categories: list[str] | None = Field(None, description="Categories")
    mode: str = Field("preset", pattern="^(preset|redteam)$", description="Mode")
    policy: dict[str, Any] | None = Field(None, description="Policy")


class SafetyRunResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    evaluation_id: str = Field(description="Associated evaluation identifier")
    run_at: str = Field(description="Run at")
    mode: str = Field(description="Mode")
    policy_json: str = Field(description="Policy json")
    results_json: str = Field(description="Results json")
    category_scores_json: str = Field(description="Category scores json")
    overall_score: float = Field(description="Overall score")
    pass_rate: float = Field(description="Pass rate")
    flagged_count: int = Field(description="Flagged count")
    total_count: int = Field(description="Total count")
    # Transparency fields: indicates whether AI judging was available.
    # Present on /run responses; absent on historical /safety-results rows.
    ai_judge_available: bool | None = Field(
        None, description="Whether AI judging was used (False = heuristic fallback)"
    )
    ai_judge_error: str | None = Field(
        None, description="Reason AI judging was unavailable, if applicable"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


async def _send_to_agent(agent_id: str, user_id: str, message: str) -> str:
    """Send a message to an agent and return the text response."""
    from exo_web.services.agent_runtime import _load_agent_row, _resolve_provider

    row = await _load_agent_row(agent_id)
    provider_type = row.get("model_provider", "")
    model_name = row.get("model_name", "")
    if not provider_type or not model_name:
        return "[error: agent has no model configured]"

    try:
        provider = await _resolve_provider(provider_type, model_name, user_id)
        instructions = row.get("instructions", "")
        messages: list[dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": message})
        resp = await provider.complete(messages=messages, model=model_name)
        return resp.content
    except Exception as exc:
        return f"[error: {exc}]"


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("")
async def list_evaluations(
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """List evaluation suites for the current user."""
    async with get_db() as db:
        result = await paginate(
            db,
            table="evaluations",
            conditions=["user_id = ?"],
            params=[user["id"]],
            cursor=cursor,
            limit=limit,
            row_mapper=_row_to_dict,
        )
        return result.model_dump()


@router.post("", response_model=EvaluationResponse, status_code=201)
async def create_evaluation(
    body: EvaluationCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new evaluation suite."""
    eval_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Validate agent exists
    async with get_db() as db:
        cur = await db.execute(
            "SELECT id FROM agents WHERE id = ? AND user_id = ?",
            (body.agent_id, user["id"]),
        )
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        test_cases_json = json.dumps([tc.model_dump() for tc in body.test_cases])

        await db.execute(
            """
            INSERT INTO evaluations (id, name, agent_id, test_cases_json, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id,
                sanitize_html(body.name),
                body.agent_id,
                test_cases_json,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/safety/categories")
async def list_safety_categories(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return the list of safety categories and default policy."""
    categories = {
        key: {
            "label": val["label"],
            "description": val["description"],
            "case_count": len(val["test_cases"]),
        }
        for key, val in SAFETY_CATEGORIES.items()
    }
    return {"categories": categories, "default_policy": DEFAULT_POLICY}


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single evaluation suite."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _row_to_dict(row)


@router.put("/{evaluation_id}", response_model=EvaluationResponse)
async def update_evaluation(
    evaluation_id: str,
    body: EvaluationUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update an evaluation suite."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        updates: dict[str, Any] = {}
        if body.name is not None:
            updates["name"] = sanitize_html(body.name)
        if body.agent_id is not None:
            # Validate agent
            cur = await db.execute(
                "SELECT id FROM agents WHERE id = ? AND user_id = ?",
                (body.agent_id, user["id"]),
            )
            if await cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Agent not found")
            updates["agent_id"] = body.agent_id
        if body.test_cases is not None:
            updates["test_cases_json"] = json.dumps([tc.model_dump() for tc in body.test_cases])

        if not updates:
            raise HTTPException(status_code=422, detail="No fields to update")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), evaluation_id]
        await db.execute(
            f"UPDATE evaluations SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{evaluation_id}", status_code=204)
async def delete_evaluation(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an evaluation suite and all its results."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        await db.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


@router.post("/{evaluation_id}/run", response_model=EvalResultResponse)
async def run_evaluation(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Run all test cases against the linked agent and score the responses."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    evaluation = _row_to_dict(row)

    try:
        test_cases: list[dict[str, Any]] = json.loads(evaluation["test_cases_json"])
    except (json.JSONDecodeError, TypeError):
        test_cases = []

    if not test_cases:
        raise HTTPException(status_code=422, detail="No test cases defined")

    agent_id = evaluation["agent_id"]
    results: list[dict[str, Any]] = []
    total_score = 0.0
    pass_count = 0

    for tc in test_cases:
        input_msg = tc.get("input", "")
        expected = tc.get("expected", "")
        evaluator_type = tc.get("evaluator", "exact_match")

        # Validate evaluator type
        if evaluator_type not in EVALUATORS:
            evaluator_type = "exact_match"

        # Send to agent
        actual = await _send_to_agent(agent_id, user["id"], input_msg)

        # Score the response
        kwargs: dict[str, Any] = {}
        if evaluator_type == "llm_as_judge":
            from exo_web.services.agent_runtime import _resolve_provider

            async with get_db() as db2:
                cur = await db2.execute(
                    "SELECT model_provider, model_name FROM agents WHERE id = ?",
                    (agent_id,),
                )
                agent_row = await cur.fetchone()

            if agent_row:
                agent_data = dict(agent_row)
                kwargs["provider_resolver"] = _resolve_provider
                kwargs["provider_type"] = agent_data.get("model_provider", "")
                kwargs["model_name"] = agent_data.get("model_name", "")
                kwargs["user_id"] = user["id"]

        score = await run_evaluator(evaluator_type, expected, actual, **kwargs)
        passed = score >= 0.5
        if passed:
            pass_count += 1
        total_score += score

        results.append(
            {
                "input": input_msg,
                "expected": expected,
                "actual": actual,
                "evaluator": evaluator_type,
                "score": round(score, 4),
                "passed": passed,
            }
        )

    num_cases = len(test_cases)
    overall_score = round(total_score / num_cases, 4) if num_cases else 0.0
    pass_rate = round(pass_count / num_cases, 4) if num_cases else 0.0

    # Persist the result
    result_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO eval_results (id, evaluation_id, run_at, results_json, overall_score, pass_rate)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                evaluation_id,
                now,
                json.dumps(results),
                overall_score,
                pass_rate,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM eval_results WHERE id = ?", (result_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Safety evaluation
# ---------------------------------------------------------------------------


@router.post("/{evaluation_id}/safety-run", response_model=SafetyRunResponse)
async def run_safety_evaluation(
    evaluation_id: str,
    body: SafetyRunCreate | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Run safety evaluation against the agent with adversarial test cases.

    Supports two modes:
    - preset: Uses built-in adversarial test cases per category.
    - redteam: Uses an LLM to generate novel adversarial inputs.
    """
    if body is None:
        body = SafetyRunCreate()

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    evaluation = _row_to_dict(row)
    agent_id = evaluation["agent_id"]

    # Resolve policy
    policy = body.policy if body.policy else DEFAULT_POLICY
    pass_threshold = policy.get("pass_threshold", 0.7)
    category_thresholds = policy.get("category_thresholds", {})
    rules: list[str] = policy.get("rules", DEFAULT_POLICY["rules"])

    # Resolve provider for LLM judging.  If resolution fails we fall back to
    # heuristic judging and surface the degradation to the caller.
    provider = None
    model_name = ""
    ai_judge_available = True
    ai_judge_error: str | None = None
    try:
        from exo_web.services.agent_runtime import _load_agent_row, _resolve_provider

        agent_row = await _load_agent_row(agent_id)
        provider_type = agent_row.get("model_provider", "")
        model_name = agent_row.get("model_name", "")
        if provider_type and model_name:
            provider = await _resolve_provider(provider_type, model_name, user["id"])
        else:
            ai_judge_available = False
            ai_judge_error = "Agent has no model configured; heuristic judging used"
            _log.warning("Safety evaluation %s: %s", evaluation_id, ai_judge_error)
    except Exception as exc:
        ai_judge_available = False
        ai_judge_error = f"Provider resolution failed ({exc!s}); heuristic judging used"
        _log.warning(
            "Safety evaluation %s: provider resolution failed — falling back to heuristic judging: %s",
            evaluation_id,
            exc,
        )

    # Determine which categories to test
    selected_categories = body.categories if body.categories else list(SAFETY_CATEGORIES.keys())
    selected_categories = [c for c in selected_categories if c in SAFETY_CATEGORIES]

    if not selected_categories:
        raise HTTPException(status_code=422, detail="No valid safety categories selected")

    # Gather test cases
    all_results: list[dict[str, Any]] = []
    category_scores: dict[str, dict[str, Any]] = {}

    for cat_key in selected_categories:
        cat_info = SAFETY_CATEGORIES[cat_key]

        # Get test cases: preset or redteam-generated
        if body.mode == "redteam":
            test_cases = await generate_redteam_cases(
                cat_key, count=5, provider=provider, model_name=model_name
            )
        else:
            test_cases = cat_info["test_cases"]

        cat_total = 0.0
        cat_passed = 0
        cat_results: list[dict[str, Any]] = []

        for tc in test_cases:
            user_input = tc["input"]
            concern = tc["concern"]

            # Send adversarial input to agent
            actual = await _send_to_agent(agent_id, user["id"], user_input)

            # Judge the response
            judgment = await judge_safety(
                category=cat_key,
                concern=concern,
                user_input=user_input,
                agent_response=actual,
                rules=rules,
                provider=provider,
                model_name=model_name,
            )

            result_entry = {
                "category": cat_key,
                "category_label": cat_info["label"],
                "input": user_input,
                "concern": concern,
                "response": actual,
                "score": judgment["score"],
                "passed": judgment["passed"],
                "explanation": judgment["explanation"],
            }
            cat_results.append(result_entry)
            cat_total += judgment["score"]
            if judgment["passed"]:
                cat_passed += 1

        cat_count = len(cat_results)
        cat_avg = round(cat_total / cat_count, 4) if cat_count else 0.0
        cat_threshold = category_thresholds.get(cat_key, pass_threshold)
        category_scores[cat_key] = {
            "label": cat_info["label"],
            "score": cat_avg,
            "passed": cat_avg >= cat_threshold,
            "threshold": cat_threshold,
            "total": cat_count,
            "passed_count": cat_passed,
        }
        all_results.extend(cat_results)

    # Compute overall metrics
    total_count = len(all_results)
    flagged_count = sum(1 for r in all_results if not r["passed"])
    overall_score = (
        round(sum(r["score"] for r in all_results) / total_count, 4) if total_count else 0.0
    )
    overall_pass_rate = (
        round(sum(1 for r in all_results if r["passed"]) / total_count, 4) if total_count else 0.0
    )

    # Persist the safety run
    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO safety_runs
                (id, evaluation_id, run_at, mode, policy_json, results_json,
                 category_scores_json, overall_score, pass_rate, flagged_count, total_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                evaluation_id,
                now,
                body.mode,
                json.dumps(policy),
                json.dumps(all_results),
                json.dumps(category_scores),
                overall_score,
                overall_pass_rate,
                flagged_count,
                total_count,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM safety_runs WHERE id = ?", (run_id,))
        row = await cursor.fetchone()
        result = _row_to_dict(row)
        # Surface AI judging availability so callers know whether results used
        # heuristic fallback.  These fields are not persisted to the DB.
        result["ai_judge_available"] = ai_judge_available
        result["ai_judge_error"] = ai_judge_error
        return result


@router.get(
    "/{evaluation_id}/safety-results",
    response_model=list[SafetyRunResponse],
)
async def list_safety_results(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all safety run results for an evaluation suite."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        cursor = await db.execute(
            "SELECT * FROM safety_runs WHERE evaluation_id = ? ORDER BY run_at DESC",
            (evaluation_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@router.get("/{evaluation_id}/results", response_model=list[EvalResultResponse])
async def list_eval_results(
    evaluation_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all result runs for an evaluation suite."""
    async with get_db() as db:
        # Verify ownership
        cursor = await db.execute(
            "SELECT id FROM evaluations WHERE id = ? AND user_id = ?",
            (evaluation_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        cursor = await db.execute(
            "SELECT * FROM eval_results WHERE evaluation_id = ? ORDER BY run_at DESC",
            (evaluation_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
