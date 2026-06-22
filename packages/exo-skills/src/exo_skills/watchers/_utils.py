"""Shared utilities for skill watchers."""

from __future__ import annotations

from exo.skills import Skill, SkillChangeEvent, _skill_fingerprint


def _diff_snapshots(
    old: dict[str, Skill],
    new: dict[str, Skill],
    source_path: str,
) -> list[SkillChangeEvent]:
    """Compare two skill snapshots and return a list of change events.

    Args:
        old: Previous snapshot mapping skill name to Skill.
        new: Current snapshot mapping skill name to Skill.
        source_path: The source URL or path being watched (for event metadata).

    Returns:
        List of :class:`SkillChangeEvent` describing added, removed, and
        modified skills.  Returns an empty list when the snapshots are
        identical.
    """
    events: list[SkillChangeEvent] = []

    old_names = set(old)
    new_names = set(new)

    # Added skills
    for name in sorted(new_names - old_names):
        events.append(
            SkillChangeEvent(
                kind="added",
                skill_name=name,
                skill=new[name],
                source_path=source_path,
            )
        )

    # Removed skills
    for name in sorted(old_names - new_names):
        events.append(
            SkillChangeEvent(
                kind="removed",
                skill_name=name,
                skill=None,
                source_path=source_path,
            )
        )

    # Modified skills
    for name in sorted(old_names & new_names):
        if _skill_fingerprint(old[name]) != _skill_fingerprint(new[name]):
            events.append(
                SkillChangeEvent(
                    kind="modified",
                    skill_name=name,
                    skill=new[name],
                    source_path=source_path,
                )
            )

    return events
