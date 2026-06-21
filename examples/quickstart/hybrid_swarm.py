"""Hybrid swarm — parallel groups, workflow chaining, and nested swarms.

Shows advanced orchestration: a ``ParallelGroup`` runs agents
concurrently, a workflow-mode ``Swarm`` chains them sequentially, and a
``SwarmNode`` nests one swarm inside another.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/hybrid_swarm.py
"""

from exo import Agent, ParallelGroup, Swarm, SwarmNode, run

# --- Parallel research stage ------------------------------------------------
# Two researchers run concurrently; their outputs are joined.

researcher_a = Agent(
    name="history_researcher",
    model="openai:gpt-4o-mini",
    instructions="Provide 2 historical facts about the given topic.",
)

researcher_b = Agent(
    name="science_researcher",
    model="openai:gpt-4o-mini",
    instructions="Provide 2 scientific facts about the given topic.",
)

research_group = ParallelGroup(
    name="research",
    agents=[researcher_a, researcher_b],
)

# --- Serial editing stage ---------------------------------------------------
# Draft then polish, sequentially — a workflow-mode Swarm chains them.

drafter = Agent(
    name="drafter",
    model="openai:gpt-4o-mini",
    instructions="Combine the research into a short article.",
)

editor = Agent(
    name="editor",
    model="openai:gpt-4o-mini",
    instructions="Polish the article for clarity and brevity.",
)

edit_pipeline = SwarmNode(
    swarm=Swarm(agents=[drafter, editor], flow="drafter >> editor", mode="workflow"),
    name="editing",
)

# --- Outer swarm: research (parallel) >> editing (serial) -------------------

pipeline = Swarm(
    agents=[research_group, edit_pipeline],
    flow="research >> editing",
    mode="workflow",
)

# --- Nested swarm inside a larger pipeline ----------------------------------

intro_agent = Agent(
    name="intro",
    model="openai:gpt-4o-mini",
    instructions="Write a one-sentence introduction for the given topic.",
)

nested = SwarmNode(swarm=pipeline, name="article_pipeline")

outer = Swarm(
    agents=[intro_agent, nested],
    flow="intro >> article_pipeline",
    mode="workflow",
)

if __name__ == "__main__":
    result = run.sync(outer, "The Moon")
    print(result.output)
