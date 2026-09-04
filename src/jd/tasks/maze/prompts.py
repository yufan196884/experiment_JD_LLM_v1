from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .dataset import MazeSpec, maze_from_record


# =============================================================================
# Preference definition
# =============================================================================

MAZE_PREFERENCE_NAMES = (
    "completion",
    "gold",
    "diamond",
    "lava_avoidance",
)

NUM_MAZE_PREFERENCES = len(MAZE_PREFERENCE_NAMES)


def validate_maze_preference(
    preference: Sequence[float],
) -> tuple[float, float, float, float]:
    """
    Validate and normalize the Python representation of a Maze preference.

    Reward / preference order:

        [
            completion,
            gold,
            diamond,
            lava_avoidance,
        ]

    The preference must lie on the 4-dimensional probability simplex.
    """
    values = tuple(
        float(value)
        for value in preference
    )

    if len(values) != NUM_MAZE_PREFERENCES:
        raise ValueError(
            "Maze preference must contain exactly "
            f"{NUM_MAZE_PREFERENCES} values in the order "
            f"{MAZE_PREFERENCE_NAMES}, got {len(values)}."
        )

    if any(
        not math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "Maze preference values must all be finite."
        )

    if any(
        value < 0.0
        for value in values
    ):
        raise ValueError(
            "Maze preference values must be non-negative."
        )

    total = math.fsum(values)

    if not math.isclose(
        total,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError(
            "Maze preference values must sum to 1. "
            f"Got {total}."
        )

    return values


def render_maze_preference(
    preference: Sequence[float],
) -> str:
    """
    Render the preference vector in the stable machine-readable format
    supplied to the language model.
    """
    values = validate_maze_preference(
        preference
    )

    lines = [
        "<preference>",
    ]

    lines.extend(
        f"{name}: {value:.6f}"
        for name, value in zip(
            MAZE_PREFERENCE_NAMES,
            values,
        )
    )

    lines.append(
        "</preference>"
    )

    return "\n".join(lines)


# =============================================================================
# System prompt
# =============================================================================

MAZE_SYSTEM_PROMPT = """You solve maze-navigation problems.

You will receive a 9x9 maze and a preference vector specifying the relative
importance of four reward objectives:

- completion: reaching the exit E;
- gold: collecting Gold tiles;
- diamond: collecting Diamond tiles;
- lava_avoidance: avoiding Lava tiles.

Higher preference weight means that objective is more important.
The preference weights are non-negative and sum to 1.

Choose exactly one route that best matches the supplied preference.

You must reach E. If you do not reach E within the step budget, every reward
component is zero.

First, reason briefly about the maze, and then output exactly one route.
Begin the rote with <route> and end with </route>.
Inside <route>...</route>, write only a space-separated sequence of:
UP, DOWN, LEFT, RIGHT."""


def get_maze_system_prompt() -> str:
    """Return the preference-conditioned single-route Maze system prompt."""
    return MAZE_SYSTEM_PROMPT


# =============================================================================
# User prompt
# =============================================================================

def render_maze_user_prompt(
    maze: MazeSpec,
    *,
    preference: Sequence[float],
) -> str:
    """
    Render one preference-conditioned Maze task.

    One call corresponds to:

        (Maze x, preference omega) -> one route
    """
    grid_text = "\n".join(
        maze.render_grid()
    )

    preference_text = render_maze_preference(
        preference
    )

    return f"""Navigate the following 9x9 maze from S to E.

Preference:
{preference_text}

Higher preference weight means that objective is more important.
Use these weights to decide how to trade off Gold collection, Diamond
collection, and Lava avoidance while reaching E.

Grid:
{grid_text}

Maze symbols:
- S: starting position.
- E: exit.
- G: Gold.
- D: Diamond.
- L: Lava.
- B: Bonus marker. It has no reward effect in this experiment.
- .: open cell.
- #: wall.

Movement and reward rules:
- Valid moves are UP, DOWN, LEFT, and RIGHT.
- You cannot move through walls or leave the grid.
- You have at most {maze.step_budget} moves.
- Reaching E gives the completion reward.
- Gold reward is based on distinct G cells collected before reaching E.
- Diamond reward is based on distinct D cells collected before reaching E.
- Lava avoidance is better when fewer distinct L cells are visited.
- Repeated visits to the same G, D, or L cell count only once.
- The trajectory ends immediately when E is reached.
- If E is not reached within the step budget, all four rewards are zero.

This maze contains:
- {len(maze.gold_cells)} Gold cells;
- {len(maze.diamond_cells)} Diamond cells;
- {len(maze.lava_cells)} Lava cells.

After solving the maze, provide your final route using:

<route>
UP RIGHT DOWN LEFT
</route>
"""


# =============================================================================
# Dataset-record -> prompt conversion
# =============================================================================

def build_maze_messages(
    record: Mapping[str, Any],
    *,
    preference: Sequence[float],
) -> list[dict[str, str]]:
    """
    Convert one Maze record and one preference vector into chat messages.

    The same `preference` must later be used by preference-weighted UPGrad
    for the corresponding optimization group.
    """
    maze = maze_from_record(
        dict(record)
    )

    return [
        {
            "role": "system",
            "content": get_maze_system_prompt(),
        },
        {
            "role": "user",
            "content": render_maze_user_prompt(
                maze,
                preference=preference,
            ),
        },
    ]


def add_maze_prompt(
    record: Mapping[str, Any],
    *,
    preference: Sequence[float],
) -> dict[str, Any]:
    """
    Return a copy of a dataset record with a preference-conditioned
    `prompt` field added.
    """
    output = dict(record)

    output["prompt"] = build_maze_messages(
        record,
        preference=preference,
    )

    return output