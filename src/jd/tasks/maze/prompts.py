from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .dataset import MazeSpec, maze_from_record


# =============================================================================
# System prompt
# =============================================================================

MAZE_SYSTEM_PROMPT_TEMPLATE = (
    "You solve maze-navigation problems.\n"
    "Output exactly {num_routes} routes.\n"
    "Begin immediately with <route_1>.\n"
    "Inside each <route_i>...</route_i> tag, write only "
    "space-separated UP, DOWN, LEFT, and RIGHT moves.\n"
    "Every route tag must have a closing tag."
)


def get_maze_system_prompt(
    num_routes: int = 3,
) -> str:
    """
    Return the Maze system prompt.

    The baseline VPO experiments use three routes, but this function is kept
    configurable so that num_routes can be changed in ablation experiments.
    """
    if num_routes <= 0:
        raise ValueError("num_routes must be positive.")

    return MAZE_SYSTEM_PROMPT_TEMPLATE.format(
        num_routes=num_routes
    )


# =============================================================================
# User prompt
# =============================================================================

def render_maze_user_prompt(
    maze: MazeSpec,
    *,
    num_routes: int = 3,
) -> str:
    """
    Render the task-specific user prompt for one Maze instance.

    This contains:
      - the rendered 9x9 grid;
      - the movement rules;
      - the reward-relevant objects;
      - the step budget;
      - the required route tags.

    Output-format enforcement is primarily handled by the system prompt.
    """
    if num_routes <= 0:
        raise ValueError("num_routes must be positive.")

    grid_text = "\n".join(maze.render_grid())

    route_tags = ", ".join(
        f"<route_{index}>...</route_{index}>"
        for index in range(1, num_routes + 1)
    )

    return f"""Navigate a 9x9 maze from S to E. Collect gold and diamonds, avoid lava.

Grid:
{grid_text}

- Move: UP, DOWN, LEFT, RIGHT. # is a wall -- you cannot enter it.
- Do not leave the grid.
- Collect: G (Gold), D (Diamond), B (Bonus) tiles by stepping on them.
- Avoid: L (Lava) tiles. Stepping on lava costs you.
- Visiting a B cell multiplies your other scores -- explore!
- You MUST reach E. If you do not reach E, your score is zero everywhere.
- Items only count if collected BEFORE you reach E; the trajectory ends at E.
- You have {maze.step_budget} steps per route.

This maze has {len(maze.gold_cells)} Gold, \
{len(maze.diamond_cells)} Diamond, \
{len(maze.lava_cells)} Lava, and 1 Bonus tile.

Provide {num_routes} genuinely different routes from S to E.
Each route must be a space-separated sequence of UP/DOWN/LEFT/RIGHT moves.

Inside each route tag, put ONLY moves.
Required route tags: {route_tags}
"""


# =============================================================================
# Dataset-record -> prompt conversion
# =============================================================================

def build_maze_messages(
    record: Mapping[str, Any],
    *,
    num_routes: int = 3,
) -> list[dict[str, str]]:
    """
    Convert one raw Maze dataset record into chat messages.

    Example output:

        [
            {
                "role": "system",
                "content": "...",
            },
            {
                "role": "user",
                "content": "...",
            },
        ]
    """
    maze = maze_from_record(dict(record))

    return [
        {
            "role": "system",
            "content": get_maze_system_prompt(
                num_routes=num_routes
            ),
        },
        {
            "role": "user",
            "content": render_maze_user_prompt(
                maze,
                num_routes=num_routes,
            ),
        },
    ]


def add_maze_prompt(
    record: Mapping[str, Any],
    *,
    num_routes: int = 3,
) -> dict[str, Any]:
    """
    Return a copy of a dataset record with a `prompt` field added.

    This format is convenient for TRL / GRPO trainers.
    """
    output = dict(record)

    output["prompt"] = build_maze_messages(
        record,
        num_routes=num_routes,
    )

    return output