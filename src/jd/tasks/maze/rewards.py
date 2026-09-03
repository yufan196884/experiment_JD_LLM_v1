from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .dataset import DIRECTIONS, MazeSpec, maze_from_record
from .parser import parse_route


# =============================================================================
# Reward definitions
# =============================================================================

RewardVector = tuple[float, float, float, float]

MAZE_REWARD_NAMES = (
    "completion",
    "gold",
    "diamond",
    "lava_avoidance",
)

NUM_MAZE_REWARDS = len(MAZE_REWARD_NAMES)

ZERO_REWARD: RewardVector = (
    0.0,
    0.0,
    0.0,
    0.0,
)


# =============================================================================
# Single-route simulation
# =============================================================================

def simulate_route(
    maze: MazeSpec,
    moves: Sequence[str],
) -> RewardVector:
    """
    Execute one generated route inside a Maze and return its reward vector.

    Reward components:

        0. completion
            1 if the route reaches E within the step budget, otherwise 0.

        1. gold
            Fraction of distinct gold cells visited before reaching E.

        2. diamond
            Fraction of distinct diamond cells visited before reaching E.

        3. lava_avoidance
            1 - fraction of distinct lava cells visited before reaching E.

    Important environment rules:

      - Only the first `maze.step_budget` actions are executed.
      - Walking into a wall or outside the grid consumes a step but leaves the
        agent in its current position.
      - The trajectory terminates immediately upon reaching E.
      - Repeated visits to the same Gold / Diamond / Lava cell count only once.
      - If E is not reached, every reward component is zero.
      - The bonus cell currently has no effect, matching the existing
        baseline_vpo implementation.
    """
    position = maze.start

    visited_gold = set()
    visited_diamond = set()
    visited_lava = set()

    # Only actions within the movement budget are executed.
    for raw_move in moves[: maze.step_budget]:
        move = raw_move.upper()

        # parser.py should already prevent this, but keeping this guard makes
        # simulate_route safe to call independently.
        if move not in DIRECTIONS:
            return ZERO_REWARD

        row_delta, col_delta = DIRECTIONS[move]

        next_position = (
            position[0] + row_delta,
            position[1] + col_delta,
        )

        # If next_position is a wall or outside the maze, the agent stays in
        # place. The attempted action still consumes one step.
        if next_position in maze.open_cells:
            position = next_position

        # Reaching E terminates the trajectory immediately.
        if position == maze.end:
            gold_reward = (
                len(visited_gold)
                / len(maze.gold_cells)
            )

            diamond_reward = (
                len(visited_diamond)
                / len(maze.diamond_cells)
            )

            lava_avoidance_reward = 1.0 - (
                len(visited_lava)
                / len(maze.lava_cells)
            )

            return (
                1.0,
                _clamp01(gold_reward),
                _clamp01(diamond_reward),
                _clamp01(lava_avoidance_reward),
            )

        # Items only count before E is reached.
        if position in maze.gold_cells:
            visited_gold.add(position)

        if position in maze.diamond_cells:
            visited_diamond.add(position)

        if position in maze.lava_cells:
            visited_lava.add(position)

        # NOTE:
        # The existing baseline_vpo implementation intentionally gives the
        # bonus cell no effect.

    # Failure to reach E zeros every objective.
    return ZERO_REWARD


def _clamp01(value: float) -> float:
    """Clamp a floating-point reward into [0, 1]."""
    return max(
        0.0,
        min(1.0, float(value)),
    )

# =============================================================================
# Completion -> candidate reward matrix
# =============================================================================

def compute_reward_vector(
    maze: MazeSpec,
    completion: str,
) -> RewardVector:
    """
    Parse one model completion and return its four-dimensional reward vector.

    Expected completion format:

        <route>
        UP RIGHT DOWN ...
        </route>

    Malformed completions receive ZERO_REWARD.
    """
    route = parse_route(
        completion
    )

    if route is None:
        return ZERO_REWARD

    return simulate_route(
        maze,
        route,
    )


def compute_reward_vector_from_record(
    record: Mapping[str, Any],
    completion: str,
) -> RewardVector:
    """
    Convenience wrapper for Hugging Face Dataset rows.
    """
    maze = maze_from_record(
        dict(record)
    )

    return compute_reward_vector(
        maze,
        completion,
    )


def reward_vector_to_dict(
    reward_vector: Sequence[float],
) -> dict[str, float]:
    """
    Convert a four-dimensional Maze reward vector to named components.
    """
    if len(reward_vector) != NUM_MAZE_REWARDS:
        raise ValueError(
            f"Expected {NUM_MAZE_REWARDS} Maze reward components, "
            f"got {len(reward_vector)}."
        )

    return {
        name: float(value)
        for name, value in zip(
            MAZE_REWARD_NAMES,
            reward_vector,
        )
    }


def get_reward_component(
    reward_vector: Sequence[float],
    component: str,
) -> float:
    """
    Extract one named Maze reward component.
    """
    if component not in MAZE_REWARD_NAMES:
        raise ValueError(
            f"Unknown Maze reward component: {component}. "
            f"Expected one of {MAZE_REWARD_NAMES}."
        )

    index = MAZE_REWARD_NAMES.index(
        component
    )

    return float(
        reward_vector[index]
    )