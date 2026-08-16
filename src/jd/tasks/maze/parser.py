from __future__ import annotations

import re
from typing import Sequence


VALID_MOVES = frozenset({
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
})

MoveSequence = list[str]


def parse_moves(route_text: str) -> MoveSequence | None:
    """
    Parse the contents of a single <route_i>...</route_i> tag.

    The route must consist only of whitespace-separated movement commands:

        UP
        DOWN
        LEFT
        RIGHT

    Parsing is case-insensitive. For example:

        "up RIGHT down"

    becomes:

        ["UP", "RIGHT", "DOWN"]

    Returns None if:
      - the route is empty;
      - any token is not a valid movement command.

    We intentionally reject invalid tokens rather than silently ignoring them,
    because malformed routes should not accidentally receive reward.
    """
    tokens = route_text.strip().split()

    if not tokens:
        return None

    moves = [
        token.upper()
        for token in tokens
    ]

    if any(
        move not in VALID_MOVES
        for move in moves
    ):
        return None

    return moves


def extract_numbered_routes(
    completion: str,
    *,
    num_routes: int = 3,
) -> list[MoveSequence] | None:
    """
    Extract numbered Maze routes from a model completion.

    Expected format:

        <route_1>
        UP RIGHT DOWN ...
        </route_1>

        <route_2>
        ...
        </route_2>

        ...

        <route_K>
        ...
        </route_K>

    Tags are matched case-insensitively and route contents may span multiple
    lines.

    Returns:
        A list of K parsed routes:

            [
                ["UP", "RIGHT", ...],
                ["DOWN", "LEFT", ...],
                ...
            ]

        or None if any required route is missing or malformed.
    """
    if num_routes <= 0:
        raise ValueError(
            "num_routes must be positive."
        )

    if not isinstance(completion, str):
        raise TypeError(
            "completion must be a string."
        )

    routes: list[MoveSequence] = []

    for index in range(1, num_routes + 1):
        pattern = (
            rf"<route_{index}>"
            rf"(.*?)"
            rf"</route_{index}>"
        )

        match = re.search(
            pattern,
            completion,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match is None:
            return None

        moves = parse_moves(
            match.group(1)
        )

        if moves is None:
            return None

        routes.append(moves)

    return routes


def has_valid_route_format(
    completion: str,
    *,
    num_routes: int = 3,
) -> bool:
    """
    Return whether a completion contains all required valid Maze routes.

    This is a convenience function for format rewards and evaluation.
    """
    return (
        extract_numbered_routes(
            completion,
            num_routes=num_routes,
        )
        is not None
    )


def routes_are_distinct(
    routes: Sequence[Sequence[str]],
) -> bool:
    """
    Return True iff all routes are different movement sequences.

    Note that route uniqueness is logically separate from parsing:
    a completion may be syntactically valid while containing duplicate routes.
    """
    normalized = [
        tuple(move.upper() for move in route)
        for route in routes
    ]

    return len(normalized) == len(set(normalized))


def completion_has_distinct_routes(
    completion: str,
    *,
    num_routes: int = 3,
) -> bool:
    """
    Return True iff the completion is valid and all generated routes
    are distinct.
    """
    routes = extract_numbered_routes(
        completion,
        num_routes=num_routes,
    )

    if routes is None:
        return False

    return routes_are_distinct(routes)