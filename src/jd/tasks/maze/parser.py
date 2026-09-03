from __future__ import annotations

import re


VALID_MOVES = frozenset({
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
})

MoveSequence = list[str]


# =============================================================================
# Move parsing
# =============================================================================

def parse_moves(
    route_text: str,
) -> MoveSequence | None:
    """
    Parse the contents of one <route>...</route> tag.

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

    Invalid tokens are rejected rather than silently ignored so that
    malformed routes cannot accidentally receive reward.
    """
    if not isinstance(route_text, str):
        raise TypeError(
            "route_text must be a string."
        )

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


# =============================================================================
# Completion parsing
# =============================================================================

_ROUTE_PATTERN = re.compile(
    r"<route>\s*(.*?)\s*</route>",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_route(
    completion: str,
) -> MoveSequence | None:
    """
    Extract the first valid <route>...</route> block from a completion.

    Text before or after the route block is ignored.
    """
    if not isinstance(completion, str):
        raise TypeError(
            "completion must be a string."
        )

    match = _ROUTE_PATTERN.search(
        completion
    )

    if match is None:
        return None

    return parse_moves(
        match.group(1)
    )


def has_valid_route_format(
    completion: str,
) -> bool:
    """
    Return whether the completion contains exactly one valid Maze route.
    """
    return (
        parse_route(completion)
        is not None
    )