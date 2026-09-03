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
    r"\s*<route>\s*(.*?)\s*</route>\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_route(
    completion: str,
) -> MoveSequence | None:
    """
    Parse exactly one route from a model completion.

    Expected completion format:

        <route>
        UP RIGHT DOWN LEFT ...
        </route>

    The model is required to output exactly one route and no other text.
    Therefore, apart from surrounding whitespace, the entire completion
    must consist of one <route>...</route> block.

    Route tags are matched case-insensitively and the route contents may
    span multiple lines.

    Returns:
        A parsed movement sequence such as:

            ["UP", "RIGHT", "DOWN"]

        or None if:
          - the <route> tag is missing;
          - the closing </route> tag is missing;
          - text appears outside the route block;
          - the route is empty;
          - any route token is invalid.
    """
    if not isinstance(completion, str):
        raise TypeError(
            "completion must be a string."
        )

    match = _ROUTE_PATTERN.fullmatch(
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