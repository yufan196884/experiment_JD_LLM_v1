from jd.tasks.maze.parser import (
    has_parsable_route,
    parse_moves,
    parse_route,
)


# =============================================================================
# Low-level move parsing
# =============================================================================

def test_parse_moves():
    assert parse_moves(
        "UP down LEFT right"
    ) == [
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
    ]


def test_parse_moves_rejects_invalid_token():
    assert parse_moves(
        "UP RIGHT JUMP DOWN"
    ) is None


def test_parse_moves_rejects_empty_route():
    assert parse_moves("") is None
    assert parse_moves("   ") is None


# =============================================================================
# Single-route completion parsing
# =============================================================================

def test_parse_route():
    completion = """
    <route>
    UP RIGHT DOWN LEFT
    </route>
    """

    route = parse_route(
        completion
    )

    assert route == [
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
    ]


def test_parse_route_case_insensitive():
    completion = """
    <RoUtE>
    up right DOWN left
    </rOuTe>
    """

    route = parse_route(
        completion
    )

    assert route == [
        "UP",
        "RIGHT",
        "DOWN",
        "LEFT",
    ]


def test_parse_route_allows_surrounding_whitespace():
    completion = """

        <route>
        UP RIGHT DOWN
        </route>

    """

    assert parse_route(
        completion
    ) == [
        "UP",
        "RIGHT",
        "DOWN",
    ]


def test_missing_opening_tag_is_invalid():
    completion = """
    UP RIGHT DOWN
    </route>
    """

    assert parse_route(
        completion
    ) is None


def test_missing_closing_tag_is_invalid():
    completion = """
    <route>
    UP RIGHT DOWN
    """

    assert parse_route(
        completion
    ) is None


def test_empty_route_is_invalid():
    completion = """
    <route>
    </route>
    """

    assert parse_route(
        completion
    ) is None


def test_invalid_move_invalidates_completion():
    completion = """
    <route>
    UP RIGHT JUMP DOWN
    </route>
    """

    assert parse_route(
        completion
    ) is None


# =============================================================================
# Exact output-format enforcement
# =============================================================================

def test_text_before_route_is_allowed():
    completion = """
    Here is the best route:

    <route>
    UP RIGHT DOWN
    </route>
    """

    assert parse_route(completion) == [
        "UP",
        "RIGHT",
        "DOWN",
    ]


def test_text_after_route_is_allowed():
    completion = """
    <route>
    UP RIGHT DOWN
    </route>

    This route avoids lava.
    """

    assert parse_route(completion) == [
        "UP",
        "RIGHT",
        "DOWN",
    ]


def test_multiple_routes_uses_first_route():
    completion = """
    <route>
    UP RIGHT
    </route>

    <route>
    DOWN LEFT
    </route>
    """

    assert parse_route(completion) == [
        "UP",
        "RIGHT",
    ]


def test_old_numbered_route_format_is_invalid():
    completion = """
    <route_1>
    UP RIGHT DOWN
    </route_1>
    """

    assert parse_route(
        completion
    ) is None


# =============================================================================
# Convenience format checker
# =============================================================================

def test_has_parsable_route():
    completion = """
    <route>
    UP RIGHT DOWN LEFT
    </route>
    """

    assert has_parsable_route(
        completion
    )


def test_has_parsable_route_rejects_invalid_completion():
    completion = """
    <route>
    UP TELEPORT RIGHT
    </route>
    """

    assert not has_parsable_route(
        completion
    )