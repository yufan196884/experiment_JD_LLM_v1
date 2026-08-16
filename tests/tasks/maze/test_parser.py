from jd.tasks.maze.parser import (
    completion_has_distinct_routes,
    extract_numbered_routes,
    parse_moves,
    routes_are_distinct,
)


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


def test_extract_routes():
    completion = """
    <route_1>UP RIGHT DOWN</route_1>
    <route_2>LEFT LEFT DOWN</route_2>
    <route_3>RIGHT UP RIGHT</route_3>
    """

    routes = extract_numbered_routes(
        completion,
        num_routes=3,
    )

    assert routes == [
        ["UP", "RIGHT", "DOWN"],
        ["LEFT", "LEFT", "DOWN"],
        ["RIGHT", "UP", "RIGHT"],
    ]


def test_extract_routes_case_insensitive():
    completion = """
    <ROUTE_1>up right</ROUTE_1>
    <Route_2>left down</Route_2>
    <route_3>right right</route_3>
    """

    routes = extract_numbered_routes(
        completion,
        num_routes=3,
    )

    assert routes == [
        ["UP", "RIGHT"],
        ["LEFT", "DOWN"],
        ["RIGHT", "RIGHT"],
    ]


def test_missing_route_is_invalid():
    completion = """
    <route_1>UP RIGHT</route_1>
    <route_2>LEFT DOWN</route_2>
    """

    assert (
        extract_numbered_routes(
            completion,
            num_routes=3,
        )
        is None
    )


def test_invalid_move_invalidates_completion():
    completion = """
    <route_1>UP RIGHT</route_1>
    <route_2>LEFT JUMP</route_2>
    <route_3>DOWN RIGHT</route_3>
    """

    assert (
        extract_numbered_routes(
            completion,
            num_routes=3,
        )
        is None
    )


def test_route_uniqueness():
    routes = [
        ["UP", "RIGHT"],
        ["DOWN", "RIGHT"],
        ["UP", "RIGHT"],
    ]

    assert not routes_are_distinct(routes)


def test_completion_distinct_routes():
    completion = """
    <route_1>UP RIGHT</route_1>
    <route_2>DOWN RIGHT</route_2>
    <route_3>LEFT DOWN</route_3>
    """

    assert completion_has_distinct_routes(
        completion,
        num_routes=3,
    )