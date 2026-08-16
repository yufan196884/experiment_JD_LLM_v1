from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from datasets import Dataset, DatasetDict, load_dataset


# =============================================================================
# Constants
# =============================================================================

GRID_SIZE = 9
CENTER = (4, 4)

CORNERS = (
    (0, 0),
    (0, 8),
    (8, 0),
    (8, 8),
)

DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

Coord = tuple[int, int]


# =============================================================================
# Maze representation
# =============================================================================

@dataclass(frozen=True)
class MazeSpec:
    """
    Complete specification of a generated Maze instance.

    This contains everything required later to:
      - render the prompt;
      - reconstruct the maze;
      - evaluate generated routes;
      - compute the four reward components.
    """

    seed: int

    open_cells: frozenset[Coord]

    start: Coord
    end: Coord

    gold_corner: Coord
    diamond_corner: Coord

    gold_cells: frozenset[Coord]
    diamond_cells: frozenset[Coord]
    lava_cells: frozenset[Coord]

    bonus_cell: Coord

    step_budget: int
    n_cycles: int

    # Useful generation diagnostics / invariants.
    via_gold: int
    via_diamond: int
    via_both_gold_first: int
    via_both_diamond_first: int

    def render_grid(self) -> list[str]:
        """
        Return a human-readable rendering of the maze.

        Symbols:
            S = start
            E = end
            G = gold
            D = diamond
            L = lava
            B = bonus
            . = open cell
            # = wall
        """
        rows: list[str] = []

        for row in range(GRID_SIZE):
            symbols: list[str] = []

            for col in range(GRID_SIZE):
                cell = (row, col)

                if cell == self.start:
                    symbol = "S"
                elif cell == self.end:
                    symbol = "E"
                elif cell == self.bonus_cell:
                    symbol = "B"
                elif cell in self.gold_cells:
                    symbol = "G"
                elif cell in self.diamond_cells:
                    symbol = "D"
                elif cell in self.lava_cells:
                    symbol = "L"
                elif cell in self.open_cells:
                    symbol = "."
                else:
                    symbol = "#"

                symbols.append(symbol)

            rows.append(" ".join(symbols))

        return rows

    def to_record(self) -> dict:
        """
        Convert this MazeSpec into a JSON / Hugging Face Dataset-compatible
        dictionary.

        Sets and tuples are converted into lists because JSON and Arrow do not
        represent Python sets/tuples directly.
        """
        return {
            "id": f"maze-{self.seed}",
            "seed": self.seed,

            "grid": self.render_grid(),

            "open_cells": _coordinates_to_lists(self.open_cells),

            "start": list(self.start),
            "end": list(self.end),

            "gold_corner": list(self.gold_corner),
            "diamond_corner": list(self.diamond_corner),

            "gold_cells": _coordinates_to_lists(self.gold_cells),
            "diamond_cells": _coordinates_to_lists(self.diamond_cells),
            "lava_cells": _coordinates_to_lists(self.lava_cells),

            "bonus_cell": list(self.bonus_cell),

            "step_budget": self.step_budget,

            "n_gold": len(self.gold_cells),
            "n_diamond": len(self.diamond_cells),
            "n_lava": len(self.lava_cells),

            "n_cycles": self.n_cycles,

            "via_gold": self.via_gold,
            "via_diamond": self.via_diamond,
            "via_both_gold_first": self.via_both_gold_first,
            "via_both_diamond_first": self.via_both_diamond_first,
        }


def _coordinates_to_lists(
    coordinates: Iterable[Coord],
) -> list[list[int]]:
    """Convert coordinates into stable JSON-compatible lists."""
    return [list(cell) for cell in sorted(coordinates)]


def maze_from_record(record: dict) -> MazeSpec:
    """
    Reconstruct a MazeSpec from a Hugging Face Dataset row.

    This will later be useful in rewards.py:

        maze = maze_from_record(example)
        rewards = simulate_route(maze, moves)
    """

    def coord(value: Sequence[int]) -> Coord:
        if len(value) != 2:
            raise ValueError(f"Invalid coordinate: {value}")

        return int(value[0]), int(value[1])

    def coord_set(
        values: Sequence[Sequence[int]],
    ) -> frozenset[Coord]:
        return frozenset(coord(value) for value in values)

    return MazeSpec(
        seed=int(record["seed"]),
        open_cells=coord_set(record["open_cells"]),

        start=coord(record["start"]),
        end=coord(record["end"]),

        gold_corner=coord(record["gold_corner"]),
        diamond_corner=coord(record["diamond_corner"]),

        gold_cells=coord_set(record["gold_cells"]),
        diamond_cells=coord_set(record["diamond_cells"]),
        lava_cells=coord_set(record["lava_cells"]),

        bonus_cell=coord(record["bonus_cell"]),

        step_budget=int(record["step_budget"]),
        n_cycles=int(record["n_cycles"]),

        via_gold=int(record["via_gold"]),
        via_diamond=int(record["via_diamond"]),

        via_both_gold_first=int(
            record["via_both_gold_first"]
        ),
        via_both_diamond_first=int(
            record["via_both_diamond_first"]
        ),
    )


# =============================================================================
# Grid helpers
# =============================================================================

def _all_cells() -> Iterator[Coord]:
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            yield row, col


def _neighbors(cell: Coord) -> Iterator[Coord]:
    row, col = cell

    for row_delta, col_delta in DIRECTIONS.values():
        next_row = row + row_delta
        next_col = col + col_delta

        if (
            0 <= next_row < GRID_SIZE
            and 0 <= next_col < GRID_SIZE
        ):
            yield next_row, next_col


def _bfs_distance(
    open_cells: set[Coord] | frozenset[Coord],
    start: Coord,
    end: Coord,
    blocked: set[Coord] | frozenset[Coord] = frozenset(),
) -> int | None:
    """
    Find the shortest path distance between two cells.

    Returns None if no valid path exists.
    """
    if (
        start not in open_cells
        or end not in open_cells
        or start in blocked
        or end in blocked
    ):
        return None

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        cell, distance = queue.popleft()

        if cell == end:
            return distance

        for neighbor in _neighbors(cell):
            if (
                neighbor in open_cells
                and neighbor not in blocked
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return None


def _required_distance(
    open_cells: set[Coord],
    start: Coord,
    end: Coord,
) -> int:
    distance = _bfs_distance(
        open_cells,
        start,
        end,
    )

    if distance is None:
        raise ValueError(
            f"No path from {start} to {end}."
        )

    return distance


def _manhattan_ball(
    center: Coord,
    radius: int,
) -> list[Coord]:
    """
    Return every cell within the specified Manhattan distance.
    """
    center_row, center_col = center

    return [
        cell
        for cell in _all_cells()
        if (
            abs(cell[0] - center_row)
            + abs(cell[1] - center_col)
            <= radius
        )
    ]


# =============================================================================
# Maze construction
# =============================================================================

def _carve_prim_tree(
    rng: random.Random,
) -> set[Coord]:
    """
    Construct the initial maze using the Prim-style procedure from the
    VPO Maze experiment.

    Begin with an all-wall grid and one randomly chosen open cell.
    Repeatedly consider frontier walls. A wall becomes open iff it has
    exactly one currently-open four-neighbor.
    """
    seed_cell = (
        rng.randrange(GRID_SIZE),
        rng.randrange(GRID_SIZE),
    )

    open_cells = {seed_cell}

    frontier = list(_neighbors(seed_cell))
    frontier_set = set(frontier)

    while frontier:
        index = rng.randrange(len(frontier))

        cell = frontier.pop(index)
        frontier_set.remove(cell)

        open_neighbor_count = sum(
            neighbor in open_cells
            for neighbor in _neighbors(cell)
        )

        if open_neighbor_count != 1:
            continue

        open_cells.add(cell)

        for neighbor in _neighbors(cell):
            if (
                neighbor not in open_cells
                and neighbor not in frontier_set
            ):
                frontier.append(neighbor)
                frontier_set.add(neighbor)

    return open_cells


def _inject_cycles(
    open_cells: set[Coord],
    rng: random.Random,
    n_cycles: int,
) -> bool:
    """
    Open additional walls in order to introduce cycles and alternative routes.

    A candidate wall must have at least two currently-open neighbors.
    """
    for _ in range(n_cycles):
        eligible_walls = [
            cell
            for cell in _all_cells()
            if (
                cell not in open_cells
                and sum(
                    neighbor in open_cells
                    for neighbor in _neighbors(cell)
                )
                >= 2
            )
        ]

        if not eligible_walls:
            return False

        open_cells.add(
            rng.choice(eligible_walls)
        )

    return True


def _sample_open_cells(
    *,
    open_cells: set[Coord],
    candidates: Sequence[Coord],
    count: int,
    forbidden: set[Coord],
    rng: random.Random,
) -> set[Coord] | None:
    """
    Sample `count` currently-open cells from candidates while avoiding
    forbidden locations.
    """
    available = [
        cell
        for cell in candidates
        if (
            cell in open_cells
            and cell not in forbidden
        )
    ]

    if len(available) < count:
        return None

    return set(
        rng.sample(available, count)
    )


def generate_maze(
    seed: int,
    *,
    check_both_item_orders: bool = True,
) -> MazeSpec | None:
    """
    Generate one candidate maze.

    The function returns None when the candidate does not satisfy the Maze
    experiment's rejection criteria.

    Dataset generation should therefore repeatedly call this function using
    successive seeds until enough surviving mazes have been obtained.
    """
    rng = random.Random(seed)

    # ---------------------------------------------------------------------
    # 1. Generate the base maze.
    # ---------------------------------------------------------------------

    open_cells = _carve_prim_tree(rng)

    # Add between 18 and 28 openings to create cycles.
    n_cycles = rng.randint(18, 28)

    if not _inject_cycles(
        open_cells,
        rng,
        n_cycles,
    ):
        return None

    # The experiment requires all four corners and the center to be traversable.
    if not all(
        corner in open_cells
        for corner in CORNERS
    ):
        return None

    if CENTER not in open_cells:
        return None

    # ---------------------------------------------------------------------
    # 2. Choose start, end, gold corner, diamond corner.
    # ---------------------------------------------------------------------

    # S and E occupy diagonally-opposite corners.
    if rng.randrange(2) == 0:
        start = (0, 0)
        end = (8, 8)

        item_corners = [
            (0, 8),
            (8, 0),
        ]

    else:
        start = (0, 8)
        end = (8, 0)

        item_corners = [
            (0, 0),
            (8, 8),
        ]

    # Randomly choose which remaining corner is Gold vs Diamond.
    rng.shuffle(item_corners)

    gold_corner, diamond_corner = item_corners

    # ---------------------------------------------------------------------
    # 3. Compute path lengths used to determine the movement budget.
    # ---------------------------------------------------------------------

    start_to_gold = _required_distance(
        open_cells,
        start,
        gold_corner,
    )

    start_to_diamond = _required_distance(
        open_cells,
        start,
        diamond_corner,
    )

    gold_to_diamond = _required_distance(
        open_cells,
        gold_corner,
        diamond_corner,
    )

    gold_to_end = _required_distance(
        open_cells,
        gold_corner,
        end,
    )

    diamond_to_end = _required_distance(
        open_cells,
        diamond_corner,
        end,
    )

    via_gold = (
        start_to_gold
        + gold_to_end
    )

    via_diamond = (
        start_to_diamond
        + diamond_to_end
    )

    via_both_gold_first = (
        start_to_gold
        + gold_to_diamond
        + diamond_to_end
    )

    via_both_diamond_first = (
        start_to_diamond
        + gold_to_diamond
        + gold_to_end
    )

    # Budget allows reaching either individual resource region,
    # plus an additional seven moves.
    step_budget = (
        max(via_gold, via_diamond)
        + 7
    )

    # A route visiting both resource corners should not fit.
    if via_both_gold_first <= step_budget:
        return None

    if (
        check_both_item_orders
        and via_both_diamond_first <= step_budget
    ):
        return None

    # ---------------------------------------------------------------------
    # 4. Place resource/hazard cells.
    # ---------------------------------------------------------------------

    n_gold = rng.randint(3, 5)
    n_diamond = rng.randint(3, 5)
    n_lava = rng.randint(3, 5)

    forbidden = {
        start,
        end,
        CENTER,
    }

    # Gold is placed around the Gold corner.
    gold_cells = _sample_open_cells(
        open_cells=open_cells,
        candidates=_manhattan_ball(
            gold_corner,
            radius=2,
        ),
        count=n_gold,
        forbidden=forbidden,
        rng=rng,
    )

    if gold_cells is None:
        return None

    forbidden.update(gold_cells)

    # Diamonds are placed around the Diamond corner.
    diamond_cells = _sample_open_cells(
        open_cells=open_cells,
        candidates=_manhattan_ball(
            diamond_corner,
            radius=2,
        ),
        count=n_diamond,
        forbidden=forbidden,
        rng=rng,
    )

    if diamond_cells is None:
        return None

    forbidden.update(diamond_cells)

    # Lava is placed within the central 5x5 area.
    interior_cells = [
        (row, col)
        for row in range(2, 7)
        for col in range(2, 7)
    ]

    lava_cells = _sample_open_cells(
        open_cells=open_cells,
        candidates=interior_cells,
        count=n_lava,
        forbidden=forbidden,
        rng=rng,
    )

    if lava_cells is None:
        return None

    # ---------------------------------------------------------------------
    # 5. Require at least one lava-free S -> E path within the budget.
    # ---------------------------------------------------------------------

    lava_avoiding_distance = _bfs_distance(
        open_cells,
        start,
        end,
        blocked=lava_cells,
    )

    if (
        lava_avoiding_distance is None
        or lava_avoiding_distance > step_budget
    ):
        return None

    return MazeSpec(
        seed=seed,
        open_cells=frozenset(open_cells),

        start=start,
        end=end,

        gold_corner=gold_corner,
        diamond_corner=diamond_corner,

        gold_cells=frozenset(gold_cells),
        diamond_cells=frozenset(diamond_cells),
        lava_cells=frozenset(lava_cells),

        bonus_cell=CENTER,

        step_budget=step_budget,
        n_cycles=n_cycles,

        via_gold=via_gold,
        via_diamond=via_diamond,

        via_both_gold_first=via_both_gold_first,
        via_both_diamond_first=via_both_diamond_first,
    )


# =============================================================================
# Dataset generation
# =============================================================================

def generate_surviving_mazes(
    *,
    start_seed: int,
    count: int,
    check_both_item_orders: bool = True,
    max_attempts: int = 1_000_000,
) -> list[MazeSpec]:
    """
    Generate the first `count` valid mazes obtained by scanning successive
    candidate seeds.

    Invalid candidates are discarded.
    """
    if count < 0:
        raise ValueError(
            "count must be non-negative."
        )

    mazes: list[MazeSpec] = []

    candidate_seed = start_seed
    attempts = 0

    while len(mazes) < count:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Only generated {len(mazes)} valid mazes after "
                f"{max_attempts} candidate seeds."
            )

        maze = generate_maze(
            candidate_seed,
            check_both_item_orders=check_both_item_orders,
        )

        if maze is not None:
            mazes.append(maze)

        candidate_seed += 1
        attempts += 1

    return mazes


def build_maze_dataset(
    train_size: int = 1000,
    test_size: int = 100,
    *,
    train_start_seed: int = 42,
    test_start_seed: int = 4242,
    check_both_item_orders: bool = True,
) -> DatasetDict:
    """
    Construct the complete Maze DatasetDict.

    Defaults reproduce the split sizes and starting seeds used by the existing
    VPO baseline implementation.
    """
    train_mazes = generate_surviving_mazes(
        start_seed=train_start_seed,
        count=train_size,
        check_both_item_orders=check_both_item_orders,
    )

    test_mazes = generate_surviving_mazes(
        start_seed=test_start_seed,
        count=test_size,
        check_both_item_orders=check_both_item_orders,
    )

    train_records = [
        maze.to_record()
        for maze in train_mazes
    ]

    test_records = [
        maze.to_record()
        for maze in test_mazes
    ]

    return DatasetDict(
        {
            "train": Dataset.from_list(
                train_records
            ),
            "test": Dataset.from_list(
                test_records
            ),
        }
    )


# =============================================================================
# Saving / loading
# =============================================================================

def save_maze_dataset(
    dataset: DatasetDict,
    output_dir: str | Path,
) -> None:
    """
    Save the Maze dataset as train.jsonl and test.jsonl.
    """
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    dataset["train"].to_json(
        str(train_path)
    )

    dataset["test"].to_json(
        str(test_path)
    )


def generate_and_save_maze_dataset(
    output_dir: str | Path,
    *,
    train_size: int = 1000,
    test_size: int = 100,
    train_start_seed: int = 42,
    test_start_seed: int = 4242,
    check_both_item_orders: bool = True,
) -> DatasetDict:
    """
    Convenience wrapper that generates and saves the Maze dataset.
    """
    dataset = build_maze_dataset(
        train_size=train_size,
        test_size=test_size,
        train_start_seed=train_start_seed,
        test_start_seed=test_start_seed,
        check_both_item_orders=check_both_item_orders,
    )

    save_maze_dataset(
        dataset,
        output_dir,
    )

    return dataset


def load_maze_dataset(
    data_dir: str | Path,
) -> DatasetDict:
    """
    Load train.jsonl and test.jsonl as a Hugging Face DatasetDict.
    """
    data_dir = Path(data_dir)

    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"

    missing = [
        path
        for path in (train_path, test_path)
        if not path.exists()
    ]

    if missing:
        missing_text = ", ".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            f"Maze dataset files not found: {missing_text}. "
            "Generate the dataset first."
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "test": str(test_path),
        },
    )

    return dataset