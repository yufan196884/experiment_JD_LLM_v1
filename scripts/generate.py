#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Allow this script to be run directly from the repository root:
#
#     python scripts/smoke_test_maze_generation.py
#
# without first installing the jd package.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from jd.models.loading import (
    DEFAULT_MODEL_NAME,
    count_parameters,
    load_model_and_tokenizer,
)
from jd.models.generation import generate_completions

from jd.tasks.maze.dataset import (
    build_maze_dataset,
    load_maze_dataset,
)
from jd.tasks.maze.prompts import build_maze_messages
from jd.tasks.maze.parser import (
    extract_numbered_routes,
    routes_are_distinct,
)
from jd.tasks.maze.rewards import (
    MAZE_REWARD_NAMES,
    compute_candidate_rewards_from_record,
)


NUM_ROUTES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test for Maze dataset -> prompt -> "
            "Qwen generation -> parser -> reward matrix."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local model path.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing train.jsonl and test.jsonl. "
            "If omitted, a tiny Maze dataset is generated in memory."
        ),
    )

    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset example index.",
    )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help=(
            "Number G of independent model completions. "
            "Use 2 for a smoke test; later use 8 for training."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="For example: cuda, cuda:0, cpu.",
    )

    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch sampling seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Reproducibility for stochastic generation.
    # -----------------------------------------------------------------------

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -----------------------------------------------------------------------
    # 2. Load or construct Maze dataset.
    # -----------------------------------------------------------------------

    if args.data_dir is not None:
        print(
            f"\nLoading Maze dataset from: {args.data_dir}"
        )

        dataset = load_maze_dataset(
            args.data_dir
        )

    else:
        print(
            "\nNo --data-dir supplied."
            "\nGenerating a tiny 1-train / 1-test Maze dataset in memory..."
        )

        dataset = build_maze_dataset(
            train_size=1,
            test_size=1,
        )

    print("\nDataset:")
    print(dataset)

    split = dataset[args.split]

    if not (0 <= args.index < len(split)):
        raise IndexError(
            f"Index {args.index} is invalid for split "
            f"{args.split!r} with {len(split)} examples."
        )

    record = split[args.index]

    print(
        f"\nSelected example:"
        f"\n  split = {args.split}"
        f"\n  index = {args.index}"
        f"\n  id    = {record['id']}"
        f"\n  seed  = {record['seed']}"
    )

    # -----------------------------------------------------------------------
    # 3. Convert Maze record into chat messages.
    # -----------------------------------------------------------------------

    messages = build_maze_messages(
        record,
        num_routes=NUM_ROUTES,
    )

    print("\n" + "=" * 80)
    print("CHAT PROMPT")
    print("=" * 80)

    for message in messages:
        print(
            f"\n[{message['role'].upper()}]"
        )
        print(
            message["content"]
        )

    # -----------------------------------------------------------------------
    # 4. Load Qwen3.5-0.8B.
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    bundle = load_model_and_tokenizer(
        model_name_or_path=args.model,
        device=args.device,
        dtype=args.dtype,
        freeze_vision=True,
    )

    trainable_parameters, total_parameters = count_parameters(
        bundle.model
    )

    print(
        f"\nModel:  {args.model}"
        f"\nDevice: {bundle.device}"
        f"\nTrainable parameters: {trainable_parameters:,}"
        f"\nTotal parameters:     {total_parameters:,}"
    )

    # -----------------------------------------------------------------------
    # 5. Generate G rollouts.
    #
    # IMPORTANT:
    #
    # Each generated text here is ONE rollout.
    # Each rollout should itself contain N=3 routes.
    #
    # So:
    #
    #     G completions
    #         x
    #     N=3 routes per completion
    #
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        f"GENERATING {args.num_generations} ROLLOUT(S)"
    )
    print("=" * 80)

    generations = generate_completions(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        messages=messages,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=20,
    )

    # -----------------------------------------------------------------------
    # 6. Inspect generation tensors.
    # -----------------------------------------------------------------------

    print("\nGeneration tensor shapes:")

    print(
        "  prompt_input_ids:     ",
        tuple(generations.prompt_input_ids.shape),
    )

    print(
        "  prompt_attention_mask:",
        tuple(generations.prompt_attention_mask.shape),
    )

    print(
        "  sequences:            ",
        tuple(generations.sequences.shape),
    )

    print(
        "  completion_ids:       ",
        tuple(generations.completion_ids.shape),
    )

    print(
        "  completion_mask:      ",
        tuple(generations.completion_mask.shape),
    )

    # -----------------------------------------------------------------------
    # 7. Print, parse, and reward each rollout.
    # -----------------------------------------------------------------------

    for rollout_index, completion in enumerate(
        generations.texts,
        start=1,
    ):
        print("\n" + "#" * 80)
        print(
            f"ROLLOUT {rollout_index}"
        )
        print("#" * 80)

        print("\nRAW MODEL OUTPUT:\n")
        print(completion)

        # ---------------------------------------------------------------
        # Parse the three route tags.
        # ---------------------------------------------------------------

        routes = extract_numbered_routes(
            completion,
            num_routes=NUM_ROUTES,
        )

        print("\nPARSED ROUTES:")

        if routes is None:
            print(
                "  INVALID: parser could not extract "
                "all three valid routes."
            )

        else:
            for route_index, route in enumerate(
                routes,
                start=1,
            ):
                print(
                    f"  route_{route_index}: "
                    f"{route}"
                )

            print(
                "\nAll routes distinct:",
                routes_are_distinct(routes),
            )

        # ---------------------------------------------------------------
        # This additionally verifies rewards.py.
        #
        # Expected conceptual shape:
        #
        #       [N=3, M=4]
        #
        # ---------------------------------------------------------------

        reward_matrix = compute_candidate_rewards_from_record(
            record,
            completion,
            num_routes=NUM_ROUTES,
            require_distinct=False,
        )

        print("\nREWARD MATRIX [N=3, M=4]:")

        print(
            "  columns:",
            MAZE_REWARD_NAMES,
        )

        for route_index, reward_vector in enumerate(
            reward_matrix,
            start=1,
        ):
            print(
                f"  route_{route_index}: "
                f"{reward_vector}"
            )

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()