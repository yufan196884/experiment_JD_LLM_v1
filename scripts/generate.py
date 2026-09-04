#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Allow this script to be run directly from the repository root:
#
#     python scripts/generate.py
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
from jd.training.advantages import (
    compute_reward_decoupled_advantages,
)
from jd.tasks.maze.dataset import (
    build_maze_dataset,
    load_maze_dataset,
    maze_from_record,
)
from jd.tasks.maze.prompts import (
    MAZE_PREFERENCE_NAMES,
    build_maze_messages,
    validate_maze_preference,
)
from jd.tasks.maze.parser import parse_route
from jd.tasks.maze.rewards import (
    MAZE_REWARD_NAMES,
    ZERO_REWARD,
    compute_reward_vector,
    simulate_route,
)


# =============================================================================
# Single-route parsing REMOVED
# =============================================================================



# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test for preference-conditioned single-route Maze "
            "generation: dataset -> prompt -> Qwen generation -> "
            "parser -> reward vector -> reward-decoupled advantages."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local model path.",
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking mode.",
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
        "--preference",
        type=float,
        nargs=4,
        default=(0.25, 0.25, 0.25, 0.25),
        help=(
            "Four preference weights in the order: "
            "completion gold diamond lava_avoidance. "
            "Weights must be non-negative and sum to 1. "
            "Example: --preference 0.1 0.7 0.1 0.1"
        ),
    )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help=(
            "Number G of independent one-route rollouts. "
            "Use 2 for a smoke test; use 8 initially for training."
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


# =============================================================================
# Main smoke test
# =============================================================================

def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Validate preference.
    #
    # One preference omega is shared by ALL G rollouts in this group.
    #
    #     (Maze x, omega)
    #           |
    #           +--> rollout 1 -> one route
    #           +--> rollout 2 -> one route
    #           ...
    #           +--> rollout G -> one route
    #
    # -----------------------------------------------------------------------

    preference = validate_maze_preference(
        args.preference
    )

    print("\nPreference vector:")

    for name, weight in zip(
        MAZE_PREFERENCE_NAMES,
        preference,
    ):
        print(
            f"  {name:16s} = {weight:.6f}"
        )

    # -----------------------------------------------------------------------
    # 2. Reproducibility for stochastic generation.
    # -----------------------------------------------------------------------

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    # -----------------------------------------------------------------------
    # 3. Load or construct Maze dataset.
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

    split = dataset[
        args.split
    ]

    if not (
        0
        <= args.index
        < len(split)
    ):
        raise IndexError(
            f"Index {args.index} is invalid for split "
            f"{args.split!r} with {len(split)} examples."
        )

    record = split[
        args.index
    ]

    maze = maze_from_record(
        dict(record)
    )

    print(
        f"\nSelected example:"
        f"\n  split = {args.split}"
        f"\n  index = {args.index}"
        f"\n  id    = {record['id']}"
        f"\n  seed  = {record['seed']}"
    )

    # -----------------------------------------------------------------------
    # 4. Convert (Maze, preference) into chat messages.
    #
    # IMPORTANT:
    #
    # The preference is part of the policy conditioning:
    #
    #     pi_theta(y | x, omega)
    #
    # The same omega must be used for all G rollouts in this group.
    # -----------------------------------------------------------------------

    messages = build_maze_messages(
        record,
        preference=preference,
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "CHAT PROMPT"
    )
    print(
        "=" * 80
    )

    for message in messages:
        print(
            f"\n[{message['role'].upper()}]"
        )
        print(
            message["content"]
        )

    # -----------------------------------------------------------------------
    # 5. Load model.
    # -----------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )
    print(
        "LOADING MODEL"
    )
    print(
        "=" * 80
    )

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
    # 6. Generate G independent rollouts.
    #
    # NEW STRUCTURE:
    #
    #     one prompt = (Maze x, preference omega)
    #
    #     rollout 1 -> ONE route
    #     rollout 2 -> ONE route
    #     ...
    #     rollout G -> ONE route
    #
    # There is NO VPO N=3 candidate dimension.
    # -----------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )
    print(
        f"GENERATING {args.num_generations} "
        "INDEPENDENT ONE-ROUTE ROLLOUT(S)"
    )
    print(
        "=" * 80
    )

    generations = generate_completions(
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        messages=messages,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95 if args.thinking else 1.0,
        top_k=20,
        enable_thinking=args.thinking,
    )

    # -----------------------------------------------------------------------
    # 7. Inspect generation tensors.
    # -----------------------------------------------------------------------

    print(
        "\nGeneration tensor shapes:"
    )

    print(
        "  prompt_input_ids:      ",
        tuple(
            generations.prompt_input_ids.shape
        ),
    )

    print(
        "  prompt_attention_mask: ",
        tuple(
            generations.prompt_attention_mask.shape
        ),
    )

    print(
        "  sequences:              ",
        tuple(
            generations.sequences.shape
        ),
    )

    print(
        "  completion_ids:         ",
        tuple(
            generations.completion_ids.shape
        ),
    )

    print(
        "  completion_mask:        ",
        tuple(
            generations.completion_mask.shape
        ),
    )

    # -----------------------------------------------------------------------
    # 8. Parse and reward each rollout independently.
    #
    # For one rollout:
    #
    #     completion -> one route -> RewardVector [M=4]
    #
    # Across the complete group:
    #
    #     G rollouts -> reward tensor [G, M]
    #
    # With the initial training setting:
    #
    #     G = 8
    #     M = 4
    #
    # so:
    #
    #     R in R^{8 x 4}
    #
    # -----------------------------------------------------------------------

    group_rewards = []

    for rollout_index, completion in enumerate(
        generations.texts,
        start=1,
    ):
        print(
            "\n"
            + "#" * 80
        )
        print(
            f"ROLLOUT {rollout_index}"
        )
        print(
            "#" * 80
        )

        print(
            "\nRAW MODEL OUTPUT:\n"
        )
        print(
            completion
        )

        route = parse_route(
            completion
        )

        print(
            "\nPARSED ROUTE:"
        )

        if route is None:
            print(
                "  INVALID: could not extract a valid "
                "<route>...</route> block."
            )
        else:
            print(
                " ",
                route,
            )

        reward_vector = compute_reward_vector(
            maze,
            completion,
        )

        group_rewards.append(
            reward_vector
        )

        print(
            "\nREWARD VECTOR [M=4]:"
        )

        for reward_name, reward_value in zip(
            MAZE_REWARD_NAMES,
            reward_vector,
        ):
            print(
                f"  {reward_name:16s} = {reward_value:.6f}"
            )

    # -----------------------------------------------------------------------
    # 9. Assemble the raw GDPO group reward tensor.
    #
    # No scalarization is performed here.
    #
    # This is the tensor that later enters reward-decoupled normalization:
    #
    #     R [G, M] -> A [G, M]
    #
    # -----------------------------------------------------------------------

    reward_tensor = torch.tensor(
        group_rewards,
        dtype=torch.float32,
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "GROUP REWARD TENSOR"
    )
    print(
        "=" * 80
    )

    print(
        "\nReward columns:"
    )

    for index, name in enumerate(
        MAZE_REWARD_NAMES
    ):
        print(
            f"  {index}: {name}"
        )

    print(
        "\nR ="
    )
    print(
        reward_tensor
    )

    print(
        "\nShape:",
        tuple(
            reward_tensor.shape
        ),
    )

    expected_shape = (
        args.num_generations,
        len(MAZE_REWARD_NAMES),
    )

    if tuple(
        reward_tensor.shape
    ) != expected_shape:
        raise RuntimeError(
            "Unexpected reward tensor shape: "
            f"expected {expected_shape}, "
            f"got {tuple(reward_tensor.shape)}."
        )

    # -----------------------------------------------------------------------
    # 10. Compute GDPO-style reward-decoupled advantages.
    #
    # Each reward objective is normalized independently across the G
    # rollouts:
    #
    #                R[g, i] - mean_g(R[:, i])
    #     A[g, i] = --------------------------------
    #                     std_g(R[:, i]) + eps
    #
    # Input:
    #
    #     R [G, M]
    #
    # Output:
    #
    #     A [G, M]
    #
    # IMPORTANT:
    #
    # We deliberately stop after per-reward normalization.
    #
    # We DO NOT:
    #
    #   - sum advantages across rewards;
    #   - multiply them by the preference vector;
    #   - perform GDPO's final batch-wise normalization;
    #   - collapse [G, M] into [G].
    #
    # The M separate advantage streams are required later to construct
    # M separate PPO losses and therefore M separate objective gradients
    # for Jacobian Descent.
    # -----------------------------------------------------------------------

    advantages = compute_reward_decoupled_advantages(
        reward_tensor
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "REWARD-DECOUPLED ADVANTAGES"
    )
    print(
        "=" * 80
    )

    print(
        "\nAdvantage columns:"
    )

    for index, name in enumerate(
        MAZE_REWARD_NAMES
    ):
        print(
            f"  {index}: {name}"
        )

    print(
        "\nA ="
    )
    print(
        advantages
    )

    print(
        "\nShape:",
        tuple(
            advantages.shape
        ),
    )

    if advantages.shape != reward_tensor.shape:
        raise RuntimeError(
            "Reward-decoupled advantage tensor must have the same shape "
            "as the reward tensor: "
            f"rewards={tuple(reward_tensor.shape)}, "
            f"advantages={tuple(advantages.shape)}."
        )

    # -----------------------------------------------------------------------
    # 11. Print per-objective normalization diagnostics.
    #
    # For a reward channel with non-zero variance, the advantages should
    # have approximately zero mean across the rollout group.
    #
    # A constant reward channel should instead contain all-zero advantages.
    #
    # These diagnostics make it easy to verify the normalization during
    # the smoke test before the PPO loss is implemented.
    # -----------------------------------------------------------------------

    print(
        "\nPer-objective diagnostics:"
    )

    for reward_index, reward_name in enumerate(
        MAZE_REWARD_NAMES
    ):
        raw_values = reward_tensor[
            :,
            reward_index,
        ]

        advantage_values = advantages[
            :,
            reward_index,
        ]

        raw_mean = raw_values.mean()
        raw_std = raw_values.std()

        advantage_mean = advantage_values.mean()
        advantage_std = advantage_values.std()

        print(
            f"\n  {reward_name}:"
        )
        print(
            f"    reward mean      = {raw_mean.item():.6f}"
        )
        print(
            f"    reward std       = {raw_std.item():.6f}"
        )
        print(
            f"    advantage mean   = {advantage_mean.item():.6f}"
        )
        print(
            f"    advantage std    = {advantage_std.item():.6f}"
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "SMOKE TEST COMPLETE"
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()