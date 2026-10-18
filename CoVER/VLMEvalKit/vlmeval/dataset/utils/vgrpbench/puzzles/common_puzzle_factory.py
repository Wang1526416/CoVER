from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import copy
import random
import json
import os

import argparse

def hint_type(value):
    if value == "random":
        return "random"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' must be 'random' or an integer")


class PuzzleFactory():
    def __init__(self) -> None:
        self.constraints = []
        self.game_name = "unknown"
        self.size = 0

        self.train_ratio = 8
        self.val_ratio = 1
        self.ablation_ratio = 1

    def sample_hints(self, board: List[List[int]], num_sample_hints: int) -> List[List[int]]:

        new_board = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]

        sampled_cells = random.sample(range(len(board) * len(board[0])), num_sample_hints)
        for cell in sampled_cells:
            row = cell // len(board[0])
            col = cell % len(board[0])
            new_board[row][col] = board[row][col]
        return new_board
    def save_puzzles(self, puzzles: List[Dict[str, Any]], save_path: str = "datasets/", filename: str = None) -> None:
        if filename is None:
            base_path = f"{save_path}/{self.game_name}_{self.size}x{self.size}_puzzles"
        else:
            base_path = f"{save_path}/{filename.rsplit('.', 1)[0]}"


        solution_groups = {}
        for puzzle in puzzles:
            solution_key = str(puzzle['solution'])
            if solution_key not in solution_groups:
                solution_groups[solution_key] = []
            solution_groups[solution_key].append(puzzle)


        sorted_groups = sorted(solution_groups.items(), key=lambda x: len(x[1]), reverse=True)

        total_puzzles = len(puzzles)
        target_val_size = total_puzzles * self.val_ratio // 10
        target_ablation_size = total_puzzles * self.ablation_ratio // 10

        train_puzzles = []
        val_puzzles = []
        ablation_puzzles = []

        val_solutions = set()
        current_val_size = 0
        val_group_idx = 0
        while val_group_idx < len(sorted_groups) and current_val_size < target_val_size:
            group = sorted_groups[val_group_idx][1]
            if current_val_size + len(group) <= target_val_size * 1.2:
                val_puzzles.extend(group)
                val_solutions.add(sorted_groups[val_group_idx][0])
                current_val_size += len(group)
            val_group_idx += 1


        train_solutions = set()
        current_ablation_size = 0

        for solution, group in sorted_groups:
            if solution in val_solutions:
                continue

            train_solutions.add(solution)

            if current_ablation_size < target_ablation_size:

                space_left = target_ablation_size - current_ablation_size

                ablation_count = min(max(1, len(group) // 5), space_left)

                ablation_indices = random.sample(range(len(group)), ablation_count)
                for i in range(len(group)):
                    if i in ablation_indices:
                        ablation_puzzles.append(group[i])
                        current_ablation_size += 1
                    else:
                        train_puzzles.append(group[i])
            else:
                train_puzzles.extend(group)


        random.shuffle(train_puzzles)
        random.shuffle(val_puzzles)
        random.shuffle(ablation_puzzles)


        os.makedirs(os.path.dirname(f"{base_path}_train.json"), exist_ok=True)


        for split_name, split_puzzles in [
            ("train", train_puzzles),
            ("val", val_puzzles),
            ("ablation", ablation_puzzles)
        ]:
            split_path = f"{base_path}_{split_name}.json"
            with open(split_path, "w") as f:
                json.dump(split_puzzles, f, indent=2)
        print(f"\nSplit and saved {len(puzzles)} puzzles:")
        print(f"Train: {len(train_puzzles)} puzzles ({len(train_solutions)} unique solutions)")
        print(f"Val: {len(val_puzzles)} puzzles ({len(val_solutions)} unique solutions)")
        print(f"Ablation: {len(ablation_puzzles)} puzzles (solutions shared with train)")
        print(f"Files saved to {base_path}_[train/val/ablation].json")

    def check(self, game_state: Dict[str, Any]) -> bool:
        for constraint in self.constraints:
            if not constraint.check(game_state):
                return False
        return True

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        pass
