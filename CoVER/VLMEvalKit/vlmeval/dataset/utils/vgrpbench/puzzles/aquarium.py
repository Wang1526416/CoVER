import argparse
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory

class ConstraintAquariumFill(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_aquarium_fill"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        aquariums = game_state.get("clues", {}).get("aquariums", [])

        for aquarium in aquariums:

            highest_water_row = float('inf')
            for r, c in aquarium:
                if board[r][c] == "s":
                    highest_water_row = min(highest_water_row, r)

            if highest_water_row == float('inf'):
                continue


            for r, c in aquarium:
                if r >= highest_water_row:
                    if board[r][c] == "e":
                        return False

        return True

class ConstraintAquariumCount(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_aquarium_count"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True

        size = len(board)
        row_counts = clues["row_counts"]
        col_counts = clues["col_counts"]


        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)

            if 0 not in board[i]:
                if row_selected != row_counts[i]:
                    return False
            else:

                if row_selected > row_counts[i]:
                    return False

                if row_selected + row_undefined < row_counts[i]:
                    return False


        for j in range(size):
            col_cells = [board[i][j] for i in range(size)]
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(cell != 0 for cell in col_cells):
                if col_selected != col_counts[j]:
                    return False
            else:
                if col_selected > col_counts[j]:
                    return False
                if col_selected + col_undefined < col_counts[j]:
                    return False

        return True


class AquariumPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4:
            raise ValueError("Size must be at least 4")

        self.game_name = "aquarium"
        self.size = size
        self.constraints = [ConstraintAquariumFill(), ConstraintAquariumCount()]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(
        self, game_state: Dict[str, Any], row: int, col: int
    ) -> List[str]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
