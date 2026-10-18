import random
import numpy as np
import argparse
import os
from typing import List, Dict, Any, Tuple

from .common_puzzle_factory import PuzzleFactory
from .common_constriants import Constraint

class ConstraintBase:
    def _check_line_hints(self, line, hints) -> bool:

        runs = []
        count = 0
        current_run = []

        if hints == [0]:

            return line.count('s') == 0


        for cell in line:
            if cell == "s":
                count += 1
                current_run.append(cell)
            elif cell == "e":
                if count > 0:
                    runs.append(count)
                    count = 0
                current_run = []
            else:
                if count > 0:
                    current_run.append(cell)

        if count > 0:
            runs.append(count)

        filled_cells = line.count("s")
        undefined_cells = line.count(0)
        required_cells = sum(hints)


        if filled_cells + undefined_cells < required_cells:
            return False


        if undefined_cells == 0:

            if runs != hints:
                return False
        else:

            definite_runs = []
            count = 0

            for cell in line:
                if cell == "s":
                    count += 1
                elif (cell == "e" or cell == 0) and count > 0:
                    definite_runs.append(count)
                    count = 0
                    if cell == 0:
                        break
            if count > 0:
                definite_runs.append(count)

            if definite_runs:

                if len(definite_runs) > len(hints):
                    return False




        return True

class ConstraintRowHints(ConstraintBase):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state.get("hints", None)
        if not hints:
            raise ValueError("Hints are not provided")
        row_hints = hints["row_hints"]

        for i, row in enumerate(board):
            if not self._check_line_hints(row, row_hints[i]):
                return False
        return True

class ConstraintColHints(ConstraintBase):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state.get("hints", None)
        if not hints:
            raise ValueError("Hints are not provided")

        col_hints = hints["col_hints"]
        size = len(board)

        for j in range(size):
            col = [board[i][j] for i in range(size)]
            if not self._check_line_hints(col, col_hints[j]):
                return False
        return True

class NonogramPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "nonogram"
        self.size = size
        self.constraints = [
            ConstraintRowHints(),
            ConstraintColHints()
        ]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[str]:
        board = game_state["board"]
        if board[row][col] != 0:
            return []

        possible_values = []
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
