from typing import List, Tuple, Union, Dict, Any
import random
import copy
import os
import json

from .common_puzzle_factory import PuzzleFactory
from .common_constriants import Constraint

class ConstraintOddEven(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_odd_even"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        cell_types = game_state.get("cell_types", None)


        if cell_types is None:
            return True

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    is_even = board[i][j] % 2 == 0
                    if (cell_types[i][j] == 'w' and not is_even) or \
                       (cell_types[i][j] == 'b' and is_even):
                        return False
        return True

class ConstraintRowNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        for row in board:

            values = [x for x in row if x != 0]

            if len(values) != len(set(values)):
                return False
        return True

class ConstraintColNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for col in range(size):

            values = [board[row][col] for row in range(size) if board[row][col] != 0]

            if len(values) != len(set(values)):
                return False
        return True

class ConstraintSubGridNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_sub_grid_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        sub_size = int(size ** 0.5)

        for box_row in range(0, size, sub_size):
            for box_col in range(0, size, sub_size):

                values = []
                for i in range(sub_size):
                    for j in range(sub_size):
                        value = board[box_row + i][box_col + j]
                        if value != 0:
                            values.append(value)

                if len(values) != len(set(values)):
                    return False
        return True

class OddEvenSudokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "oddevensudoku"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintSubGridNoRepeat(),
            ConstraintOddEven()
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        possible_values = []
        board = game_state["board"]
        cell_types = game_state.get("cell_types", None)
        original_value = board[row][col]


        if cell_types:
            cell_type = cell_types[row][col]
            filtered_values = [v for v in self.all_possible_values if
                             (cell_type == 'w' and v % 2 == 0) or
                             (cell_type == 'b' and v % 2 == 1)]
        else:
            filtered_values = self.all_possible_values

        for value in filtered_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
