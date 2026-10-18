from typing import List, Dict, Any, Tuple
import random
import copy
import os
import json
import argparse

from .common_puzzle_factory import PuzzleFactory
from .common_constriants import Constraint

class ConstraintThermometerFill(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_fill"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        thermometers = game_state.get("clues", {}).get("thermometers", [])



        thermometer_positions = {(r, c) for therm in thermometers for r, c in therm}


        for i in range(len(board)):
            for j in range(len(board[i])):
                if (i, j) not in thermometer_positions and board[i][j] == "s":
                    return False

        for thermometer in thermometers:

            first_empty = -1
            for i, (r, c) in enumerate(thermometer):
                if board[r][c] == "e":
                    first_empty = i
                    break

            if first_empty != -1:
                for i, (r, c) in enumerate(thermometer):
                    if i > first_empty and board[r][c] == "s":
                        return False
        return True

class ConstraintThermometerCount(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_count"
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
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 for i in range(size)):
                if col_selected != col_counts[j]:
                    return False
            else:
                if col_selected > col_counts[j]:
                    return False
                if col_selected + col_undefined < col_counts[j]:
                    return False
        return True

class ThermometersPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4:
            raise ValueError("Size must be at least 4")
        self.game_name = "thermometers"
        self.size = size
        self.constraints = [
            ConstraintThermometerFill(),
            ConstraintThermometerCount()
        ]

        self.all_possible_values = ["e", "s"]
    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[str]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
