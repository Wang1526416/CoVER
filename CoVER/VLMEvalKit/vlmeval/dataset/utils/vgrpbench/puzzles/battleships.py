import random
import argparse
import os
from typing import List, Dict, Any, Tuple

from .common_puzzle_factory import PuzzleFactory
from .common_constriants import Constraint

class ConstraintBattleships(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        for i in range(size):
            for j in range(size):
                if isinstance(board[i][j], tuple):
                    ship_cell, direction = board[i][j]

                    if direction in "<>-":

                        for di in [-1, 1]:
                            if 0 <= i + di < size and board[i + di][j] == "s":
                                return False
                    elif direction in "^V|":

                        for dj in [-1, 1]:
                            if 0 <= j + dj < size and board[i][j + dj] == "s":
                                return False
                elif board[i][j] == "s":

                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (0 <= ni < size and 0 <= nj < size and
                                (board[ni][nj] == "s" or (isinstance(board[ni][nj], tuple) and board[ni][nj][0] == "s")) and
                                (di != 0 and dj != 0)):
                                return False
        return True

class ConstraintBattleshipsHints(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state["hints"]
        row_hints = hints["row_hints"]
        col_hints = hints["col_hints"]
        ships = hints["ships"]
        size = len(board)

        total_ship_cells_required = sum(int(length) * int(count) for length, count in ships.items())
        total_ship_cells_selected = sum(1 for i in range(size) for j in range(size) if board[i][j] == "s")
        total_undefined_cells = sum(1 for i in range(size) for j in range(size) if board[i][j] == 0)


        if total_ship_cells_selected + total_undefined_cells < total_ship_cells_required:
            return False


        if total_ship_cells_selected > total_ship_cells_required:
            return False


        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)

            if all(cell != 0 and cell != -1 for cell in board[i]):
                if row_selected != row_hints[i]:
                    return False
            else:
                if row_selected > row_hints[i]:
                    return False
                if row_selected + row_undefined < row_hints[i]:
                    return False

        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 and board[i][j] != -1 for i in range(size)):
                if col_selected != col_hints[j]:
                    return False
            else:
                if col_selected > col_hints[j]:
                    return False
                if col_selected + col_undefined < col_hints[j]:
                    return False

        if total_undefined_cells == 0:

            visited = [[False] * size for _ in range(size)]
            ship_lengths = []

            def get_ship_length(i: int, j: int) -> int:
                if (i < 0 or i >= size or j < 0 or j >= size or
                    visited[i][j] or board[i][j] != "s"):
                    return 0

                visited[i][j] = True
                length = 1


                if (j + 1 < size and board[i][j + 1] == "s"):

                    for col in range(j + 1, size):
                        if board[i][col] != "s":
                            break
                        visited[i][col] = True
                        length += 1

                elif (i + 1 < size and board[i + 1][j] == "s"):

                    for row in range(i + 1, size):
                        if board[row][j] != "s":
                            break
                        visited[row][j] = True
                        length += 1

                return length


            for i in range(size):
                for j in range(size):
                    if not visited[i][j] and board[i][j] == "s":
                        ship_lengths.append(get_ship_length(i, j))

            ship_counts = {}
            for length in ship_lengths:
                ship_counts[length] = ship_counts.get(length, 0) + 1

            for length, count in ships.items():
                if ship_counts.get(int(length), 0) != int(count):
                    return False
        return True

class BattleshipsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "battleships"
        self.size = size
        self.constraints = [
            ConstraintBattleships(),
            ConstraintBattleshipsHints()
        ]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
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
