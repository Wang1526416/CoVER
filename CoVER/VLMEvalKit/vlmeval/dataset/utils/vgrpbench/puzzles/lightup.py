import argparse
import random
from typing import Any, Dict, List, Tuple
import os

from .common_puzzle_factory import PuzzleFactory
from .common_constriants import Constraint, ConstraintRowNoRepeat, ConstraintColNoRepeat, ConstraintSubGridNoRepeat

class ConstraintLightUpBulb(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_bulb"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        for row in range(size):
            for col in range(size):
                if board[row][col] == 's':

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = row + dx, col + dy
                        while 0 <= nx < size and 0 <= ny < size:
                            if board[nx][ny] == 'w':
                                break
                            if board[nx][ny] == 's':
                                return False

                            nx += dx
                            ny += dy
        return True

class ConstraintLightUpWall(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_wall"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]

        wall_numbers = game_state["wall_numbers"]

        if not wall_numbers:
            return True

        size = len(board)

        for row in range(size):
            for col in range(size):

                if board[row][col] == 'w' and wall_numbers[row][col] != -1:
                    light_count = 0
                    undefined_count = 0

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = row + dx, col + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if board[nx][ny] == 's':
                                light_count += 1
                            elif board[nx][ny] == 0:
                                undefined_count += 1




                    if (light_count > wall_numbers[row][col] or
                        light_count + undefined_count < wall_numbers[row][col]):
                        return False
        return True

class ConstraintLightUpIllumination(Constraint):

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_lightup_illumination"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)


        for row in range(size):
            for col in range(size):
                if board[row][col] == 'e':
                    can_be_illuminated = False

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = row + dx, col + dy
                        while 0 <= nx < size and 0 <= ny < size:
                            if board[nx][ny] == 'w':
                                break
                            if board[nx][ny] == 's' or board[nx][ny] == 0:
                                can_be_illuminated = True
                                break
                            nx += dx
                            ny += dy

                        if can_be_illuminated:
                            break

                    if not can_be_illuminated:
                        return False

        return True

class LightUpPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 3:
            raise ValueError("Size must be at least 3")

        self.game_name = "lightup"
        self.size = size

        self.constraints = [
            ConstraintLightUpBulb(),
            ConstraintLightUpWall(),
            ConstraintLightUpIllumination()
        ]

        self.all_possible_values = ['s', 'e']

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        board = game_state["board"]
        if board[row][col] in [-1, 1, 2, 3, 4]:
            return []

        possible_values = []
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
