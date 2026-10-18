import logging
from typing import Dict, Any
from .common.conversions import str_to_set
from .common.metrics import jaccard_index


def chess_transform(move_sequence: str) -> set:
    move_sequence = str_to_set(move_sequence)
    return {move_san.removesuffix("!").removesuffix("#") for move_san in move_sequence}


class ChessMoveJaccard:

    @classmethod
    def match(cls, responses: str | None, targets: str) -> float:
        if responses is None:
            return 0
        responses = chess_transform(responses)
        targets = chess_transform(targets)

        return jaccard_index(responses, targets)
