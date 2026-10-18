import logging
import ast
from .common.conversions import str_to_bboxes
from .common.metrics import calculate_iou
import numpy as np


class NbboxIouTuple:

    @classmethod
    def match(cls, responses, targets) -> float:
        logging.debug(f"{responses=}, {targets=}")
        if not isinstance(responses, (tuple | list)):
            responses = str_to_bboxes(responses)
        if not isinstance(targets, (tuple | list)):
            targets = str_to_bboxes(targets)

        try:
            iou_scores = calculate_iou(responses, targets)
        except:
            return 0

        if not iou_scores:
            return 0


        return sum(iou_scores) / len(iou_scores)


class NbboxIouSingle:

    @classmethod
    def match(cls, responses, targets) -> float:
        logging.debug(f"{responses=}, {targets=}")
        targets = ast.literal_eval(targets)
        try:
            responses = ast.literal_eval(responses)
        except SyntaxError:
            return 0

        try:
            iou_scores = calculate_iou(
                [
                    responses,
                ],
                [
                    targets,
                ],
            )
            if not iou_scores:
                return 0
        except:
            return 0


        return sum(iou_scores) / len(iou_scores)


class NbboxIouSequence:

    @classmethod
    def match(cls, responses, targets) -> float:
        if not isinstance(responses, (tuple | list)):
            responses = str(responses) if not isinstance(responses, str) else responses
            responses = str_to_bboxes(responses)
        if not isinstance(targets, (tuple | list)):
            targets = str_to_bboxes(targets)

        if len(targets) != len(responses):
            return 0

        scores = []
        for res, tgt in zip(responses, targets):
            scores.append(
                calculate_iou(
                    [
                        res,
                    ],
                    [
                        tgt,
                    ],
                )
            )
        avg_iou = np.mean(scores)

        return avg_iou
