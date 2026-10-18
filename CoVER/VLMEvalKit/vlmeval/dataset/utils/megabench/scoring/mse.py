import ast
import numpy as np
import math
from .common.metrics import mse
from .common.conversions import str_to_list


class MSE:

    @staticmethod
    def match(response: str, correct_answer: str) -> int:
        try:
            return mse(ast.literal_eval(response), ast.literal_eval(correct_answer))
        except (SyntaxError, ValueError):
            return 0


class NormalizedRMSE:

    MIN = 0.0
    MAX = 0.1

    @classmethod
    def match(cls, response: str, correct_answer: str) -> int:
        try:
            mse_val = mse(ast.literal_eval(response), ast.literal_eval(correct_answer))
            rmse = np.clip(np.sqrt(mse_val), cls.MIN, cls.MAX)
            norm_rmse = 1 - (rmse - cls.MIN) / (cls.MAX - cls.MIN)
            return norm_rmse
        except (SyntaxError, ValueError):
            return 0


class AngleSeqFloatRMSE:

    MIN = 0.0
    MAX = 10.0

    @classmethod
    def match(cls, responses, targets) -> float:
        responses = str_to_list(responses)
        targets = str_to_list(targets)

        if len(responses) != len(targets):
            return 0

        try:
            res = np.array(responses)
            tgt = np.array(targets)
            rmse = np.sqrt(mse(res, tgt)).sum() / len(targets)
        except:
            return 0

        rmse = np.clip(rmse, cls.MIN, cls.MAX)
        norm_rmse = 1 - (rmse - cls.MIN) / (cls.MAX - cls.MIN)
        if math.isnan(norm_rmse):
            return 0
        return norm_rmse
