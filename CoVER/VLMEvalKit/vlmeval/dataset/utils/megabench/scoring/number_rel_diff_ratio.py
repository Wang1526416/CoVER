import ast
import math
from numbers import Number


class NumberRelDiffRatio:

    @staticmethod
    def match(response: str | Number, correct_answer: str) -> int:
        try:
            if isinstance(response, Number):
                pred = response
            else:
                pred = ast.literal_eval(response)
            if not isinstance(pred, Number):
                return 0
            gt = ast.literal_eval(correct_answer)
            return max(0, 1 - math.fabs((pred - gt) / gt))
        except (SyntaxError, ValueError):
            return 0
