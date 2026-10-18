from numbers import Number
from typing import Dict


class MinAggregation:

    @staticmethod
    def aggregate(scores: Dict[str, Number], weights: Dict[str, Number]) -> Number:
        filtered_scores = [s for s in scores.values() if s >= 0]
        if not filtered_scores:
            return -1
        return min(filtered_scores)
