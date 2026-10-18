from .jaccard import Jaccard
from .common.conversions import cast_to_dict


class DictJaccardAggJaccard:

    @classmethod
    def match(cls, responses, targets) -> float:
        responses = cast_to_dict(responses)
        if not isinstance(responses, dict):
            return 0

        all_keys = set(responses) | set(targets)

        num_keys = 0
        total_score = 0
        for key in all_keys:
            total_score += Jaccard.match(responses.get(key, []), targets.get(key, []))
            num_keys += 1

        return total_score / num_keys
