from vlmeval.dataset.utils.megabench.scoring.set_equality import SetEquality


class DictSetEqualityAggJaccard:

    @classmethod
    def match(cls, responses, targets) -> float:
        if not isinstance(responses, dict):
            return 0

        all_keys = set(responses) | set(targets)

        num_keys = 0
        total_score = 0
        for key in all_keys:
            total_score += SetEquality.match(
                responses.get(key, []), targets.get(key, [])
            )
            num_keys += 1

        return total_score / num_keys
