from .common.conversions import cast_to_set
from .common.metrics import set_precision


class SetPrecision:

    @classmethod
    def match(cls, responses, targets) -> float:
        if responses is None:
            return 0
        responses = cast_to_set(responses)
        targets = cast_to_set(targets)

        return set_precision(responses, targets)
