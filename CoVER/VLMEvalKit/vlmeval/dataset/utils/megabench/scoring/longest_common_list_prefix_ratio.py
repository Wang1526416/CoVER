from .common.conversions import str_to_list
from .common.metrics import longest_common_prefix


class LongestCommonListPrefixRatio:

    @classmethod
    def match(cls, responses, targets) -> int:
        responses = str_to_list(responses)
        targets = str_to_list(targets)
        return len(longest_common_prefix(responses, targets)) / len(targets)
