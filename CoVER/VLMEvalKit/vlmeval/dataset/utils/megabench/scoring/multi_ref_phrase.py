from numbers import Number
from .common.conversions import str_to_iterable
from .simple_str_match import SimpleStrMatch


def replace_potential_chinese_comma(input_string):
    return input_string.replace("，", ",")


class MultipleReferencePhraseEval:

    @staticmethod
    def match(response, targets) -> Number:
        targets = replace_potential_chinese_comma(targets)
        refs = str_to_iterable(list, targets)
        matched = False
        for ref in refs:
            str_ref = ref if isinstance(ref, str) else str(ref)
            if SimpleStrMatch.match(response, str_ref):
                matched = True
                break
        return 1 if matched else 0
