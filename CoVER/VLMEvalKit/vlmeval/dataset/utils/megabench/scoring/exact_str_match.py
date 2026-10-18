import re
from ..parsing.common.utils import extract_code_block_content


def parse_single_letter(s):

    match = re.match(r"^\(?([A-Za-z])\)?(?:\s*\.\s*|\.)?(.*)", s)

    if match:

        return match.group(1)
    else:

        return s


class ExactStrMatch:

    @staticmethod
    def match(response: str, correct_answer: str) -> int:
        if not isinstance(response, str):
            response = str(response)
        if not isinstance(correct_answer, str):
            correct_answer = str(correct_answer)

        if len(correct_answer) == 1 and correct_answer.isalpha() and len(response) > 1:


            response = parse_single_letter(response)

        return 1 if response == correct_answer else 0


class CodeResultExactStrMatch:

    @staticmethod
    def match(response: str, correct_answer: str) -> int:
        correct_answer, is_code = extract_code_block_content(
            correct_answer,
            is_ascii_art=True,
            should_remove_surrounding_whitespace=False,
        )

        return ExactStrMatch.match(response, correct_answer)
