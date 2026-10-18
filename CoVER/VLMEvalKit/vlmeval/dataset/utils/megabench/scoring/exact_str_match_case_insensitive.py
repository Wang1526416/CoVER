from .exact_str_match import ExactStrMatch


class ExactStrMatchCaseInsensitive:

    @staticmethod
    def match(response, correct_answer) -> int:
        if not isinstance(response, str) and isinstance(correct_answer, str):
            return 0
        return ExactStrMatch.match(response.lower(), correct_answer.lower())
