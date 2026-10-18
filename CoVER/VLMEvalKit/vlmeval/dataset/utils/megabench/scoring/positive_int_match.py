import ast


class PositiveIntMatch:

    @staticmethod
    def match(response: str, correct_answer: str) -> int:
        try:
            response_obj = ast.literal_eval(response)
        except (SyntaxError, ValueError):
            return 0

        if not correct_answer:
            return 0

        correct_answer_obj = ast.literal_eval(correct_answer)

        assert isinstance(correct_answer_obj, int)
        if not isinstance(response_obj, int):
            return 0


        if correct_answer_obj <= 0 and response_obj <= 0:
            return -1

        return 1 if response_obj == correct_answer_obj else 0
