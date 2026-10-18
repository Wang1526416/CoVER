from .exact_str_match import ExactStrMatch


class SimpleStrMatch:

    @staticmethod
    def match(response, correct_answer: str) -> int:
        if not isinstance(response, str):
            response = str(response)
        response = (
            response.replace(" ", "")
            .replace("-", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace(".", "")
            .lower()
        )
        correct_answer = (
            correct_answer.replace(" ", "")
            .replace("-", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace(".", "")
            .lower()
        )

        return ExactStrMatch.match(response, correct_answer)
