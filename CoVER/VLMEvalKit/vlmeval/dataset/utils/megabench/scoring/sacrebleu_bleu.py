from numbers import Number
import sacrebleu


class Bleu:

    @staticmethod
    def match(response, correct_answer) -> Number:
        if isinstance(response, str) and isinstance(correct_answer, str):
            resp = [response]
            corr = [correct_answer]
        elif isinstance(response, (list, tuple)) and isinstance(
            correct_answer, (list, tuple)
        ):
            resp = tuple(response)
            corr = tuple(correct_answer)
        else:
            return 0
        result = sacrebleu.corpus_bleu(corr, [resp]).score / 100
        return result
