import rapidfuzz


class NormalizedSimilarityDamerauLevenshtein:

    @staticmethod
    def match(response, correct_answer) -> int:
        if not isinstance(response, str) and isinstance(correct_answer, str):
            return 0
        return rapidfuzz.distance.DamerauLevenshtein.normalized_similarity(
            response, correct_answer
        )
