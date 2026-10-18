from .common.parsers import parse_json
from .common.utils import evaluate_as_string


class JsonParse:

    @staticmethod
    def parse(response: str):
        parsed_res = parse_json(response)

        if isinstance(parsed_res, dict):
            for key, val in parsed_res.items():
                parsed_res[key] = evaluate_as_string(val)

        return parsed_res
