import ast
import json
import re
import regex
from typing import List
from .utils import extract_code_block_content, extract_answer_at_beginning_of_line


PARSING_TIMEOUT = 0.1


def parse_json(response: str):

    response_ = extract_answer_at_beginning_of_line(response)


    response_, _ = extract_code_block_content(response_, "json")


    json_pattern = r"(\{(?:[^{}]|(?R))*\}|\[(?:[^{}]|(?R))*\])"
    string_pattern = r'"(?:\\.|[^"\\])*"'


    try:
        potential_jsons = regex.findall(
            json_pattern, response_, timeout=PARSING_TIMEOUT
        )
    except TimeoutError:
        if response_.startswith("["):
            return []
        return {}

    valid_jsons = []

    for potential_json in potential_jsons:

        potential_json = potential_json.replace('\\"', "__DOUBLE_QUOTE__")
        potential_json = potential_json.replace("\\'", "__SINGLE_QUOTE__")


        strings = regex.findall(string_pattern, potential_json)


        for s in strings:

            unescaped = (
                s[1:-1]
                .replace("__DOUBLE_QUOTE__", '"')
                .replace("__SINGLE_QUOTE__", "'")
            )

            try:
                parsed = json.loads(unescaped)
                if isinstance(parsed, (dict, list)):

                    potential_json = potential_json.replace(s, json.dumps(parsed))
            except json.JSONDecodeError:
                pass


        potential_json = potential_json.replace("__DOUBLE_QUOTE__", '\\"')
        potential_json = potential_json.replace("__SINGLE_QUOTE__", "\\'")

        try:

            json_object = json.loads(potential_json)
            valid_jsons.append(json_object)
        except json.JSONDecodeError:


            potential_json_ = re.sub(r"(?<!\w)\'|\'(?!\w)", '"', potential_json)
            try:
                json_object = json.loads(potential_json_)
                valid_jsons.append(json_object)
            except json.JSONDecodeError:

                pass

        try:

            valid_jsons.append(ast.literal_eval(potential_json))
            continue
        except (SyntaxError, ValueError):
            pass

        potential_json = potential_json.replace('\\"', '\\\\"')
        potential_json = potential_json.replace("\\'", "\\\\'")
        try:

            valid_jsons.append(ast.literal_eval(potential_json))
        except (SyntaxError, ValueError):
            pass


    if valid_jsons:
        return valid_jsons[-1]
    if response_.startswith("["):
        return []
    return {}


def parse_nested_str_list(input_string):

    quoted_string = re.sub(r"(\w+)", r'"\1"', input_string)


    try:
        python_object = ast.literal_eval(quoted_string)
        return python_object
    except (ValueError, SyntaxError) as e:
        print(f"Failed to convert string to Python object: {e}")
        return input_string


def parse_syllable_ranges(input_str: str) -> List[List[int]]:

    def convert_to_range(match):
        match = match.strip("[]")
        if "," in match:
            start, end = map(int, match.split(","))
            return [start, end]
        elif match == " ":
            return [0, 0]
        else:
            num = int(match)
            return [num, num]


    chunks = re.findall(r"(?:\[\d+(?:,\s*\d+)?\]|\d| )", input_str.strip())


    result = [convert_to_range(chunk) for chunk in chunks]

    return result
