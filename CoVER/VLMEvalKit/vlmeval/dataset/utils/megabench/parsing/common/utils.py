import re
import ast


def extract_code_block_content(
    response,
    code_type=None,
    is_ascii_art: bool = False,
    should_remove_surrounding_whitespace=True,
):

    if code_type:
        pattern = rf"```{code_type}\s*\n*(.*?)\s*```"
    elif is_ascii_art:
        if not response.strip() or len(response) > 10000:

            response = response.rstrip()
        if should_remove_surrounding_whitespace:
            pattern = r"```\w*(?:\s*\n+)?(.*?)\s*```"
        else:
            pattern = r"```\w*(?:\s*\n+)?(.*?)(?:\n+\s*)?```"
    else:

        pattern = r"```\w*\s*\n*(.*?)\s*```"


    match = re.search(pattern, response, flags=re.DOTALL)

    if match:

        if is_ascii_art:
            return match.group(1), True
        else:
            return match.group(1).strip(), True
    else:

        return response, False


def extract_answer_content(
    response, is_ascii_art=False, should_remove_surrounding_whitespace=True
):
    if is_ascii_art:
        match = re.search(r"\*\*?Answer:(.*?)\*\*?|\bAnswer:(.*)", response, re.DOTALL)
    else:
        match = re.search(
            r"\*\*?Answer:\s*(.*?)\*\*?|\bAnswer:\s*(.*)", response, re.DOTALL
        )
    if match:

        response = match.group(1) or match.group(
            2
        )
        if response is None:
            response = ""
    if is_ascii_art:

        response = re.sub(r"^\s*$(\n^\s*$)+", "", response, flags=re.MULTILINE)

        if should_remove_surrounding_whitespace:

            response = response.rstrip()
        else:

            response = re.sub(r"(\n\s*)+$", "", response)

        response = re.sub(r"^(\s*\n)+", "", response)
    else:
        response = response.strip()

    return response


def extract_answer_at_beginning_of_line(response):

    match = re.search(r"^(?:\*\*Answer:|Answer:)\s*(.+)", response, re.MULTILINE)

    if match:

        return match.group(1).strip()
    else:

        return response.strip()


def drop_additional_text(result):

    result_first_paragraph = result.split("\n\n")[0].strip()
    potential_ans_in_single_line = re.search(
        r"^(?:(?:[a-zA-Z0-9_-]+)(?:,\s*[a-zA-Z0-9_-]+)*|(?:[a-zA-Z0-9_-]+)\.|\((?:[a-zA-Z0-9_-]+)\)$)",
        result_first_paragraph,
    )

    only_return_first_paragraph = (
        potential_ans_in_single_line
        and result_first_paragraph.strip() != ""
        and not _is_multiline_answer(result)
    )

    if only_return_first_paragraph:
        return result_first_paragraph
    else:
        return result


def _is_multiline_answer(text):

    lines = text.splitlines()


    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line != "":

            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                return False

            return True

    return False


def evaluate_as_string(s):
    try:

        evaluated = ast.literal_eval(s)

        if isinstance(evaluated, str):
            return evaluated
        else:

            return s
    except (ValueError, SyntaxError):

        return s
    except MemoryError:

        return ""
