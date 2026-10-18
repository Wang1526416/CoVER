import logging
from .common.parsers import parse_json
from .common.utils import (
    extract_code_block_content,
    extract_answer_content,
    evaluate_as_string,
    drop_additional_text,
)

logger = logging.getLogger("errorLogger")


class AnswerStrParse:

    @classmethod
    def _parse(
        cls,
        response: str,
        *,
        is_ascii_art: bool = False,
        should_remove_surrounding_whitespace=True,
        global_description: str = "",
        query_question: str = "",
        is_single_line_ans: bool = None,
    ) -> dict:
        if response is None:
            response = ""


        answer_content = extract_answer_content(
            response,
            is_ascii_art=is_ascii_art,
            should_remove_surrounding_whitespace=should_remove_surrounding_whitespace,
        )


        answer_content, is_code = extract_code_block_content(
            answer_content,
            is_ascii_art=is_ascii_art,
            should_remove_surrounding_whitespace=should_remove_surrounding_whitespace,
        )

        if not is_code and is_single_line_ans and not is_ascii_art:
            answer_content = drop_additional_text(answer_content)


        if answer_content.startswith("{") or answer_content.startswith("["):

            response_obj = parse_json(answer_content)
            if response_obj == {}:
                if "{}" not in answer_content:
                    return answer_content
            elif response_obj == []:



                if "[]" not in answer_content:
                    return answer_content
            return str(response_obj)
        else:

            answer_content = evaluate_as_string(answer_content)
            return answer_content

    @classmethod
    def parse(
        cls,
        response: str,
        answer_key: str,
        *,
        global_description: str = "",
        query_question: str = "",
        is_single_line_ans: bool = None,
    ) -> dict:
        response_parsed = cls._parse(
            response,
            is_ascii_art=False,
            global_description=global_description,
            query_question=query_question,
            is_single_line_ans=is_single_line_ans,
        )
        results = {answer_key: response_parsed}
        return results


class AsciiAnswerStrParse(AnswerStrParse):

    @classmethod
    def parse(
        cls,
        response: str,
        answer_key: str,
        *,
        global_description: str = "",
        query_question: str = "",
        is_single_line_ans: bool = None,
    ) -> dict:
        response_parsed = cls._parse(
            response,
            is_ascii_art=True,
            global_description=global_description,
            query_question=query_question,
            is_single_line_ans=is_single_line_ans,
        )
        results = {answer_key: response_parsed}
        return results


class VerbatimAnswerStrParse(AnswerStrParse):

    @classmethod
    def parse(
        cls,
        response: str,
        answer_key: str,
        *,
        global_description: str = "",
        query_question: str = "",
        is_single_line_ans: bool = None,
    ) -> dict:
        response_parsed = cls._parse(
            response,
            is_ascii_art=True,
            should_remove_surrounding_whitespace=False,
            global_description=global_description,
            query_question=query_question,
            is_single_line_ans=is_single_line_ans,
        )
        results = {answer_key: response_parsed}
        return results
