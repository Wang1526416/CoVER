
import re
from typing import List
import nltk












def check_whether_response_paragraph_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)


    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    actual_count = len(paragraphs)


    return lower_bound <= actual_count <= upper_bound




def check_whether_response_sentence_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)


    sentences = nltk.sent_tokenize(response)
    actual_count = len(sentences)


    return lower_bound <= actual_count <= upper_bound




def check_whether_each_paragraph_sentence_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)


    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    for i, paragraph in enumerate(paragraphs):

        sentences = nltk.sent_tokenize(paragraph)
        actual_count = len(sentences)

        if actual_count < lower_bound or actual_count > upper_bound:
            return False

    return True




def check_whether_each_paragraph_sentence_number_in_range_list(
    response: str, ranges: List[List[int]]
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)


    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    if len(paragraphs) != len(ranges):
        return False

    for i, (paragraph, range_pair) in enumerate(zip(paragraphs, ranges)):
        lower_bound, upper_bound = range_pair
        sentences = nltk.sent_tokenize(paragraph)
        actual_count = len(sentences)

        if not (lower_bound <= actual_count <= upper_bound):
            return False

    return True




def check_whether_response_word_count_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:

    response_clean = re.sub(r"[^\w\s.-]", "", response)
    word_list = response_clean.split()
    word_count = len(word_list)

    return lower_bound <= word_count <= upper_bound




def check_whether_each_paragraph_word_count_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:




    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)


    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    for i, paragraph in enumerate(paragraphs):
        paragraph_clean = re.sub(r"[^\w\s.-]", "", paragraph)
        word_count = len(paragraph_clean.split())

        if not (lower_bound <= word_count <= upper_bound):
            return False

    return True




def check_whether_whole_response_not_contain_certain_substrings(
    response: str, substrings: List[str]
) -> bool:




    return all(substring not in response for substring in substrings)




def check_whether_whole_response_not_contain_certain_substring(
    response: str, substring: str
) -> bool:
    return substring not in response




def check_whether_each_sentence_begin_with_certain_substring(
    response: str, substring: str
) -> bool:




    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)

    sentences = nltk.sent_tokenize(response)

    return all(sentence.startswith(substring) for sentence in sentences)




def check_whether_each_paragraph_begin_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    cleaned_response = clean_text(response)

    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    return all(paragraph.startswith(substring) for paragraph in paragraphs)




def check_whether_each_paragraph_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    cleaned_response = clean_text(response)

    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    return all(paragraph.endswith(substring) for paragraph in paragraphs)




def check_whether_each_sentence_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)

    sentences = nltk.sent_tokenize(response)

    return all(sentence.endswith(substring) for sentence in sentences)




def check_whether_whole_response_begin_with_certain_substring(
    response: str, substring: str
) -> bool:
    return response.strip().startswith(substring)




def check_whether_whole_response_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    return response.strip().endswith(substring)




def check_whether_each_keyword_in_list_metioned_in_range(
        response: str,
        keywords: List[str],
        lower_bound_times: int,
        upper_bound_times: int) -> bool:

    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)
    response_lower = response.lower()

    for keyword in keywords:

        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, response_lower)
        if len(matches) < lower_bound_times or len(
                matches) > upper_bound_times:
            return False

    return True




def check_whether_total_keyword_in_list_metioned_in_range(
        response: str,
        keywords: List[str],
        lower_bound_times: int,
        upper_bound_times: int) -> bool:

    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)
    response_lower = response.lower()

    count = 0
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, response_lower)
        count += len(matches)

    return lower_bound_times <= count <= upper_bound_times




def check_percentage_number_precision_in_response(
        response: str, precision: int) -> bool:


    pattern = r'(\d+\.\d+|\d+)\s*%'

    matches = re.findall(pattern, response)

    for num_str in matches:
        if '.' not in num_str:

            return False
        decimal_part = num_str.split('.')[1]
        if len(decimal_part) != precision:
            return False

    return True




def check_number_precision_in_response(response: str, precision: int) -> bool:


    number_pattern = r'''
        (?<!\w)                     # Not preceded by a word character
        [+-]?                      # Optional sign
        (?:                        # Number formats:
            \d{1,3}(?:,\d{3})*(?:\.\d+)?   # e.g., 1,234.56
            | \d+\.\d+             # e.g., 123.456
            | \.\d+                # e.g., .456
            | \d+                  # e.g., 123
        )
        (?:[eE][+-]?\d+)?          # Optional scientific notation
        %?                         # Optional percentage
        (?!\w)                     # Not followed by a word character
    '''

    matches = re.finditer(number_pattern, response, flags=re.VERBOSE)

    for match in matches:
        num_str = match.group()
        clean_num = num_str.replace(',', '').rstrip('%')


        if 'e' in clean_num.lower():
            mantissa = re.split('[eE]', clean_num)[0]
        else:
            mantissa = clean_num


        if '.' in mantissa:
            decimal_part = mantissa.split('.')[-1]
            if len(decimal_part) != precision:
                return False
        else:
            if precision != 0:
                return False

    return True




def check_whether_has_no_arabic_number_in_response(response: str) -> bool:
    number_pattern = r"""
        (?<![.\w])                            # Ensure no preceding . or word char
        (?:                                   # Start of number pattern
            \d{1,3}(?:,\d{3})+(?:\.\d+)?%?    |  # 1,000 or 1,000.00 or 1,000%
            \d+\.\d+%?                        |  # decimals: 3.14, 0.5%
            \d+%?                             |  # integers: 100, 100%
            \d+(?:\.\d+)?(?:[eE][+-]?\d+)        # scientific: 5e-10, 5.09e-10
        )
        (?![.\w])                             # Ensure no trailing . or word char
    """
    numbers = re.findall(
        number_pattern,
        response,
        flags=re.IGNORECASE | re.VERBOSE)

    return len(numbers) == 0































