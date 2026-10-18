import re
import json

import numpy as np
import sys
import math
import os
import argparse
import timeout_decorator
import logging

from .image_base import ImageBaseDataset
from ..utils import track_progress_rich
from ..smp import load, dump

try:
    import sympy as sp
    from sympy import simplify, Eq, sympify, Pow, pi
    from sympy.parsing.latex import parse_latex
except ImportError:
    logging.warning('sympy is not installed, please install it for MM-Math evaluation.')


class AutoScoringJudge:
    def __init__(self):

        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "厘米":"",

            "，": ",",
            "$": "",
            "（":"(",
            "）":")",
            "\\infty":"oo",
            "\\colon ":":",





            "＋":"+",
            "\\, ": "",
            "\\,":"",
            "^\\circ": "",
            "^{\\circ}": "",

        }
        self.pi = parse_latex("\\pi")

        self.precision = 1e-2

    def trans_greater_sign_to_interval(self, expr:str):
        expr_tmp = expr.split("<")
        return "(" + expr_tmp[0] + ", " + expr_tmp[-1] + ")"

    def split_by_comma(self, expr: str):

        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char in ["(", "["]:
                in_bracket_num += 1
            elif char in [")", "]"]:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):

        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)

        return new_expr_list

    def judge(self, expression1, expression2, precision=1e-2):


        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:
            return False
        if expression1 == expression2:

            return True


        expression1 = expression1 if re.fullmatch(r"[\u4e00-\u9fff]+", expression1) else re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = expression2 if re.fullmatch(r'[\u4e00-\u9fff]+', expression2) else re.sub(r'[\u4e00-\u9fff]+', '', expression2)

        if self.is_two_greater_sign(expression1):
            expression1 = self.trans_greater_sign_to_interval(expression1)

        if self.is_two_greater_sign(expression2):
            expression2 = self.trans_greater_sign_to_interval(expression2)

        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)


        if len(precision) <= 1:
            precision = precision * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False


        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                try:
                    if self.is_equal(item1, item2):
                        temp_list1.remove(item1)
                        temp_list2.remove(item2)
                        precision.remove(self.precision)
                        break
                except Exception as err:
                    logging.warning(f'{type(err)}: {err}')
                    continue
            else:

                return False


        return True

    def is_interval(self, expr):

        return expr.startswith(("(", "[")) and expr.endswith((")", "]"))

    def is_two_greater_sign(self, expr):
        match = re.findall(r'<', expr)
        return len(match) == 2

    def sympy_sub_pi(self, expression_sympy):

        return expression_sympy.subs(self.pi, math.pi)


    @timeout_decorator.timeout(30)
    def is_equal(self, expression1, expression2):

        if expression1 == expression2 and expression1 != "" and expression2 != "":

            return True


        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):

                    return True
            except:
                return False


        try:
            if self.numerical_equal(expression1, expression2):

                return True
        except:
            pass

        try:
            if self.expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):

                return True
        except:
            pass


        try:
            if self.equation_equal(expression1, expression2):

                return True
        except:
            pass

        return False

    def numerical_equal(self, expression1: str, expression2: str, include_percentage: bool = True):


        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]

        for item in gt_result:
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False

    def expression_equal(self, exp1, exp2):


        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        exp_too_long = len(exp1) > 300 or len(exp2) > 300

        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))
        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or \
                    (not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                        print("These two numbers cannot be calculated by the current computer for: "
                              f"\"{str(expr1_sym)}\" and \"{str(expr2_sym)}\"")
                        return False
                    if exp_too_long:
                        print(f'Expression {exp1} or {exp2} is too long to compute. ')
                        return False
                    if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01:
                        return True
                    else:
                        return False
                except:
                    return False
            elif exp_too_long:
                print(f'Expression {exp1} or {exp2} is too long to compute. ')
                return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)
                    num_value = simplified_expr.evalf()
                    return abs(num_value) < 1e-3
                except:
                    return False

    def equation_equal(self, expression1, expression2):


        def simplify_equation(latex_eq):
            lhs, rhs = latex_eq.split('=')

            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            equation = Eq(lhs_expr, rhs_expr)

            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        if ((division_result_1.is_Integer and division_result_1 != 0) or
                (division_result_2.is_Integer and division_result_2 != 0)):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):

        def compare_two_interval(inter1, inter2):
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True

        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")

            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):

        def extract_boxed_content(latex_str):
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")

            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str

            return results

        def sepcial_symbol_replace(expression):

            expression = expression.replace("\\text{cm}^2", '').replace("\\text{cm}", "").replace("\\,cm", '').replace("\\text{ cm}", '').replace("cm", '').replace("\\text{分米}^2", '').replace("cm^{2}", '').replace("60 \\text{ cm}^2",'').replace("\\ \\text{m}", "").replace("\\text{米}","").strip()

            expression = re.sub(r"(.+)m$", r"\1", expression)

            if "\\in " in expression:
                expression = expression.split("\\in ")[1]

            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = re.sub(r'(\\sin|\\cos|\\tan)(\d+)', r'\1((\2/180)\\pi)', expression)

            expression = expression.strip("\n,.:;^_=+`!@#%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression

        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)

        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2

    def can_compute_power(self, expr):

        if isinstance(expr, Pow):
            base, exp = expr.as_base_exp()
            if base.is_number and exp.is_number:
                MAX_EXP = 1000
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True


class MMMath(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'MM-Math': 'https://opencompass.openxlab.space/utils/VLMEval/MM-Math.tsv',
    }
    DATASET_MD5 = {
        'MM-Math': '1f064ed7c4e0e8926a3fa65849419ca5',
    }

    @classmethod
    def evaluate(self, eval_file, **kwargs):

        data = load(eval_file)
        judger = AutoScoringJudge()
        func = judger.judge

        tups = [dict(expression1=x, expression2=y) for x, y in zip(data['answer'], data['prediction'])]

        res = track_progress_rich(func, tups, nproc=16)
        data['hit'] = res
        dump(data, eval_file)

        score_file = eval_file.replace('.xlsx', '_score.json')
        score = {}
        score['overall'] = np.mean(data['hit'])

        difficulties = set(data['difficulty'])
        for d in difficulties:
            score[f'Difficulty-{d}'] = np.mean(data[data['difficulty'] == d]['hit'])


        years = set(data['year'])
        for y in years:
            score[f'Year-{y}'] = np.mean(data[data['year'] == y]['hit'])


        points = set(data['knowledge_l1'])
        for p in points:
            score[f'Knowledge-L1-{p}'] = np.mean(data[data['knowledge_l1'] == p]['hit'])


        points = set(data['knowledge_l2'])
        for p in points:
            score[f'Knowledge-L2-{p}'] = np.mean(data[data['knowledge_l2'] == p]['hit'])

        dump(score, score_file)
        return score
