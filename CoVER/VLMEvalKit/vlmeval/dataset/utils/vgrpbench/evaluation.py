import ast
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from .score import evaluate_single_puzzle

def get_eval(judge, content):
    return judge.generate(content)

def VGRPBench_atomeval(model, prompt, line):
    print("raw output", prompt)
    output = get_eval(model, prompt)
    print("formatted output", output)
    scores = parse_score(line, output)
    return scores

def parse_score(line, output):
    

    category = line['category']
    puzzle_type = category.split('_')[0]
    

    puzzle_data = line['states']
    puzzle_data = ast.literal_eval(puzzle_data)
    

    evaluation_result = evaluate_single_puzzle(output, puzzle_data, puzzle_type)

    return evaluation_result

def VGRPBench_score(data):

    cates = list(set(data['category']))
    ret = defaultdict(list)

    for c in cates:
        ret['category'].append(c)

        sub = data[data['category'] == c]
        

        perception_score = round(np.mean(sub['perception_correct']) * 100, 2)
        ret['Perception Score'].append(perception_score)
        

        answer_score = round(np.mean(sub['answer_correct']) * 100, 2)
        ret['Answer Score'].append(answer_score)
        
    return pd.DataFrame(ret)

def build_prompt(line):

    return line['prediction']

def VGRPBench_get_system_prompt(line):

    puzzle_type = line['category'].split('_')[0]


    prompt_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        "formating-prompt",
        puzzle_type,
        "filter_prompt.json"
    )


    with open(prompt_file, 'r') as f:
        prompt = json.load(f)

    prompt = str(prompt) + "According to the conversation history with the user feedback do the formatting for the current answer."

    return prompt
