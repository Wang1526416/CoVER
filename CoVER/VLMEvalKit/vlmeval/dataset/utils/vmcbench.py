import pandas as pd
import numpy as np
import random


def parse_multi_choice_response(response, all_choices, index2ans):
    response = str(response)
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f'({choice})' in response or f'{choice}. ' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)


    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)

            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)

        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


def get_mc_score(row, use_parse=True):
    if use_parse:
        if pd.isna(row["A"]):
            return False
        response = row["prediction"]
        all_choices = []
        for i in range(9):
            if chr(65 + i) in row and not pd.isna(row[chr(65 + i)]):
                all_choices.append(chr(65 + i))
        index2ans = {index: row[index] for index in all_choices}
        pred_index = parse_multi_choice_response(response, all_choices, index2ans)
    else:
        pred_index = row["output"]
    return int(pred_index == row["answer"])


def report_vmc_acc(data):
    general_datasets = ["SEEDBench", "MMStar", "A-OKVQA", "VizWiz", "MMVet", "VQAv2", "OKVQA"]
    reason_datasets = ["MMMU", "MathVista", "ScienceQA", "RealWorldQA", "GQA", "MathVision"]
    ocr_datasets = ["TextVQA", "OCRVQA"]
    doc_datasets = ["AI2D", "ChartQA","DocVQA", "InfoVQA", "TableVQABench"]
    results = {}
    for category in data['category'].unique():
        results[category] = data[data['category'] == category]['hit'].mean()
    results = pd.DataFrame(results, index=[0])
    results["Overall"] = data['hit'].mean()
    results['General'] = results[general_datasets].mean(axis=1)
    results['Reasoning'] = results[reason_datasets].mean(axis=1)
    results['OCR'] = results[ocr_datasets].mean(axis=1)
    results['Doc & Chart'] = results[doc_datasets].mean(axis=1)
    for key in results:
        results[key] = round(results[key] * 100, 2)
    results = results[['Overall', 'General', 'Reasoning', 'OCR', 'Doc & Chart']
                      + general_datasets + reason_datasets + ocr_datasets + doc_datasets]
    return results
