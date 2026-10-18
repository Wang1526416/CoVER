# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import json
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer






@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["total"],
        metadata={"help": "List of reward functions. Possible values: 'r_final', 'r_verify', 'r_format', 'total'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    data_folder: str = field(
        default="./",
        metadata={"help": "folder to training data"},
    )
    num_fast_frames: int = field(
        default=768,
        metadata={"help": "Maximum number of low-resolution global video frames"},
    )
    num_initial_clips: int = field(
        default=3,
        metadata={"help": "Number of clips selected by CLIP in Stage 2b"},
    )
    num_verification_clips: int = field(
        default=3,
        metadata={"help": "Number of verification clips in Stage 5"},
    )






def extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_boxed(text: str) -> str:
    m = re.search(r"\\boxed\{\{(.*?)\}\}", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_answer(text: str) -> str:
    return extract_tag(text, "answer")


def normalize_number(num_str: str) -> Optional[float]:
    try:
        num_str = num_str.replace(',', '').replace('. ', '.').strip()
        return float(eval(num_str))
    except Exception:
        return None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def iou_1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    start_a, end_a = a
    start_b, end_b = b
    inter_start = max(start_a, start_b)
    inter_end   = min(end_a, end_b)
    if inter_start >= inter_end:
        return 0.0
    inter_length = inter_end - inter_start
    union_length = (end_a - start_a) + (end_b - start_b) - inter_length
    return inter_length / union_length if union_length != 0 else 0.0


def parse_time_interval(text: str) -> Optional[Tuple[float, float]]:
    text = text.strip()
    if not text:
        return None
    try:
        if text.startswith('[') or text.startswith('('):
            text = text[1:]
        if text.endswith(']') or text.endswith(')'):
            text = text[:-1]
        parts = text.split(',')
        if len(parts) != 2:
            return None
        start = float(parts[0].strip())
        end   = float(parts[1].strip())
        if start > end:
            start, end = end, start
        return (start, end)
    except Exception:
        return None


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m = len(ref_words)
    n = len(hyp_words)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[m][n] / m






def compute_final_reward(
    final_answer: str,
    ground_truth: str,
    problem_type: str,
    **kwargs
) -> float:
    if not final_answer or not ground_truth:
        return 0.0

    try:
        if problem_type == "multiple choice":
            output_ans = normalize_text(final_answer)
            gt_ans     = normalize_text(extract_answer(ground_truth) if '<answer>' in ground_truth else ground_truth)
            output_letter = re.search(r'\b([a-d])\b', output_ans)
            gt_letter     = re.search(r'\b([a-d])\b', gt_ans)
            if output_letter and gt_letter:
                return 1.0 if output_letter.group(1) == gt_letter.group(1) else 0.0
            return 1.0 if output_ans == gt_ans else 0.0

        elif problem_type == "numerical":
            output_ans = final_answer.strip()
            gt_ans     = (extract_answer(ground_truth) if '<answer>' in ground_truth else ground_truth).strip()
            gt_has_decimal  = ("." in gt_ans)  or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number  = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0

        elif problem_type == "free-form":
            output_ans = normalize_text(final_answer)
            gt_ans     = normalize_text(extract_answer(ground_truth) if '<answer>' in ground_truth else ground_truth)
            if output_ans == gt_ans:
                return 1.0
            if gt_ans in output_ans or output_ans in gt_ans:
                return 0.5
            return 0.0

        elif problem_type in ("temporal", "tal", "tvg"):
            output_interval = parse_time_interval(final_answer)
            gt_interval     = parse_time_interval(
                extract_answer(ground_truth) if '<answer>' in ground_truth else ground_truth
            )
            if output_interval is None or gt_interval is None:
                return 0.0
            return iou_1d(output_interval, gt_interval)

        elif problem_type == "OCR":
            output_ans = final_answer.strip()
            gt_ans     = (extract_answer(ground_truth) if '<answer>' in ground_truth else ground_truth).strip()
            error_rate = wer(gt_ans, output_ans)
            return max(0.0, min(1.0, 1 - error_rate))

        elif problem_type == "regression":
            output_ans = final_answer.strip()
            gt_ans     = (extract_answer(ground_truth) if '<answer>' in ground_truth else ground_truth).strip()
            gt_number  = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
            rel_diff = min(1.0, max(0.0, rel_diff))
            return 1 - rel_diff

        else:
            output_ans = normalize_text(final_answer)
            gt_ans     = normalize_text(extract_answer(ground_truth) if '<answer>' in ground_truth else ground_truth)
            return 1.0 if output_ans == gt_ans else 0.0

    except Exception as e:
        if os.getenv("DEBUG_MODE") == "true":
            print(f"Error in compute_final_reward: {e}")
        return 0.0






def check_answer_correct(answer: str, ground_truth: str, problem_type: str) -> bool:
    if not answer or not ground_truth:
        return False
    try:
        gt = extract_answer(ground_truth) if '<answer>' in ground_truth else ground_truth
        if problem_type == "multiple choice":
            a_norm  = normalize_text(answer)
            gt_norm = normalize_text(gt)
            a_letter  = re.search(r'\b([a-d])\b', a_norm)
            gt_letter = re.search(r'\b([a-d])\b', gt_norm)
            if a_letter and gt_letter:
                return a_letter.group(1) == gt_letter.group(1)
            return a_norm == gt_norm
        elif problem_type == "numerical":
            a_num  = normalize_number(answer)
            gt_num = normalize_number(gt)
            if a_num is None or gt_num is None:
                return False
            return round(a_num, 2) == round(gt_num, 2)
        else:
            return normalize_text(answer) == normalize_text(gt)
    except Exception:
        return False


def check_answers_equal(answer1: str, answer2: str, problem_type: str) -> bool:
    if not answer1 or not answer2:
        return False
    try:
        if problem_type == "multiple choice":
            a1 = re.search(r'\b([a-d])\b', normalize_text(answer1))
            a2 = re.search(r'\b([a-d])\b', normalize_text(answer2))
            if a1 and a2:
                return a1.group(1) == a2.group(1)
        elif problem_type == "numerical":
            n1 = normalize_number(answer1)
            n2 = normalize_number(answer2)
            if n1 is not None and n2 is not None:
                return round(n1, 2) == round(n2, 2)
        return normalize_text(answer1) == normalize_text(answer2)
    except Exception:
        return False


def compute_state_action_consistency(verification_state: str, answers_equal: bool) -> float:
    if verification_state == "VERIFIED":
        return 1.0 if answers_equal else 0.0
    elif verification_state == "NOT_VERIFIED":
        return 1.0 if not answers_equal else 0.0
    elif verification_state == "PARTIAL":
        return 0.5
    else:
        return 0.0


def compute_state_gt_consistency(
    verification_state: str,
    init_correct: bool,
    final_correct: bool,
) -> float:
    if init_correct:
        if verification_state == "VERIFIED":
            return 1.0 if final_correct else 0.5
        elif verification_state == "PARTIAL":
            return 0.5
        elif verification_state == "NOT_VERIFIED":
            return 0.0
        else:
            return 0.0
    else:
        if verification_state == "NOT_VERIFIED":
            return 1.0 if final_correct else 0.5
        elif verification_state == "PARTIAL":
            return 0.75 if final_correct else 0.25
        elif verification_state == "VERIFIED":
            return 0.0
        else:
            return 0.0


def compute_verify_reward(
    initial_answer: str,
    final_answer: str,
    ground_truth: str,
    problem_type: str = "multiple choice",
    verification_state: str = "",
    **kwargs
) -> float:
    if not final_answer:
        return -1.0
    if not initial_answer:
        return -0.5

    try:
        init_correct  = check_answer_correct(initial_answer, ground_truth, problem_type)
        final_correct = check_answer_correct(final_answer,   ground_truth, problem_type)

        if final_correct:
            return 1.0
        if init_correct and not final_correct:
            return -0.5
        return 0.0

    except Exception as e:
        if os.getenv("DEBUG_MODE") == "true":
            print(f"Error in compute_verify_reward: {e}")
        return 0.0






def compute_format_reward(output_text: str, **kwargs) -> float:
    if not output_text:
        return 0.0

    try:





        think_open  = len(re.findall(r"<think>",  output_text, re.IGNORECASE))
        think_close = len(re.findall(r"</think>", output_text, re.IGNORECASE))
        if think_open != think_close or think_open == 0:
            return -0.5

        total_score = 0.0


        final_answer_tag = extract_tag(output_text, "final_answer")
        if final_answer_tag and extract_boxed(final_answer_tag):
            total_score += 0.35
        elif final_answer_tag:
            total_score += 0.15


        clue = extract_tag(output_text, "clue")
        if clue and len(clue.strip()) > 3:
            total_score += 0.25


        draft_answer_tag = extract_tag(output_text, "draft_answer")
        if draft_answer_tag and extract_boxed(draft_answer_tag):
            total_score += 0.25
        elif draft_answer_tag:
            total_score += 0.10


        pseudo_queries_tag = extract_tag(output_text, "pseudo_queries")
        if pseudo_queries_tag:
            valid_pqs = [l.strip() for l in pseudo_queries_tag.splitlines() if l.strip()]
            if valid_pqs:
                total_score += 0.15

        return min(1.0, total_score)

    except Exception as e:
        if os.getenv("DEBUG_MODE") == "true":
            print(f"Error in compute_format_reward: {e}")
        return 0.0






def compute_total_reward(
    output_text: str,
    initial_answer: str,
    final_answer: str,
    verification_state: str,
    ground_truth: str,
    problem_type: str,
    return_details: bool = False,
    **kwargs
):
    if output_text is None:
        output_text = ""

    try:
        r_final  = compute_final_reward(
            final_answer=final_answer, ground_truth=ground_truth, problem_type=problem_type
        )
        r_verify = compute_verify_reward(
            initial_answer=initial_answer, final_answer=final_answer,
            verification_state=verification_state,
            ground_truth=ground_truth, problem_type=problem_type
        )
        r_format = compute_format_reward(output_text=output_text)

        total_reward = 0.6 * r_final + 0.25 * r_verify + 0.15 * r_format

        if return_details:
            details = {
                "r_final":  r_final,
                "r_verify": r_verify,
                "r_format": r_format,
                "total":    total_reward,
                "weights":  {"r_final": 0.6, "r_verify": 0.25, "r_format": 0.15},
            }
            return total_reward, details

        return total_reward

    except Exception as e:
        if os.getenv("DEBUG_MODE") == "true":
            print(f"Error in compute_total_reward: {e}")
        if return_details:
            return 0.0, {"r_final": 0.0, "r_verify": 0.0, "r_format": 0.0, "total": 0.0}
        return 0.0






def r_final(completions, solutions, problem_types, **kwargs):
    rewards = []
    for completion, gt, ptype in zip(completions, solutions, problem_types):
        fa_tag = extract_tag(completion, "final_answer")
        fa = extract_boxed(fa_tag) if fa_tag else ""
        rewards.append(compute_final_reward(fa, gt, ptype))
    return rewards


def r_verify(completions, solutions, problem_types, **kwargs):
    rewards = []
    for completion, gt, ptype in zip(completions, solutions, problem_types):

        da_tag = extract_tag(completion, "draft_answer")
        ia = extract_boxed(da_tag) if da_tag else ""

        fa_tag = extract_tag(completion, "final_answer")
        fa = extract_boxed(fa_tag) if fa_tag else ""
        rewards.append(compute_verify_reward(ia, fa, gt, ptype))
    return rewards


def r_format(completions, **kwargs):
    return [compute_format_reward(c) for c in completions]


def total(completions, solutions, problem_types, **kwargs):
    rewards = []
    for completion, gt, ptype in zip(completions, solutions, problem_types):
        da_tag = extract_tag(completion, "draft_answer")
        ia = extract_boxed(da_tag) if da_tag else ""
        fa_tag = extract_tag(completion, "final_answer")
        fa = extract_boxed(fa_tag) if fa_tag else ""
        rewards.append(compute_total_reward(completion, ia, fa, "", gt, ptype))
    return rewards



reward_funcs_registry: Dict[str, Callable] = {
    "r_final":  r_final,
    "r_verify": r_verify,
    "r_format": r_format,
    "total":    total,
}






SYSTEM_PROMPT = """
Based on the video and the user question, determine whether the visual content is sufficient \
to answer the question. If you have enough information, reason using the visual content and \
provide your final answer within \\boxed{}. Otherwise, you may zoom in a specific interval of \
the video related to the question for more details based on your reasoning. Also provide the \
time span of the interval in seconds within \\boxed{[start_time, end_time]}."""






class LazySupervisedDataset(Dataset):

    TYPE_TEMPLATE = {
        "multiple choice": "\nProvide only the single option letter (e.g., A, B, C, D, etc.) as the answer.",
        "numerical":       "\nProvide the numerical value (e.g., 42 or 3.14) without units as the answer.",
        "OCR":             " \nTranscribe text from the image/video clearly and provide your text answer.",
        "free-form":       "\nProvide your text answer.",
        "regression":      "\nProvide the numerical value (e.g., 42 or 3.14) without units as the answer.",
        "tal":             "\nProvide the answer in list format, ie, [start_time, end_time].",
    }

    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super().__init__()
        self.script_args = script_args
        with open(data_path, "r") as f:
            self.list_data_dict = json.load(f)
        self.index = 0

    def __len__(self):
        return len(self.list_data_dict)

    def _make_conversation(self, example):
        if example["problem_type"] == 'multiple choice' and len(example.get("options", [])) > 0:
            question = example['problem'] + "Options:\n"



        else:
            question = example['problem']

        prompt    = question + self.TYPE_TEMPLATE[example['problem_type']]
        zoomin_intervals = example.get('zoomin_intervals', [])
        return prompt, zoomin_intervals

    def __getitem__(self, i):
        num_retries = 10
        try:
            sample = self._get_item(i)
            self.index += 1
            return sample
        except Exception as e:
            print(f"[LazySupervisedDataset] Failed to fetch sample {i}: {e}")


        for attempt in range(num_retries):
            try:
                idx = random.choice(range(len(self)))
                sample = self._get_item(idx)
                self.index += 1
                print(f"[LazySupervisedDataset] Successfully fetched sample {idx} after retry")
                return sample
            except Exception as e:
                print(f"[LazySupervisedDataset] [retry #{attempt + 1}] Failed to fetch sample {idx}: {e}")


        raise RuntimeError(f"Failed to fetch a valid sample after {num_retries} retries")

    def _get_item(self, i):
        source = self.list_data_dict[i]
        video_path = source['path']


        if not os.path.exists(video_path):
            raise RuntimeError(f"Missing video: {video_path}")

        prompt, zoomin_intervals = self._make_conversation(source)
        return {
            'path':             source['path'],
            'prompt':           prompt,
            'solution':         source['solution'],
            'problem_type':     source['problem_type'],
            'data_type':        source['data_type'],
            'tgt':              source.get('tgt'),
            'zoomin_intervals': zoomin_intervals,
        }






def main(script_args, training_args, model_args):


    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]


    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)


    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from think_verify_grpo_trainer import GRPOTrainer as ThinkVerifyGRPOTrainer

    print("Using trainer:", ThinkVerifyGRPOTrainer)


    trainer = ThinkVerifyGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        freeze_vision_modules=model_args.freeze_vision_modules,
    )


    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()


    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
