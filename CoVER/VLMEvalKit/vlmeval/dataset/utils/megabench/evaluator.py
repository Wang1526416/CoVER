import argparse
import json
import os
from typing import Any, Dict, List
import ast
from vlmeval import load, dump


from . import MetricType, AggregationType, ResponseParseType
from .parsing.common.utils import evaluate_as_string


class MEGABenchEvaluator:
    def __init__(
        self,
        subset_name: str,
        responses_file: str,
        output_file: str,
    ):
        self.hf_data = self._load_hf(subset_name)
        self.data = self._load_json(responses_file)
        self.output_file = output_file
        self.tmp_output_file = output_file.replace(".json", "_tmp.pkl")


        self.scoring_functions = {}
        for task_name, task_samples in self.hf_data.items():
            self.scoring_functions[task_name] = ast.literal_eval(
                task_samples[0]["metric_info"]
            )

    def _load_hf(self, subset_name: str) -> List[Dict[str, Any]]:
        from datasets import load_dataset
        dataset = load_dataset("TIGER-Lab/MEGA-Bench", subset_name)["test"]
        task_dict = {}
        for sample in dataset:
            task_name = sample["task_name"]
            if task_name not in task_dict:
                task_dict[task_name] = []
            task_dict[task_name].append(sample)

        return task_dict

    def _get_eval_context(self, task_name, query):
        if "query_idx" in query:
            query_idx = query["query_idx"]
            eval_context = self.hf_data[task_name][query_idx]["eval_context"]
        else:
            global_idx = query["global_idx"]
            global_idx_to_sample = {sample["id"]: sample for sample in self.hf_data[task_name]}
            eval_context = global_idx_to_sample[global_idx]["eval_context"]

        eval_context = ast.literal_eval(eval_context)
        return eval_context

    def evaluate(self):
        if os.path.exists(self.tmp_output_file):
            exist_records = load(self.tmp_output_file)
        else:
            exist_records = {}
        num_tasks = 0
        num_queries = 0
        total_query_score = 0.0
        total_task_score = 0.0


        for task in self.data:
            task_name = task.get("task_name", "")
            if task_name not in exist_records:
                exist_records[task_name] = {}


            score_config = self.scoring_functions.get(
                task_name,
                {
                    "field_score_function": {},
                    "aggregation": {"function": None, "field_weights": {}},
                    "response_parse_function": None,
                },
            )
            if not task.get("query_response"):

                continue

            num_tasks += 1
            task_score_sum = 0.0

            aggregator = AggregationType.from_string(score_config["aggregation"]["function"])
            field_weights = score_config["aggregation"]["field_weights"]


            field_score_functions = score_config.get("field_score_function", {})
            global_aux_metrics = score_config.get("global_aux_metrics", {})
            parser_type_str = score_config.get("response_parse_function", "dummy")
            parser = ResponseParseType.from_string(parser_type_str)


            first_correct = task["query_response"][0]["correct_answer"]
            all_fields = list(first_correct.keys())

            answer_fields = [f for f in all_fields if not f.startswith("##")]


            for idx, query in enumerate(task["query_response"]):
                num_queries += 1
                response_text = query.get("response", "")
                correct_answer = query["correct_answer"]


                response_obj = self._parse_response(
                    task_name,
                    parser,
                    response_text,
                    correct_answer,
                    answer_fields,
                    query,
                    task,
                )

                if idx in exist_records[task_name]:
                    query["scores"] = exist_records[task_name][idx]
                else:

                    query["scores"] = {"field": {}, "info": {}}


                    for fld, fld_metric_name in field_score_functions.items():
                        metric = self._build_metric(fld_metric_name, score_config)
                        self._evaluate_field(
                            task_name,
                            metric,
                            fld,
                            response_obj,
                            correct_answer,
                            query
                        )


                    for fld, fld_metric_name in global_aux_metrics.items():
                        metric = self._build_metric(fld_metric_name, score_config)


                        tmp_obj = {fld: response_obj}
                        self._evaluate_field(
                            task_name,
                            metric,
                            fld,
                            tmp_obj,
                            correct_answer,
                            query,
                            is_aux=True,
                        )

                    exist_records[task_name][idx] = query["scores"]
                    if idx % 10 == 0 or idx == len(task["query_response"]) - 1:
                        dump(exist_records, self.tmp_output_file)


                query["scores"]["query"] = aggregator.aggregate(
                    query["scores"]["field"],
                    field_weights,
                )

                if query["scores"]["query"] >= 0:
                    task_score_sum += query["scores"]["query"]


            if task["query_response"]:
                mean_score = task_score_sum / len(task["query_response"])
            else:
                mean_score = 0.0
            task["task_score"] = task_score_sum
            task["mean_task_score"] = mean_score

            total_query_score += task_score_sum
            total_task_score += mean_score

            print(f"[Task: {task_name}] Score = {task_score_sum} / {len(task['query_response'])}")


        summary = {}
        if num_tasks > 0:
            macro_mean_score = total_task_score / num_tasks
            summary["macro_mean_score"] = macro_mean_score
        else:
            summary["macro_mean_score"] = 0.0

        if num_queries > 0:
            micro_mean_score = total_query_score / num_queries
            summary["micro_mean_score"] = micro_mean_score
        else:
            summary["micro_mean_score"] = 0.0

        summary["num_tasks"] = num_tasks
        summary["num_queries"] = num_queries



        output_data = {
            "data": self.data,
            "summary": summary,
        }
        self._save_results(self.output_file, output_data)
        print(f"Evaluation complete! Results saved to {self.output_file}")

    def _evaluate_field(
        self,
        task_name: str,
        metric: Any,
        field: str,
        response_obj: Dict[str, Any],
        correct_answer: Dict[str, Any],
        query: Dict[str, Any],
        is_aux: bool = False,
    ) -> float:
        eval_context = self._get_eval_context(task_name, query)

        if metric == MetricType.UNSUPPORTED:
            print(f"The metric for {field} in task {task_name} is not supported")
            return 0.0
        elif metric == MetricType.SYMBOLIC_PLANNING_TEST or metric == MetricType.PROGRAM_JUDGE:
            query["scores"]["field"][field] = metric.match(
                response_obj.get(field),
                eval_context,
            )
        elif metric == MetricType.CONSTRAINED_GENERATION:
            score, eval_info = metric.match(response_obj, eval_context)
            query["scores"]["field"][field] = score
            query["scores"]["info"][field] = eval_info
        elif metric == MetricType.XML_NORM_POINT_IN_BBOX:
            score, eval_info = metric.match(response_obj.get(field), eval_context)
            query["scores"]["field"][field] = score
            query["scores"]["info"][field] = eval_info
        elif isinstance(metric, MetricType.VLM_AS_JUDGE.class_impl):
            images = query.get("images", [])
            question = query.get("question", "")
            correct_val = correct_answer.get(field, "") if not is_aux else correct_answer
            response_info = (
                response_obj.get(field)
                if isinstance(response_obj, dict)
                else response_obj
            )
            query["scores"]["field"][field] = metric.match(
                response_info,
                correct_val,
                images=images,
                question=question,
                eval_context=eval_context,
            )
        else:
            correct_val = correct_answer.get(field, "") if not is_aux else correct_answer
            correct_val = evaluate_as_string(correct_val)
            predicted_val = response_obj.get(field, "")
            query["scores"]["field"][field] = metric.match(predicted_val, correct_val)

    def _parse_response(
        self,
        task_name: str,
        parser,
        response_text: str,
        correct_answer: Dict[str, Any],
        answer_fields: List[str],
        query: Dict[str, Any],
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        res_parsing_pass = True
        if parser.is_single_field_parser():

            assert (
                len(answer_fields) == 1
            ), "The answer_string parse must be used when the answer has a single field"
            answer_key = answer_fields[0]

            global_description = task["task_description"]
            query_question = query["question"]
            is_single_line_ans = "\n" not in correct_answer[answer_key]

            response_obj = parser.parse(
                response_text,
                answer_key,
                global_description=global_description,
                query_question=query_question,
                is_single_line_ans=is_single_line_ans,
            )
            assert isinstance(response_obj[answer_key], str), "Single-field parsing results must be string"
        else:

            response_obj = parser.parse(response_text)

            if parser == ResponseParseType.JSON and (
                not isinstance(response_obj, dict) or not response_obj
            ):


                res_parsing_pass = False
                response_obj = {}
                for field in correct_answer:
                    response_obj[field] = response_text

        if not res_parsing_pass:
            print(
                f"Task:{task_name}, cannot parse query with global idx {query['global_idx']}"
            )
        return response_obj

    def _build_metric(self, metric_name: str, score_config: Dict[str, Any]):
        metric = MetricType.from_string(metric_name)
        if metric == MetricType.VLM_AS_JUDGE:

            gpt4o_configs = score_config.get("gpt4o_eval_configs", {})
            metric = metric.class_impl(gpt4o_configs)
        elif metric == MetricType.ASCII_ART_GPT4O_JUDGE:

            ascii_art_configs = score_config.get("ascii_art_eval_configs", {})
            metric = metric.class_impl(ascii_art_configs)
        return metric

    @staticmethod
    def _load_json(file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_results(file_path: str, data: Any) -> None:
        temp_filename = f"{file_path}.tmp"
        with open(temp_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_filename, file_path)


def main():
    parser = argparse.ArgumentParser(description="Simple Evaluator")
    parser.add_argument(
        "--subset_name",
        type=str,
        required=True,
        help="The subset of MEGA-Bench to evaluate.",
    )
    parser.add_argument(
        "--submission_file",
        type=str,
        required=True,
        help="Path to a JSON file containing model responses.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Where to store the evaluation results (JSON).",
    )

    args = parser.parse_args()
    evaluator = MEGABenchEvaluator(
        subset_name=args.subset_name,
        responses_file=args.submission_file,
        output_file=args.output_file,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
