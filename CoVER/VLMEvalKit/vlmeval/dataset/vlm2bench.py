
import os
import pandas as pd
from .image_base import ImageBaseDataset
from .utils.vlm2bench import (
    common_process_results,
    tf_pair_aggregate_accuracy,
    cnt_aggregate_metric,
    grp_aggregate_accuracy,
)


class VLM2Bench(ImageBaseDataset):
    TYPE = "VQA"

    DATASET_URL = {
        "VLM2Bench": 'https://huggingface.co/datasets/Sterzhang/vlm2-bench/resolve/main/VLM2Bench_img.tsv'
    }

    DATASET_MD5 = {'VLM2Bench': '16f474bfc4e269c583468bf89139da8f'}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]


        if "image_path" not in line:
            img_field = line.get("image")

            image_paths = [f"{line['index']}_{i}.jpg" for i in range(len(img_field))]
            line["image_path"] = image_paths

            line["image"] = img_field


        img_paths = self.dump_image(line)
        if not isinstance(img_paths, list):
            img_paths = [img_paths]


        prompt = f"Question: {line['question']}\n"


        msgs = [{"type": "image", "value": p} for p in img_paths]
        msgs.append({"type": "text", "value": prompt})
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        model = judge_kwargs.get("model")
        if model:
            suffix = eval_file.split('.')[-1]
            storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
            score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.csv')
            tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
            if os.path.exists(storage):
                if storage.lower().endswith(".xlsx"):
                    data = pd.read_excel(storage)
                else:
                    data = pd.read_csv(storage, sep="\t", encoding="latin1", engine="python")
            else:
                if eval_file.lower().endswith(".xlsx"):
                    data = pd.read_excel(eval_file)
                else:
                    data = pd.read_csv(eval_file, sep="\t", encoding="latin1", engine="python")
        else:
            if eval_file.lower().endswith(".xlsx"):
                data = pd.read_excel(eval_file)
            else:
                data = pd.read_csv(eval_file, sep="\t", encoding="latin1", engine="python")

        results = data.to_dict(orient="records")
        processed = common_process_results(results)


        for rec in processed:
            if rec.get("category", "").lower() in ["oc-cnt", "pc-cnt"]:
                try:
                    rec["image_seq_len"] = len(rec["image"])
                except Exception as e:
                    rec["image_seq_len"] = 2

        eval_scores = {}
        for cat in sorted(set([r["category"] for r in processed])):
            sub_results = [r for r in processed if r["category"] == cat]
            if cat in ["gc-mat", "gc-trk", "oc-cpr", "pc-cpr"]:
                score = tf_pair_aggregate_accuracy(sub_results)
            elif cat in ["oc-cnt", "pc-cnt"]:
                score = cnt_aggregate_metric(sub_results)
            elif cat in ["oc-grp", "pc-grp"]:
                score = grp_aggregate_accuracy(sub_results)
            else:
                score = None
            eval_scores[cat] = score

        score_df = pd.DataFrame({k: [v] for k, v in eval_scores.items()})
        if model:
            final_score_file = score_file
        else:
            suffix = os.path.splitext(eval_file)[1]
            final_score_file = eval_file.replace(suffix, "_score.csv")
        score_df.to_csv(final_score_file, index=False)
        return score_df
