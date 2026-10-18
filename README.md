# CoVER: See More, Think Deeper

Official implementation of **CoVER** from the paper *See More, Think Deeper*.

Licensed under [Apache-2.0](./LICENSE).

---

## 1. Repository layout

```
.
├── LICENSE
├── README.md
└── CoVER
    ├── src/open_r1/                     # Training entry points
    │   ├── sft_video_cover_phase1.py     # Stage 1: See More SFT (two turns)
    │   ├── sft_video_cover.py            # Stage 2: See More + Think Deeper SFT (three turns)
    │   ├── sft_video.py                 # Generic video SFT baseline
    │   ├── grpo.py                      # Generic GRPO baseline
    │   ├── clip_worker.py               # CLIP feature-extraction worker process
    │   ├── qwen2_5vl_monkey_patch.py    # Qwen2.5-VL flash-attn / forward patches
    │   └── trainer/                     # Custom SFT trainer and video processing
    └── VLMEvalKit/                      # Inference and evaluation framework
        ├── run.py                       # Inference / evaluation entry point
        ├── setup.py
        └── vlmeval/
            ├── config.py                # Model registry (CoVER entries at the end)
            ├── dataset/                 # Per-benchmark loading and scoring
            └── vlm/qwen2_vl/            # CoVER core implementation
                ├── model_cover_infer.py                   # Two-stage inference (CoVER)
                ├── model_cover_draft_infer.py             # Draft-stage inference
                ├── model_sample_frame_pesudo_queries.py  # See More: pseudo-queries + CLIP clip selection
                ├── model_sample_frame_think_verify.py    # Think Deeper: reflection-and-verify clip selection
                ├── model_sample_frame.py                 # Uniform-sampling baseline
                ├── verify_reward.py                      # Stage 3: Think-Then-Verify GRPO entry point
                ├── think_verify_grpo_trainer.py          # Custom GRPO trainer
                └── merge_stage1.py                       # LoRA adapter merging script (edit 3 paths at the top)
```

---

## 2. Environment

```bash
conda create -n cover python=3.10 -y
conda activate cover

pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.49" accelerate deepspeed peft trl datasets
pip install flash-attn --no-build-isolation
pip install decord qwen-vl-utils pillow numpy opencv-python av
pip install nltk rouge-score
```

The evaluation framework can be used in two ways:

```bash
# Option A: run the script directly (recommended, no install needed)
cd CoVER/VLMEvalKit && python run.py --help

# Option B: install as a package
cd CoVER/VLMEvalKit && pip install -e .
```

---

## 3. Model weights

All model paths in the code are relative. Either reproduce the layout below or edit the
paths listed in the table.

```
./Model
├── Qwen2.5-VL-7B-Instruct     # Base model (starting point for Stage 1/2)
├── CoVER-7B                   # CoVER weights produced by Stage 1/2/3
├── CoVER-7B-Draft             # Draft-stage weights
└── Clip-14                    # CLIP vision encoder (used for clip retrieval)
```

| Path | Where to change it |
| --- | --- |
| `./Model/CoVER-7B` | `CoVER/VLMEvalKit/vlmeval/config.py` (entry `Qwen2.5-VL-7B-Instruct`) |
| `./Model/CoVER-7B-Draft` | `CoVER/VLMEvalKit/vlmeval/config.py` (entry `Qwen2.5-VL-7B-CoVER-Draft`) |
| `./Model/Qwen2.5-VL-7B-Instruct` | `config.py` and `--model_name_or_path` in the training commands |
| `./Model/Clip-14` | `CLIP_MODEL_PATH` in `CoVER/src/open_r1/clip_worker.py`, or pass `--clip_model_name` |

---

## 4. Data preparation

### 4.1 Stage 1 / Stage 2 (SFT)

`--dataset_name` points to a `.json` or `.jsonl` file. Each sample carries at least:

```json
{
  "video": "videos/xxx.mp4",
  "original_question": "... (plain-text question, no <image> token)",
  "final_cot": "... (teacher model output containing <pseudo_queries> / <draft_answer> / <clue> / <final_answer>)"
}
```

- `video` is resolved relative to `--data_folder`.
- `final_cot` is parsed into two or three dialogue turns (Stage 1 uses the first two, Stage 2
  uses all three).

### 4.2 Stage 3 (Think-Then-Verify GRPO)

`--dataset_name` points to a `.json` list. Each sample:

```json
{
  "path": "/abs/path/video.mp4",
  "problem": "...",
  "problem_type": "multiple choice | numerical | OCR | free-form | regression | tal",
  "options": ["A. ...", "B. ..."],
  "solution": "...",
  "data_type": "...",
  "tgt": "... (optional, used by temporal-grounding tasks)",
  "zoomin_intervals": [[start, end]]
}
```

- `path` is used as-is and is **not** joined with `--data_folder`; prefer absolute paths.
- `problem_type` selects the prompt template and the reward branch.

### 4.3 Evaluation data

Evaluation paths are hard-coded in `vlmeval/dataset/*.py`. Place or edit them as needed:

| Benchmark | Directory | Registered names |
| --- | --- | --- |
| Video-MME | `./data/Video-MME/` (TSV cache `./cache/videomme`) | `Video-MME_8frame` / `_64frame` / `_1fps` |
| MLVU | `./data/MLVU/video/...` + json (cache `./Dataset`) | `MLVU_8frame` / `_64frame` |
| LVBench | `./data/LVBench` | `LVBench_8frame` / `_64frame` |
| LongVideoBench | `./data/LongVideoBench` | `LongVideoBench_8frame` |

See `CoVER/VLMEvalKit/vlmeval/dataset/video_dataset_config.py` for the full list.

---

## 5. Training

Run every training script with the **repository root** as the working directory — there is
relative-path based `sys.path` bootstrapping inside.

### Stage 1: See More SFT (two turns: pseudo-query retrieval → draft answer)

```bash
accelerate launch --num_processes 8 --main_process_port 29501 \
  CoVER/src/open_r1/sft_video_cover_phase1.py \
  --model_name_or_path ./Model/Qwen2.5-VL-7B-Instruct \
  --dataset_name ./data/sft_stage1.json \
  --data_folder ./data/videos \
  --output_dir ./ckpt/stage1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --bf16 True \
  --gradient_checkpointing True \
  --freeze_vision_modules True \
  --clip_model_name ./Model/Clip-14 \
  --num_clips 3 --clip_duration 30.0 --clip_sim_threshold 0.20
```

### Stage 2: See More + Think Deeper SFT (three turns: adds clues and verification clips)

```bash
accelerate launch --num_processes 8 --main_process_port 29501 \
  CoVER/src/open_r1/sft_video_cover.py \
  --model_name_or_path ./ckpt/stage1 \
  --dataset_name ./data/sft_stage2.json \
  --data_folder ./data/videos \
  --output_dir ./ckpt/stage2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --bf16 True \
  --gradient_checkpointing True \
  --freeze_vision_modules True \
  --clip_model_name ./Model/Clip-14 \
  --verification_num_clips 3 --verification_clip_duration 10.0
```

### Stage 3: Think-Then-Verify GRPO

```bash
python CoVER/VLMEvalKit/vlmeval/vlm/qwen2_vl/verify_reward.py \
  --model_name_or_path ./ckpt/stage2 \
  --dataset_name ./data/grpo_train.json \
  --output_dir ./ckpt/stage3 \
  --reward_funcs total \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --bf16 True \
  --num_fast_frames 768 \
  --num_initial_clips 3 \
  --num_verification_clips 3
```

`--reward_funcs` accepts `r_final` / `r_verify` / `r_format` / `total` (`total` is the
weighted sum of the three and is the default).

### Baselines (optional)

```bash
python CoVER/src/open_r1/sft_video.py  ...   # uniform-sampling video SFT
python CoVER/src/open_r1/grpo.py       ...   # generic GRPO
```

---

## 6. Inference and evaluation

```bash
cd CoVER/VLMEvalKit

# CoVER two-stage inference (registered name from config.py)
python run.py \
  --model Qwen2.5-VL-7B-Instruct \
  --data Video-MME_8frame \
  --work-dir ./outputs \
  --mode all

# Multi-GPU
accelerate launch --num_processes 8 run.py \
  --model Qwen2.5-VL-7B-Instruct \
  --data MLVU_8frame LVBench_8frame \
  --work-dir ./outputs
```

Available model names (from `config.py`):

| Registered name | Class | Description |
| --- | --- | --- |
| `Qwen2.5-VL-7B-Instruct` | `Qwen2VLChatCoVERInfer` | Full CoVER two-stage inference |
| `Qwen2.5-VL-7B-CoVER-Draft` | `Qwen2VLChatCoVERDraftInfer` | Draft stage only |
| `Qwen2.5-VL-7B-Instruct-Sample-Frame` | `Qwen2VLChatSampleFrame` | Uniform-sampling baseline |

Results are written to `<work-dir>/<model>/<timestamp>/`, containing per-dataset prediction
files and an `.xlsx` metric summary.

> `--use-vllm` only applies to the Llama4 family. Do not enable it for CoVER.

---

## 7. Troubleshooting

1. **Always use the repository root as the working directory.** `clip_worker`,
   `verify_reward` and `think_verify_grpo_trainer` import each other through relative paths;
   changing directories raises `ModuleNotFoundError`.
2. **The CLIP worker fails to start.** Training spawns a CLIP worker (`clip_worker.py`) and
   errors out if it is not ready within 300 s. Check that `--clip_model_name` exists and that
   no process is saturating the GPUs.
3. **Only stdout is used.** Nothing is written to disk automatically; redirect the output
   yourself if you need a log file (`> train.log 2>&1`).
4. **Out of memory.** Keep `--per_device_train_batch_size 1` together with
   `--gradient_accumulation_steps`, enable `--gradient_checkpointing`, and consider
   `--freeze_vision_modules True` to reduce the footprint further.
5. **Automatic dataset download fails.** Evaluation data is not downloaded automatically.
   Place it under the directories above, or set the `LMUData` environment variable to a local
   cache root.

---

## 8. Citation

```bibtex
@article{cover2026seemore,
  title   = {See More, Think Deeper},
  author  = {CoVER Authors},
  journal = {arXiv preprint},
  year    = {2026}
}
```
