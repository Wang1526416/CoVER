# Copyright 2020-2025 The HuggingFace Team. All rights reserved.

import os
import re
import textwrap
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import transformers
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_datasets_available, is_peft_available
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_deepspeed_available, is_liger_kernel_available, is_rich_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    print_prompt_completions_sample,
)

try:
    from .vision_process import process_vision_info, get_video_hw
except ImportError:
    from vision_process import process_vision_info, get_video_hw

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import multiprocessing as mp
import uuid
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_reward import compute_total_reward
from clip_worker import worker_main, CLIP_MODEL_PATH

if is_deepspeed_available():
    import deepspeed

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
FONT_PATH = "./CoVER/Arial.Unicode.ttf"


def add_frame_numbers_batch(frame_batch, frame_numbers):
    results = []
    for frame, frame_number in zip(frame_batch, frame_numbers):
        numbered_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(numbered_frame)
        width, height = numbered_frame.size
        font_size = int(min(width, height) / 8)
        try:
            font = ImageFont.truetype(FONT_PATH, size=font_size)
        except OSError:
            font = ImageFont.load_default()
        text = str(frame_number)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0] + 10
        text_height = text_bbox[3] - text_bbox[1] + 10
        x = width - text_width
        y = height - text_height - text_height / 3
        draw.text((x, y), text, font=font, fill='red')
        results.append(torch.tensor(np.array(numbered_frame)))
    results = torch.stack(results).permute(0, 3, 1, 2)
    return results


def iou_1d(a, b):
    start_a, end_a = a
    start_b, end_b = b
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    if inter_start >= inter_end:
        return 0.0
    inter_length = inter_end - inter_start
    union_length = (end_a - start_a) + (end_b - start_b) - inter_length
    return inter_length / union_length if union_length != 0 else 0.0


def overlap_1d(a, b):
    start_a, end_a = a
    start_b, end_b = b
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    if inter_start >= inter_end:
        return 0.0
    inter_length = inter_end - inter_start
    union_length = end_b - start_b
    return inter_length / union_length if union_length != 0 else 0.0


def find_assistant_spans(input_ids, start_tokens=[151644, 77091, 198], end_token=151645):
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("input_ids must be a torch.Tensor")
    device = input_ids.device
    seq_len = input_ids.shape[0]
    spans = []
    start_tensor = torch.tensor(start_tokens, device=device)
    start_len = len(start_tensor)
    i = 0
    while i <= seq_len - start_len:
        if torch.equal(input_ids[i:i + start_len], start_tensor):
            j = i + start_len
            while j < seq_len:
                if input_ids[j].item() == end_token:
                    spans.append((i + len(start_tokens), j + 1))
                    break
                if j == seq_len - 1:
                    spans.append((i + len(start_tokens), j + 1))
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return spans


class RepeatRandomSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i: i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance *= count / (count - 1)
    return torch.sqrt(variance)


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class GRPOTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    SYSTEM_PROMPT = (
        "You are a helpful video analyst. "
        "The red numbers overlaid on each frame represent the timestamp in seconds. "
        "You can refer to them for temporal grounding."
    )

    TURN1_USER_SUFFIX = """\
Video details:
- Duration: {duration:.1f}s  |  FPS: {fps:.2f}  |  Total frames: {total_frames}

Question: {question}

Watch the full video carefully.
- If the visual evidence is sufficient, provide:
  (1) your reasoning in <think>...</think>
  (2) a draft answer in <draft_answer>\\boxed{{X}}</draft_answer> where X is EXACTLY one letter from the given options (A/B/C/D/E). Do NOT write anything other than a single letter inside \\boxed{{}}.
  (3) a single discriminative visual clue in <clue>...</clue>

- If the evidence is insufficient, provide:
  (1) your reasoning in <think>...</think>
  (2) exactly 3 pseudo-queries in <pseudo_queries>...</pseudo_queries>.

      Pseudo-queries are used to retrieve the missing visual evidence needed to answer
      the question. Each query will be encoded by a retrieval model (CLIP) and its
      text-frame similarity scores will be computed across all video frames. The frames
      with the highest similarity scores will be selected as zoom-in clips and shown
      to you in the next turn for further reasoning.

      Each query must target ONE specific missing visual fact (object identity,
      human action, object state, or temporal relation).
      Write them as direct English questions. Do NOT be vague.
"""

    TURN2_B_USER_SUFFIX = """\
You are given two streams of visual input above:
- **Full video (low resolution)**: the complete video at reduced resolution for global context.
- **Zoom-in clips (high resolution)**: the most relevant segments retrieved using your pseudo-queries, \
selected by computing text-frame similarity between each query and all video frames.

Based on these clips, reason over the visual evidence in <think>...</think>. \
In your reasoning, use the full video for temporal context and the zoom-in clips for fine-grained details. \
Also state what observable facts should be visible in the video \
if your answer is correct (i.e. form a verifiable hypothesis). \
Then provide your candidate answer in <draft_answer>\\boxed{{X}}</draft_answer> where X is EXACTLY one letter \
from the given options (A/B/C/D/E). Do NOT write anything other than a single letter inside \\boxed{{}}. \
Then extract one short, visually checkable clue in <clue>...</clue> \
that distinguishes this answer from other options.

You MUST output <draft_answer>\\boxed{{X}}</draft_answer> — even if uncertain, pick the most likely option.
You MUST also output <clue>...</clue> after <draft_answer>.
Do NOT output <pseudo_queries> in this turn. You have already retrieved the zoom-in clips; now you must commit to an answer.

Question: {question}
"""

    TURN_VERIFY_USER = """\
You are given two streams of visual input above:
- **Full video (low resolution)**: the complete video at reduced resolution for global context.
- **Verification clips (high resolution)**: segments retrieved by computing CLIP similarity \
between your clue and all video frames, then selecting the highest-scoring frames.

In <think>, use the full video for temporal context and the verification clips for fine-grained \
checking. Verify whether the visual evidence in the clips supports your clue and candidate answer.
- If the clips provide clear and sufficient evidence that your candidate answer is wrong, revise to the correct answer.
- If the evidence is ambiguous, unclear, or does not strongly contradict your candidate answer, keep your original candidate answer unchanged.
Then output the final answer in <final_answer>\\boxed{{X}}</final_answer> where X is EXACTLY one letter \
from the given options (A/B/C/D/E). Do NOT write anything other than a single letter inside \\boxed{{}}.

You MUST output <final_answer>\\boxed{{X}}</final_answer> — no exceptions, no refusals. \
When in doubt, trust your original candidate answer.
"""

    def _start_clip_worker(self):
        import time
        if hasattr(self, "_clip_worker_process") and self._clip_worker_process is not None:
            return
        rank = getattr(self.accelerator, 'process_index', '?') if hasattr(self, 'accelerator') else 0
        ctx = mp.get_context("spawn")
        self._clip_request_queue = ctx.Queue()
        self._clip_response_queue = ctx.Queue()
        self._clip_worker_process = ctx.Process(
            target=worker_main,
            args=(self._clip_request_queue, self._clip_response_queue, CLIP_MODEL_PATH),
            daemon=False,
        )
        self._clip_worker_process.start()

        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                msg = self._clip_response_queue.get(timeout=5)
                if msg.get("ready"):
                    print(f"[Think-Verify] Rank {rank} CLIP worker ready, pid={msg.get('pid')}")
                    break
            except Exception:
                pass
            if not self._clip_worker_process.is_alive():
                raise RuntimeError("[Think-Verify] CLIP worker process died during startup")
        else:
            raise RuntimeError("[Think-Verify] CLIP worker did not send ready signal within 120s")
        print(f"[Think-Verify] Rank {rank} worker alive: {self._clip_worker_process.is_alive()}")

    def _stop_clip_worker(self):
        if hasattr(self, "_clip_request_queue") and self._clip_request_queue is not None:
            self._clip_request_queue.put(None)
        if hasattr(self, "_clip_worker_process") and self._clip_worker_process is not None:
            self._clip_worker_process.join(timeout=5)
            if self._clip_worker_process.is_alive():
                self._clip_worker_process.terminate()

    def _retrieve_clips_via_worker(self, video_path, queries, n_clips, clip_duration,
                                    mode="zoom", question="", sim_threshold=0.15,
                                    num_fast_frames=768):
        if not hasattr(self, "_clip_worker_process") or self._clip_worker_process is None or not self._clip_worker_process.is_alive():
            self._clip_worker_process = None
            self._clip_request_queue = None
            self._clip_response_queue = None
            self._start_clip_worker()

        req_id = str(uuid.uuid4())
        self._clip_request_queue.put({
            "req_id": req_id,
            "video_path": video_path,
            "queries": queries,
            "n_clips": n_clips,
            "clip_duration": clip_duration,
            "mode": mode,
            "question": question,
            "sim_threshold": sim_threshold,
            "num_fast_frames": num_fast_frames,
        })

        while True:
            msg = self._clip_response_queue.get(timeout=180)

            if msg.get("ready"):
                continue
            if msg.get("req_id") != req_id:
                continue
            if not msg["ok"]:
                raise RuntimeError(msg["error"])
            return msg["result"]

    def _extract_hires_frames_from_clips(self, clips, fps, total_frames, vreader, hi_res_token_budget=128):
        valid_clips, hires_frames = [], []
        for clip in clips:
            try:
                start_f = max(0, int(clip["start"] * fps))
                end_f = min(total_frames - 1, int(clip["end"] * fps))
                num_f = min(int((clip["end"] - clip["start"]) * 2), 8)
                fids = torch.linspace(start_f, end_f, num_f).round().long().tolist()
                raw = vreader.get_batch(fids).asnumpy()
                secs = [int(fid / fps) for fid in fids]
                slow_v = add_frame_numbers_batch(raw, secs)
                H, W = slow_v.shape[-2:]
                rh, rw = get_video_hw(H, W, hi_res_token_budget * 28 * 28)
                slow_v = transforms.functional.resize(
                    slow_v, [rh, rw], interpolation=InterpolationMode.BICUBIC, antialias=True
                ).float()
                valid_clips.append(clip)
                hires_frames.append(slow_v)
            except Exception as e:
                print(f"[CLIP retrieve] hi-res extraction failed: {e}")
        return valid_clips, hires_frames

    @property
    def clip_model(self):
        raise AttributeError("clip_model is no longer used. Use CLIP worker instead.")

    @property
    def clip_processor(self):
        raise AttributeError("clip_processor is no longer used. Use CLIP worker instead.")

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        freeze_vision_modules: Optional[bool] = True,
        script_args=None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        if isinstance(model, str):
            model_id = model
            torch_dtype = torch.bfloat16
            if isinstance(torch_dtype, str) and torch_dtype not in ["auto", None]:
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")

            _is_qwen25_vl = False
            _is_qwen2_vl = False
            try:
                import json as _json
                with open(os.path.join(model_id, "config.json")) as _f:
                    _cfg = _json.load(_f)
                _model_type = _cfg.get("model_type", "").lower()
                if "qwen2_5_vl" in _model_type:
                    _is_qwen25_vl = True
                elif "qwen2_vl" in _model_type:
                    _is_qwen2_vl = True
            except Exception:

                _basename = os.path.basename(model_id).lower()
                if "qwen2.5-vl" in _basename or "qwen2_5_vl" in _basename:
                    _is_qwen25_vl = True
                elif "qwen2-vl" in _basename:
                    _is_qwen2_vl = True
            if _is_qwen25_vl:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif _is_qwen2_vl:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError("You passed `model_init_kwargs` but model is already instantiated.")

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`.")
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in list(lora_module_names):
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, ['visual'])
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        if freeze_vision_modules:
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in ['visual']):
                    p.requires_grad = False

        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        if processing_class is None:
            if _is_qwen25_vl or _is_qwen2_vl:
                processing_class = AutoProcessor.from_pretrained(model_id)
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError("Number of reward weights must match number of reward functions")
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        elif len(reward_processing_classes) != len(reward_funcs):
            raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        self.shuffle_dataset = args.shuffle_dataset

        self._clip_worker_process = None
        self._clip_request_queue = None
        self._clip_response_queue = None

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values()))
        ):
            raise NotImplementedError("Iterable datasets are not yet supported in GRPOTrainer.")

        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self._step = 0
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        model.warnings_issued["estimate_tokens"] = True

        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError("Liger is required to use `liger_loss`.")
            if is_peft_model(model):
                raise TypeError("Liger loss is not supported with a PEFT model.")
            if self.loss_type != "bnpo":
                raise ValueError("Only bnpo is supported with liger loss.")
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.ref_model is not None,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print

        if script_args is not None:
            self.num_fast_frames = getattr(script_args, 'num_fast_frames', 768)
            self.num_initial_clips = getattr(script_args, 'num_initial_clips', 3)
            self.num_verification_clips = getattr(script_args, 'num_verification_clips', 3)
        else:
            self.num_fast_frames = 768
            self.num_initial_clips = 3
            self.num_verification_clips = 3

        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Valid values: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Valid values: {possible_values}."
                )

        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError("vLLM is not available and `use_vllm` is set to True.")
            if self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                )
                self.vllm_client.init_communicator()
            self.guided_decoding_regex = args.vllm_guided_decoding_regex
            self._last_loaded_step = -1
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        self.model_accepts_loss_kwargs = False
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self._start_clip_worker()

    def __del__(self):
        pass

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        model.config.use_cache = False
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )
        if use_reentrant:
            model.enable_input_require_grads()
        return model

    @profiling_decorator
    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep=None):
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]
        return last_hidden_state

    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    @profiling_decorator
    def _move_model_to_vllm(self):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        if is_peft_model(self.model):
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()
                for name, param in self.model.named_parameters():
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                self.model.unmerge_adapter()
        else:
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(accumulated_local_batch)
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        else:
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        self._step += 1
        return inputs

    def _extract_tag(self, text: str, tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _extract_boxed(self, text: str) -> str:
        m = re.search(r"\\boxed\{\{(.*?)\}\}", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _build_dummy_generation_result(self, device):
        prompt_ids = torch.tensor([self.processing_class.pad_token_id], dtype=torch.long, device=device)
        prompt_mask = torch.tensor([1], dtype=torch.long, device=device)
        completion_ids = torch.tensor([self.processing_class.eos_token_id], dtype=torch.long, device=device)
        completion_mask = torch.tensor([1], dtype=torch.long, device=device)
        loss_mask = torch.tensor([0], dtype=torch.long, device=device)
        multimodal_inputs = {
            "pixel_values": None,
            "image_grid_thw": None,
            "pixel_values_videos": None,
            "video_grid_thw": None,
        }
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "loss_mask": loss_mask,
            "multimodal_inputs": multimodal_inputs,
            "full_output_text": "",
            "completion_text": "",
            "reward_map": {
                "r_final": 0.0, "r_verify": 0.0, "r_format": 0.0,
                "accuracy_reward": 0.0, "verify_reward": 0.0, "format_reward": 0.0,
                "total": 0.0, "total_reward": 0.0,
            },
            "total_reward": 0.0,
            "r_final": 0.0,
            "r_verify": 0.0,
            "r_format": 0.0,
            "final_answer": "",
            "verification_state": "",
            "prompt_text_for_log": "",
        }

    def _run_generation(self, messages, video_fps):
        images, videos = process_vision_info([messages])
        video_kwargs = {"fps": video_fps} if video_fps else {}
        prompt_text = [self.processing_class.apply_chat_template(messages, add_generation_prompt=True)]
        prompt_inputs = self.processing_class(
            text=prompt_text,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **video_kwargs,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)



        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                full_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=self.generation_config,
                    use_cache=True,
                )
        finally:
            if was_training:
                self.model.train()
        completion_ids = full_ids[:, prompt_length:]
        output_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)[0]
        return output_text, prompt_ids, prompt_mask, full_ids, prompt_inputs

    def _clip_retrieve(self, video_path, fps, total_frames, queries, n_clips, clip_duration,
                        hi_res_token_budget, vreader, mode="zoom", question="",
                        sim_threshold=0.15, num_fast_frames=768):
        try:
            worker_result = self._retrieve_clips_via_worker(
                video_path=video_path,
                queries=queries,
                n_clips=n_clips,
                clip_duration=clip_duration,
                mode=mode,
                question=question,
                sim_threshold=sim_threshold,
                num_fast_frames=num_fast_frames,
            )
            clips = worker_result["clips"]
            fps = worker_result["fps"]
            clips, hires_frames = self._extract_hires_frames_from_clips(
                clips=clips,
                fps=fps,
                total_frames=total_frames,
                vreader=vreader,
                hi_res_token_budget=hi_res_token_budget,
            )
            return clips, hires_frames
        except Exception as e:
            import traceback
            print(f"[CLIP retrieve] Worker failed: {e}")
            traceback.print_exc()
            clips = []
            step = (total_frames / fps) / max(1, n_clips)
            for i in range(n_clips):
                start_t = i * step
                end_t = min(start_t + clip_duration, total_frames / fps)
                clips.append({
                    "start": round(start_t, 2),
                    "end": round(end_t, 2),
                    "query": queries[min(i, len(queries) - 1)],
                    "similarity": 0.0,
                })
            return self._extract_hires_frames_from_clips(
                clips=clips,
                fps=fps,
                total_frames=total_frames,
                vreader=vreader,
                hi_res_token_budget=hi_res_token_budget,
            )


    def _rollout_one_generation(self, sample, gen_seed: int):
        device = self.accelerator.device
        video_path = sample.get("path", "")
        prompt = sample.get("prompt", "")
        sol_raw = sample.get("solution", "")
        gt = sol_raw[len("<answer>"): -len("</answer>")] if sol_raw.startswith("<answer>") else sol_raw

        with torch.random.fork_rng(devices=None):
            torch.manual_seed(gen_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(gen_seed)

            try:
                vreader = VideoReader(video_path, ctx=cpu(0))
            except Exception as e:
                print(f"[Think-Verify] Error loading video {video_path}: {e}")
                return self._build_dummy_generation_result(device)

            if vreader is None:
                return self._build_dummy_generation_result(device)

            fps = vreader.get_avg_fps()
            duration = len(vreader) / fps
            total_frames = len(vreader)





            fast_video_resolution = 32
            factor = 2 if duration < 120 else 1
            num_frames_global = min(int(duration * 2), self.num_fast_frames)
            num_frames_global = min(num_frames_global, total_frames)
            fast_frame_ids = torch.linspace(0, total_frames - 1, num_frames_global).round().long().tolist()
            try:
                raw_frames = vreader.get_batch(fast_frame_ids).asnumpy()
            except Exception as e:
                print(f"[Think-Verify] decord get_batch failed for {video_path} (retry with num_threads=1): {e}")
                try:
                    vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                    raw_frames = vreader.get_batch(fast_frame_ids).asnumpy()
                except Exception as e2:
                    print(f"[Think-Verify] decord get_batch retry also failed for {video_path}: {e2}")
                    return self._build_dummy_generation_result(device)
            fast_frame_seconds = [int(fid / fps) for fid in fast_frame_ids]
            fast_video = add_frame_numbers_batch(raw_frames, fast_frame_seconds)
            H, W = fast_video.shape[-2:]
            resized_h, resized_w = get_video_hw(H, W, fast_video_resolution * 28 * 28 * factor)
            fast_video = transforms.functional.resize(
                fast_video, [resized_h, resized_w], interpolation=InterpolationMode.BICUBIC, antialias=True
            ).float()
            fast_sample_fps = round(num_frames_global / max(1, total_frames - 1) * fps, 2)




            turn1_user_text = self.TURN1_USER_SUFFIX.format(
                duration=duration, fps=fps,
                total_frames=total_frames, question=prompt,
            )
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
                        {"type": "video", "video": fast_video},
                        {"type": "text",  "text":  turn1_user_text},
                    ],
                },
            ]

            all_video_fps = [fast_sample_fps]
            turn1_output, _, _, _, _ = self._run_generation(messages, video_fps=all_video_fps)
            full_output_text = turn1_output + "\n\n"
            messages.append({"role": "assistant", "content": turn1_output})


            print(f"[CoVER][DEBUG] Turn1 output preview: {repr(turn1_output[:300])}")


            pseudo_queries_text = self._extract_tag(turn1_output, "pseudo_queries")
            draft_answer_t1     = self._extract_tag(turn1_output, "draft_answer")
            clue_t1             = self._extract_tag(turn1_output, "clue")

            is_pattern_b = bool(pseudo_queries_text.strip()) and not bool(draft_answer_t1.strip())

            if is_pattern_b:
                print(f"[CoVER] Pattern B (insufficient evidence, 3 turns)")
                print(f"[CoVER-B] Turn1 output length: {len(turn1_output)}, pq_lines count: {len([l for l in pseudo_queries_text.splitlines() if l.strip()])}")


                pq_lines = [
                    re.sub(r'^\d+\.\s*', '', l).strip()
                    for l in pseudo_queries_text.splitlines()
                    if l.strip()
                ]
                if not pq_lines:
                    pq_lines = [pseudo_queries_text.strip()]


                print(f"[CoVER-B] Starting CLIP zoom-in retrieval, queries={pq_lines[:1]}...")
                zoom_clips, zoom_hires_frames = self._clip_retrieve(
                    video_path=video_path,
                    fps=fps,
                    total_frames=total_frames,
                    queries=pq_lines,
                    n_clips=self.num_initial_clips,
                    clip_duration=30.0,
                    hi_res_token_budget=128,
                    vreader=vreader,
                    mode="zoom",
                    question=prompt,
                    sim_threshold=0.15,
                    num_fast_frames=num_frames_global,
                )
                print(f"[CoVER-B] CLIP zoom-in done, got {len(zoom_clips)} clips")


                turn2_user_content = [
                    {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
                    {"type": "video", "video": fast_video},
                    {"type": "text",  "text":  "Retrieved zoom-in clips (selected via pseudo-query similarity):"},
                ]
                turn2_video_fps = [fast_sample_fps]
                for clip_info, clip_tensor in zip(zoom_clips, zoom_hires_frames):
                    turn2_user_content.append({
                        "type": "text",
                        "text": f"[{clip_info['start']:.1f}s – {clip_info['end']:.1f}s]",
                    })
                    turn2_user_content.append({"type": "video", "video": clip_tensor})
                    turn2_video_fps.append(2.0)
                turn2_user_content.append({
                    "type": "text",
                    "text": self.TURN2_B_USER_SUFFIX.format(question=prompt),
                })
                messages.append({"role": "user", "content": turn2_user_content})


                all_video_fps.append(fast_sample_fps)
                all_video_fps.extend([2.0] * len(zoom_hires_frames))
                print(f"[CoVER-B] Starting Turn2 generation, all_video_fps len={len(all_video_fps)}")
                turn2_output, _, _, _, _ = self._run_generation(messages, video_fps=all_video_fps)
                print(f"[CoVER-B] Turn2 generation done, output length: {len(turn2_output)}")
                full_output_text += turn2_output + "\n\n"
                messages.append({"role": "assistant", "content": turn2_output})


                draft_answer = self._extract_tag(turn2_output, "draft_answer")
                clue         = self._extract_tag(turn2_output, "clue")
                if not draft_answer:
                    draft_answer = self._extract_boxed(turn2_output)


                verify_query = clue if clue else "What visual evidence confirms the answer?"
                print(f"[CoVER-B] Starting CLIP verify retrieval, query='{verify_query[:60]}'...")
                verify_clips, verify_hires_frames = self._clip_retrieve(
                    video_path=video_path,
                    fps=fps,
                    total_frames=total_frames,
                    queries=[verify_query],
                    n_clips=self.num_verification_clips,
                    clip_duration=10.0,
                    hi_res_token_budget=128,
                    vreader=vreader,
                    mode="verify",
                    sim_threshold=0.15,
                    num_fast_frames=num_frames_global,
                )
                print(f"[CoVER-B] CLIP verify done, got {len(verify_clips)} clips")


                turn3_user_content = [
                    {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
                    {"type": "video", "video": fast_video},
                    {"type": "text",  "text":  "Verification clips (retrieved via clue similarity):"},
                ]
                turn3_video_fps = [fast_sample_fps]
                for clip_info, clip_tensor in zip(verify_clips, verify_hires_frames):
                    turn3_user_content.append({
                        "type": "text",
                        "text": f"[{clip_info['start']:.1f}s – {clip_info['end']:.1f}s]",
                    })
                    turn3_user_content.append({"type": "video", "video": clip_tensor})
                    turn3_video_fps.append(2.0)
                turn3_user_content.append({"type": "text", "text": self.TURN_VERIFY_USER})
                messages.append({"role": "user", "content": turn3_user_content})


                all_video_fps.append(fast_sample_fps)
                all_video_fps.extend([2.0] * len(verify_hires_frames))
                print(f"[CoVER-B] Starting Turn3 generation, all_video_fps len={len(all_video_fps)}")
                last_output, last_prompt_ids, last_prompt_mask, last_full_ids, last_prompt_inputs = \
                    self._run_generation(messages, video_fps=all_video_fps)
                print(f"[CoVER-B] Turn3 generation done, output length: {len(last_output)}")
                full_output_text += last_output
                last_messages = messages

            else:





                print(f"[CoVER] Pattern A (sufficient evidence, 2 turns)")
                print(f"[CoVER-A] Turn1 output length: {len(turn1_output)}, draft='{draft_answer_t1[:30]}', clue='{clue_t1[:40]}'")

                draft_answer = draft_answer_t1
                clue         = clue_t1
                if not draft_answer:
                    draft_answer = self._extract_boxed(turn1_output)


                verify_query = clue if clue else "What visual evidence confirms the answer?"
                print(f"[CoVER-A] Starting CLIP verify retrieval, query='{verify_query[:60]}'...")
                verify_clips, verify_hires_frames = self._clip_retrieve(
                    video_path=video_path,
                    fps=fps,
                    total_frames=total_frames,
                    queries=[verify_query],
                    n_clips=self.num_verification_clips,
                    clip_duration=10.0,
                    hi_res_token_budget=128,
                    vreader=vreader,
                    mode="verify",
                    sim_threshold=0.15,
                    num_fast_frames=num_frames_global,
                )
                print(f"[CoVER-A] CLIP verify done, got {len(verify_clips)} clips")


                turn2_user_content = [
                    {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
                    {"type": "video", "video": fast_video},
                    {"type": "text",  "text":  "Verification clips (retrieved via clue similarity):"},
                ]
                turn2_video_fps = [fast_sample_fps]
                for clip_info, clip_tensor in zip(verify_clips, verify_hires_frames):
                    turn2_user_content.append({
                        "type": "text",
                        "text": f"[{clip_info['start']:.1f}s – {clip_info['end']:.1f}s]",
                    })
                    turn2_user_content.append({"type": "video", "video": clip_tensor})
                    turn2_video_fps.append(2.0)
                turn2_user_content.append({"type": "text", "text": self.TURN_VERIFY_USER})
                messages.append({"role": "user", "content": turn2_user_content})


                all_video_fps.append(fast_sample_fps)
                all_video_fps.extend([2.0] * len(verify_hires_frames))
                print(f"[CoVER-A] Starting Turn2 generation, all_video_fps len={len(all_video_fps)}")
                last_output, last_prompt_ids, last_prompt_mask, last_full_ids, last_prompt_inputs = \
                    self._run_generation(messages, video_fps=all_video_fps)
                print(f"[CoVER-A] Turn2 generation done, output length: {len(last_output)}")
                full_output_text += last_output
                last_messages = messages




                if is_deepspeed_zero3_enabled():
                    print(f"[CoVER-A] Dummy Turn3 generate (ZeRO-3 sync)")
                    dummy_msgs = [
                        {"role": "user",      "content": "Continue."},
                        {"role": "assistant", "content": last_output[:4]},
                        {"role": "user",      "content": "OK"},
                    ]
                    self._run_generation(dummy_msgs, video_fps=[])




            final_answer = self._extract_tag(last_output, "final_answer")
            if not final_answer:
                final_answer = self._extract_boxed(last_output)
            if not final_answer:
                final_answer = draft_answer


            def _unwrap_boxed(s: str) -> str:
                if not s:
                    return s
                m = re.search(r"\\boxed\{\{?(.*?)\}?\}", s, re.DOTALL)
                return m.group(1).strip() if m else s
            draft_answer = _unwrap_boxed(draft_answer)
            final_answer = _unwrap_boxed(final_answer)



            def _norm_letter(s):
                m = re.search(r'[A-Ea-e]', s or "")
                return m.group(0).upper() if m else (s or "").strip().upper()

            draft_norm = _norm_letter(draft_answer)
            final_norm = _norm_letter(final_answer)
            if draft_norm and final_norm:
                verification_state = "VERIFIED" if draft_norm == final_norm else "NOT_VERIFIED"
            else:
                verification_state = "PARTIAL"




            print(f"\n[Reward Debug] ============================")
            print(f"[Reward Debug] gt (ground_truth)     = {repr(gt)}")
            print(f"[Reward Debug] draft_answer          = {repr(draft_answer)}")
            print(f"[Reward Debug] final_answer          = {repr(final_answer)}")
            print(f"[Reward Debug] draft_norm            = {repr(_norm_letter(draft_answer))}")
            print(f"[Reward Debug] final_norm            = {repr(_norm_letter(final_answer))}")
            print(f"[Reward Debug] verification_state    = {verification_state}")
            print(f"[Reward Debug] full_output_text[:300]= {repr(full_output_text[:300])}")
            print(f"[Reward Debug] last_output[:300]     = {repr(last_output[:300])}")
            print(f"[Reward Debug] ============================\n")
            try:
                total_reward, details = compute_total_reward(
                    output_text=full_output_text,
                    initial_answer=draft_answer,
                    final_answer=final_answer,
                    verification_state=verification_state,
                    ground_truth=gt,
                    problem_type="multiple choice",
                    return_details=True,
                )
                r_final  = details["r_final"]
                r_verify = details["r_verify"]
                r_format = details["r_format"]
                print(f"[Reward Debug] r_final={r_final} r_verify={r_verify} r_format={r_format} total={total_reward}")
            except Exception as e:
                print(f"[Reward] Error computing reward: {e}")
                total_reward = r_final = r_verify = r_format = 0.0
















            full_seq = last_full_ids[0]



            all_spans = find_assistant_spans(full_seq)


            if all_spans:
                initial_prompt_length = all_spans[0][0]
            else:

                initial_prompt_length = last_prompt_ids.size(1)

            prompt_ids   = full_seq[:initial_prompt_length]
            traj_ids     = full_seq[initial_prompt_length:]
            traj_len     = traj_ids.size(0)


            completion_mask = torch.ones(traj_len, dtype=torch.long, device=device)


            loss_mask = torch.zeros(traj_len, dtype=torch.long, device=device)
            for span_start, span_end in all_spans:
                rel_s = span_start - initial_prompt_length
                rel_e = span_end   - initial_prompt_length
                if rel_e > 0:
                    loss_mask[max(0, rel_s): rel_e] = 1


            prompt_mask = torch.ones(initial_prompt_length, dtype=torch.long, device=device)

            multimodal_keywords = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]
            multimodal_inputs = {k: last_prompt_inputs.get(k, None) for k in multimodal_keywords}
            prompt_text_for_log = self.processing_class.apply_chat_template(
                last_messages, add_generation_prompt=True
            )

            reward_map = {
                "r_final":        float(r_final),
                "r_verify":       float(r_verify),
                "r_format":       float(r_format),
                "accuracy_reward": float(r_final),
                "verify_reward":  float(r_verify),
                "format_reward":  float(r_format),
                "total":          float(total_reward),
            }

            try:
                vreader.seek(0)
                del vreader
            except Exception:
                pass

            return {
                "prompt_ids":          prompt_ids,
                "prompt_mask":         prompt_mask,
                "completion_ids":      traj_ids,
                "completion_mask":     completion_mask,
                "loss_mask":           loss_mask,
                "multimodal_inputs":   multimodal_inputs,
                "full_output_text":    full_output_text,
                "completion_text":     last_output,
                "reward_map":          reward_map,
                "total_reward":        float(total_reward),
                "r_final":             float(r_final),
                "r_verify":            float(r_verify),
                "r_format":            float(r_format),
                "final_answer":        final_answer,
                "verification_state":  verification_state,
                "prompt_text_for_log": prompt_text_for_log,
            }


    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        if len(inputs) == 0:
            raise ValueError("`inputs` is empty in _generate_and_score_completions.")




        local_batch_size = len(inputs)




        print("=" * 80)
        print(f"[START] GRPO rollout | local_batch={local_batch_size} | num_generations={self.num_generations}")
        print("=" * 80)





        local_results = []
        local_prompt_texts = []
        local_completion_texts = []

        base_seed = getattr(self.args, "seed", 42)
        global_step = getattr(self.state, "global_step", 0)

        for sample_idx, sample in enumerate(inputs):
            gen_seed = (
                base_seed
                + global_step * 100000
                + self.accelerator.process_index * 10000
                + sample_idx * 1000
            )

            result = self._rollout_one_generation(sample, gen_seed)
            local_results.append(result)
            local_prompt_texts.append(result["prompt_text_for_log"])
            local_completion_texts.append(result["completion_text"])

        assert len(local_results) == local_batch_size





        def _pad_1d_tensors(tensors, pad_value: int, padding_side: str = "right"):
            assert padding_side in ["left", "right"]
            max_len = max(t.size(0) for t in tensors)
            out = []
            for t in tensors:
                pad_len = max_len - t.size(0)
                if pad_len == 0:
                    out.append(t)
                    continue
                pad_tensor = torch.full(
                    (pad_len,),
                    pad_value,
                    dtype=t.dtype,
                    device=t.device,
                )
                if padding_side == "left":
                    out.append(torch.cat([pad_tensor, t], dim=0))
                else:
                    out.append(torch.cat([t, pad_tensor], dim=0))
            return torch.stack(out, dim=0)

        def _cat_multimodal_inputs(mm_inputs_list):
            keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]
            batched = {}
            for k in keys:
                vals = [x.get(k, None) for x in mm_inputs_list if x is not None and x.get(k, None) is not None]
                if len(vals) == 0:
                    batched[k] = None
                else:
                    batched[k] = torch.cat(vals, dim=0)
            return batched

        prompt_ids = _pad_1d_tensors(
            [x["prompt_ids"] for x in local_results],
            pad_value=self.processing_class.pad_token_id,
            padding_side="left",
        )
        prompt_mask = _pad_1d_tensors(
            [x["prompt_mask"] for x in local_results],
            pad_value=0,
            padding_side="left",
        )
        completion_ids = _pad_1d_tensors(
            [x["completion_ids"] for x in local_results],
            pad_value=self.processing_class.pad_token_id,
            padding_side="right",
        )
        completion_mask = _pad_1d_tensors(
            [x["completion_mask"] for x in local_results],
            pad_value=0,
            padding_side="right",
        )
        loss_mask = _pad_1d_tensors(
            [x["loss_mask"] for x in local_results],
            pad_value=0,
            padding_side="right",
        )
        multimodal_inputs = _cat_multimodal_inputs([x["multimodal_inputs"] for x in local_results])







        reward_name_map = {
            "r_final": "r_final",
            "r_verify": "r_verify",
            "r_format": "r_format",
            "total": "total",
            "accuracy_reward": "r_final",
            "verify_reward": "r_verify",
            "format_reward": "r_format",
            "total_reward": "total_reward",
        }

        local_rewards_per_func = torch.zeros(local_batch_size, len(self.reward_funcs), device=device)

        for i, result in enumerate(local_results):
            reward_map = result["reward_map"]
            for j, reward_func_name in enumerate(self.reward_func_names):
                if reward_func_name not in reward_name_map:
                    raise KeyError(
                        f"Reward func name `{reward_func_name}` has no explicit mapping. "
                        f"Please add it to `reward_name_map`."
                    )
                mapped_key = reward_name_map[reward_func_name]
                if mapped_key not in reward_map:
                    raise KeyError(
                        f"Mapped reward key `{mapped_key}` not found in reward_map. "
                        f"Available keys: {list(reward_map.keys())}"
                    )
                local_rewards_per_func[i, j] = torch.tensor(
                    reward_map[mapped_key], dtype=torch.float32, device=device
                )


        gathered_rewards_per_func = gather(local_rewards_per_func)


        gathered_rewards = (
            gathered_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)




        global_total = gathered_rewards.size(0)
        if global_total % self.num_generations != 0:
            raise ValueError(
                f"global_total={global_total} is not divisible by num_generations={self.num_generations}. "
                f"This means the repeated-sample grouping is broken."
            )

        grouped_rewards = gathered_rewards.view(-1, self.num_generations)
        grouped_mean = grouped_rewards.mean(dim=1, keepdim=True)
        grouped_std = grouped_rewards.std(dim=1, keepdim=True)

        grouped_advantages = grouped_rewards - grouped_mean
        if self.scale_rewards:
            grouped_advantages = grouped_advantages / (grouped_std + 1e-4)

        gathered_advantages = grouped_advantages.view(-1)

        process_slice = slice(
            self.accelerator.process_index * local_batch_size,
            (self.accelerator.process_index + 1) * local_batch_size,
        )
        advantages = gathered_advantages[process_slice]




        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():


            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    input_ids,
                    attention_mask,
                    **multimodal_inputs,
                )[:, prompt_ids.size(1) - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    input_ids,
                    attention_mask,
                    **multimodal_inputs,
                )[:, prompt_ids.size(1) - 1:]
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        input_ids,
                        attention_mask,
                        **multimodal_inputs,
                    )[:, prompt_ids.size(1) - 1:]




        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        r_final_mean = float(np.mean([x["r_final"] for x in local_results])) if local_results else 0.0
        r_verify_mean = float(np.mean([x["r_verify"] for x in local_results])) if local_results else 0.0
        r_format_mean = float(np.mean([x["r_format"] for x in local_results])) if local_results else 0.0

        self._metrics[mode]["rewards/r_final"].append(r_final_mean)
        self._metrics[mode]["rewards/r_verify"].append(r_verify_mean)
        self._metrics[mode]["rewards/r_format"].append(r_format_mean)
        self._metrics[mode]["reward"].append(grouped_mean.mean().item())
        self._metrics[mode]["reward_std"].append(grouped_std.mean().item())


        agg_completion_len = self.accelerator.gather_for_metrics(loss_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_len.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_len.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_len.float().max().item())

        local_is_eos = []
        for x in local_results:
            comp_ids = x["completion_ids"]
            local_is_eos.append((comp_ids == self.processing_class.eos_token_id).any())
        local_is_eos = torch.tensor(local_is_eos, device=device, dtype=torch.bool)
        agg_eos = self.accelerator.gather_for_metrics(local_is_eos)

        term_len = agg_completion_len[agg_eos]
        clipped_ratio = 1 - len(term_len) / max(1, len(agg_completion_len))
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_ratio)

        if len(term_len) == 0:
            term_len = torch.zeros(1, device=device)

        self._metrics[mode]["completions/mean_terminated_length"].append(term_len.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_len.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_len.float().max().item())

        for j, name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{name}/mean"].append(
                torch.nanmean(gathered_rewards_per_func[:, j]).item()
            )
            self._metrics[mode][f"rewards/{name}/std"].append(
                nanstd(gathered_rewards_per_func[:, j]).item()
            )

        self._textual_logs["prompt"].extend(gather_object(local_prompt_texts))
        self._textual_logs["completion"].extend(gather_object(local_completion_texts))
        for j, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(gathered_rewards_per_func[:, j].tolist())




        rank = self.accelerator.process_index
        step = getattr(self.state, "global_step", 0)


        local_summaries = []
        for i, result in enumerate(local_results):
            sample = inputs[i]
            question = sample.get("problem", sample.get("question", sample.get("prompt", "?")))
            question = str(question)[:100]
            gt = sample.get("solution", sample.get("answer", sample.get("ground_truth", "?")))
            reward_map = result.get("reward_map", {})
            adv_val = advantages[i].item() if i < advantages.size(0) else float("nan")
            reward_val = gathered_rewards[rank * local_batch_size + i].item() \
                if (rank * local_batch_size + i) < gathered_rewards.size(0) else float("nan")
            local_summaries.append({
                "rank": rank,
                "question": question,
                "gt": str(gt)[:60],
                "reward": round(reward_val, 4),
                "r_final":  round(reward_map.get("r_final",  0.0), 4),
                "r_verify": round(reward_map.get("r_verify", 0.0), 4),
                "r_format": round(reward_map.get("r_format", 0.0), 4),
                "advantage": round(adv_val, 4),
            })

        all_summaries = gather_object(local_summaries)

        if rank == 0:
            print("\n" + "=" * 80)
            print(f"[SUMMARY] step={step} | {global_total} samples | {global_total // self.num_generations} groups x {self.num_generations} generations")
            print("=" * 80)

            all_summaries.sort(key=lambda x: x["rank"])
            for s in all_summaries:
                adv = s["advantage"]
                adv_tag = "up" if adv > 0 else ("down" if adv < 0 else "=")
                print(
                    f"  rank{s['rank']} | reward={s['reward']:+.4f}"
                    f" (final={s['r_final']:.2f} verify={s['r_verify']:.2f} format={s['r_format']:.2f})"
                    f" | adv={adv:+.4f} {adv_tag}"
                    f" | GT={s['gt']}"
                    f" | Q: {s['question'][:60]}"
                )
            print(f"\n  global rewards = {[round(r,4) for r in gathered_rewards.tolist()]}")
            print("=" * 80 + "\n")

        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "loss_mask": loss_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "multimodal_inputs": multimodal_inputs,
        }

        self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = result
        return result
    def compute_liger_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        loss_mask = inputs.get("loss_mask", completion_mask)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        last_hidden_state = self._get_last_hidden_state(model, input_ids, attention_mask, logits_to_keep)
        unwrapped_model = self.accelerator.unwrap_model(model)
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=loss_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            ref_per_token_logps=inputs["ref_per_token_logps"],
            old_per_token_logps=inputs["old_per_token_logps"],
        )
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]
        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            return self.compute_liger_loss(model, inputs)
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        loss_mask = inputs.get("loss_mask", completion_mask)
        multimodal_inputs = inputs["multimodal_inputs"]


        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, **multimodal_inputs)
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if inputs["old_per_token_logps"] is not None else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl


        if self.loss_type == "grpo":
            loss = ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * loss_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "eval" if self.control.should_evaluate else "train"


        is_clipped = (coef_1 != coef_2)
        clip_ratio = (is_clipped.float() * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio).nanmean().item()
        self._metrics[mode]["clip_ratio"].append(gathered_clip_ratio)

        if self.beta != 0.0:
            mean_kl = (per_token_kl * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            gathered_kl = self.accelerator.gather_for_metrics(mean_kl).nanmean().item()
            self._metrics[mode]["kl"].append(gathered_kl)
        else:
            gathered_kl = 0.0


        gathered_loss  = self.accelerator.gather_for_metrics(loss.detach()).nanmean().item()
        mean_ratio     = (coef_1 * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        gathered_ratio = self.accelerator.gather_for_metrics(mean_ratio).nanmean().item()


        if self.accelerator.is_main_process and mode == "train":
            step       = self.state.global_step
            mean_adv   = advantages.mean().item()
            std_adv    = advantages.std().item() if advantages.numel() > 1 else float("nan")

            reward_val     = self._metrics[mode]["reward"][-1]               if self._metrics[mode].get("reward")                else float("nan")
            reward_std_val = self._metrics[mode]["reward_std"][-1]           if self._metrics[mode].get("reward_std")            else float("nan")
            r_final_val    = self._metrics[mode]["rewards/r_final"][-1]      if self._metrics[mode].get("rewards/r_final")       else float("nan")
            r_verify_val   = self._metrics[mode]["rewards/r_verify"][-1]     if self._metrics[mode].get("rewards/r_verify")      else float("nan")
            r_format_val   = self._metrics[mode]["rewards/r_format"][-1]     if self._metrics[mode].get("rewards/r_format")      else float("nan")
            mean_len_val   = self._metrics[mode]["completions/mean_length"][-1]  if self._metrics[mode].get("completions/mean_length")  else float("nan")
            clip_len_val   = self._metrics[mode]["completions/clipped_ratio"][-1] if self._metrics[mode].get("completions/clipped_ratio") else float("nan")

            print(
                f"\n{'─'*80}\n"
                f"[Step {step:>6}] GRPO metrics\n"
                f"{'─'*80}\n"
                f"  loss          = {gathered_loss:.6f}\n"
                f"  clip_ratio    = {gathered_clip_ratio:.4f}   (ratio mean={gathered_ratio:.4f})\n"
                f"  kl            = {gathered_kl:.6f}\n"
                f"  advantage     = {mean_adv:.4f}  ± {std_adv:.4f}\n"
                f"  reward        = {reward_val:.4f}  ± {reward_std_val:.4f}\n"
                f"  r_final       = {r_final_val:.4f}\n"
                f"  r_verify      = {r_verify_val:.4f}\n"
                f"  r_format      = {r_format_val:.4f}\n"
                f"  comp_len(avg) = {mean_len_val:.1f}   clipped={clip_len_val:.3f}\n"
                f"{'─'*80}"
            )

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd
                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(self, model_name: Optional[str] = None, dataset_name: Optional[str] = None, tags: Union[str, list[str], None] = None):
        if not self.is_world_process_zero():
            return
        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None
        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]
        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")
        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )
        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))
