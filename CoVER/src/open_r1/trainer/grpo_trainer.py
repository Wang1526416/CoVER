# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
import copy
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

import datasets
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
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
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_deepspeed_available, is_liger_kernel_available, is_rich_available, is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from .vision_process import process_vision_info, get_video_hw
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from decord import VideoReader, cpu

if is_deepspeed_available():
    import deepspeed

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_wandb_available():
    import wandb



RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def add_frame_numbers_batch(frame_batch, frame_numbers):
    results = []
    for frame, frame_number in zip(frame_batch, frame_numbers):

        numbered_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(numbered_frame)
        

        width, height = numbered_frame.size
        font_size = int(min(width, height) / 8)
        try:
            font = ImageFont.truetype("../../Arial.Unicode.ttf", size=font_size)
        except OSError:
            font = ImageFont.load_default()
        

        text = str(frame_number)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0] + 10
        text_height = text_bbox[3] - text_bbox[1] + 10
        
        x = width - text_width
        y = height - text_height - text_height/3


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

        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

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


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


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
        script_args = None,
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
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            print("Applying LoRA...")
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():

                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, ['visual'])
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)


        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)


        if freeze_vision_modules:
            print("Freezing vision modules...")
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
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
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
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)


        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
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

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):

            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )


        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        self._step = 0



        self._buffered_inputs = [None] * args.gradient_accumulation_steps







        model.warnings_issued["estimate_tokens"] = True

        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            if is_peft_model(model):
                raise TypeError("Liger loss is not supported with a PEFT model.")

            if self.loss_type != "bnpo":
                raise ValueError(
                    f"The provided loss type (`{self.loss_type}`) is not supported with `use_liger_loss`. Liger loss "
                    "only supports `bnpo` for now."
                )

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


        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }


        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )





        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

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

    def _set_signature_columns_if_needed(self):




        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self, dataset) -> Sampler:
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
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
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
        try:
            logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits
        except:
            print("============================")
            print("error in _get_per_token_logps")
            print("============================")
            torch.save({"input_ids":input_ids, "attention_mask": attention_mask, "custom_multimodal_inputs": custom_multimodal_inputs}, 'debug.pth')
            assert False
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
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:












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

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"


        print(inputs[0]['path'])
        prompt = inputs[0]["prompt"]
        gt = inputs[0]['solution'][len('<answer>'):][:-len('</answer>')]
        if inputs[0]['tgt'] is not None:
            zoomin_intervals = []
        else:
            zoomin_intervals = inputs[0]['zoomin_intervals']
        try:
            vreader = VideoReader(inputs[0]["path"], ctx=cpu(0))
        except Exception as e:
            vreader = None
        fps = vreader.get_avg_fps()
        duration = len(vreader) / vreader.get_avg_fps()
        
        num_fast_frames = 768
        num_slow_frames = 16
        fast_video_resolution = 32
        slow_video_resolution = 256
        if duration < 60:
            factor = 2
        else:
            factor = 1
        answer = ''

        is_sample = 0.0
        mask_flag = 0.0
        first_format_reward = 1.0
        first_accuracy_reward = 0.0
        second_format_reward = 1.0
        second_accuracy_reward = 0.0
            
        try:
            zoomin_intervals = [[int(round(zoomin_interval[0])), int(round(zoomin_interval[1]))] for zoomin_interval in zoomin_intervals]
            zoomin_intervals = sorted(zoomin_intervals, key=lambda x: x[0])

            video_sample_fps_list = []
            content = []
            start_frame = 0
            end_frame = len(vreader) - 1
            num_frames = min(int(duration * 2), num_fast_frames)
            fast_frame_ids = torch.linspace(start_frame, end_frame, num_frames).round().long().tolist()
            sample_fps = round(num_frames / (end_frame - start_frame) * fps, 2)
            video_sample_fps_list.append(sample_fps)
            fast_video = vreader.get_batch(fast_frame_ids).asnumpy()
            fast_frame_seconds = [int(frame_id / fps) for frame_id in fast_frame_ids]
            fast_video = add_frame_numbers_batch(fast_video, fast_frame_seconds)

            H, W = fast_video.shape[-2:]
            resized_height, resized_width = get_video_hw(H, W, fast_video_resolution * 28 * 28 * factor)
            fast_video = transforms.functional.resize(
                fast_video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            content.append({"type": "text", "text": "Full video [%d, %d]:" % (0, int(duration))})
            content.append({"type": "video", "video": fast_video})

            for interval in zoomin_intervals:
                start_frame = max(0, int(interval[0] * fps))
                end_frame = min(len(vreader) - 1, int(interval[1] * fps))
                num_frames = min(int((interval[1] - interval[0]) * 2), num_slow_frames)
                slow_frame_ids = torch.linspace(start_frame, end_frame, num_frames).round().long().tolist()
                sample_fps = round(num_frames / (end_frame - start_frame) * fps, 2)
                slow_video = vreader.get_batch(slow_frame_ids).asnumpy()
                slow_frame_seconds = [int(frame_id / fps) for frame_id in slow_frame_ids]
                slow_video = add_frame_numbers_batch(slow_video, slow_frame_seconds)

                H, W = slow_video.shape[-2:]
                resized_height, resized_width = get_video_hw(H, W, slow_video_resolution * 28 * 28)
                slow_video = transforms.functional.resize(
                    slow_video,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                content.append({"type": "text", "text": "Subset zoom-in video clip [%d, %d]:" % (int(interval[0]), int(interval[1]))})
                content.append({"type": "video", "video": slow_video})
                video_sample_fps_list.append(sample_fps)

            content.append({"type": "text", "text": prompt})
            messages =[
                {"role": "system", "content": "You are a helpful assistant. The red numbers on each frame represent the timestamp in seconds and you can refer them during temporal grounding."},
                {
                        "role": "user",
                        "content": content
                    }]
            video_kwargs = {'fps': video_sample_fps_list}
        except Exception as e:
            print(e)
            mask_flag = 1.0
            content = []
            content.append({"type": "text", "text": "Full video [%d, %d]:" % (0, int(duration))})
            content.append({"type": "video", "video": torch.zeros((1, 3, 128, 128), dtype=torch.float)})
            content.append({"type": "text", "text": prompt})
            messages =[
                {"role": "system", "content": "You are a helpful assistant. The red numbers on each frame represent the timestamp in seconds and you can refer them during temporal grounding."},
                {
                        "role": "user",
                        "content": content
                    }]
            video_sample_fps_list = [2.0]
            video_kwargs = {'fps': video_sample_fps_list}
            zoomin_intervals = []


        images, videos = process_vision_info([messages])

        if inputs[0]['tgt'] is not None:
            prefix = "I need to zoom in on the video.\n\n"
        else:
            prefix = ''

        prompts_text = [self.processing_class.apply_chat_template(messages, add_generation_prompt=True) + prefix]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **video_kwargs
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, generation_config=self.generation_config, use_cache=True,
            )



        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]


        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        info = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)[0]
        
        if inputs[0]['tgt'] is None:
            if info.startswith("I need to zoom in on the video.\n\n"):
                is_sample = 1.0
                try:
                    answer = info.split("\\boxed{")[1].split("}")[0]
                    if not answer.startswith('[') or not answer.endswith(']'):
                        assert False

                    zoomin_interval = answer[1:-1]
                    zoomin_interval = zoomin_interval.split(',')
                    zoomin_interval = [int(frame.strip()) for frame in zoomin_interval]
                    zoomin_interval = sorted(zoomin_interval)

                    max_time = min(duration / 2, 60)
                    if zoomin_interval[0] < 0 or zoomin_interval[1] < 0 or zoomin_interval[0] > (duration + 1) or zoomin_interval[1] > (duration + 1) or (zoomin_interval[1] - zoomin_interval[0] > max_time) or (zoomin_interval[1] - zoomin_interval[0] < 1):
                        assert False, "zoomin format error %s" % (answer)
                    
                    iou_with_exist_interval = [iou_1d(zoomin_interval, tgt) for tgt in zoomin_intervals]
                    if sum([iou > 0.7 for iou in iou_with_exist_interval]) > 0:
                        assert False, "overlapping existing zoomin intervals"

                    answer_zoomin_intervals = zoomin_intervals + [zoomin_interval]
                except Exception as e:
                    print(e)
                    first_format_reward = 0.0
                    answer_zoomin_intervals = []
            else:
                answer_zoomin_intervals = zoomin_intervals
            if info.startswith("I get the answer.\n\n"):
                try:
                    temp_answer = info.split("\\boxed{")[1][0]
                    first_accuracy_reward = 1.0 if temp_answer.lower() == gt.lower() else 0.0
                except Exception as e:
                    first_accuracy_reward = 0.0
                    first_format_reward = 0.0
        else:
            answer_zoomin_intervals = inputs[0]['zoomin_intervals']
            try:
                answer = info.split("\\boxed{")[1].split("}")[0]
                if not answer.startswith('[') or not answer.endswith(']'):
                    assert False

                zoomin_interval = answer[1:-1]
                zoomin_interval = zoomin_interval.split(',')
                zoomin_interval = [int(frame.strip()) for frame in zoomin_interval]
                zoomin_interval = sorted(zoomin_interval)

                max_time = min(duration / 2, 60)
                if zoomin_interval[0] < 0 or zoomin_interval[1] < 0 or zoomin_interval[0] > (duration + 1) or zoomin_interval[1] > (duration + 1) or (zoomin_interval[1] - zoomin_interval[0] > max_time) or (zoomin_interval[1] - zoomin_interval[0] < 1):
                    assert False, "zoomin format error %s" % (answer)
                
                iou_with_exist_interval = [iou_1d(zoomin_interval, tgt) for tgt in zoomin_intervals]
                if sum([iou > 0.7 for iou in iou_with_exist_interval]) > 0:
                    assert False, "overlapping existing zoomin intervals"

                first_accuracy_reward = 1.0 if sum([iou_1d(zoomin_interval, tgt) > 0 for tgt in inputs[0]['tgt']]) > 0 else 0.0
                print(zoomin_interval, first_accuracy_reward)
            except Exception as e:
                print(e)
                first_format_reward = 0.0


        answer_content = messages[1]['content'][:2]
        video_sample_fps_list = video_sample_fps_list[:1]
        try:
            answer_zoomin_intervals = [[int(round(zoomin_interval[0])), int(round(zoomin_interval[1]))] for zoomin_interval in answer_zoomin_intervals]
            answer_zoomin_intervals = sorted(answer_zoomin_intervals, key=lambda x: x[0])
            for interval in answer_zoomin_intervals:
                start_frame = max(0, int(interval[0] * fps))
                end_frame = min(len(vreader) - 1, int(interval[1] * fps))
                num_frames = min(int((interval[1] - interval[0]) * 2), num_slow_frames)
                slow_frame_ids = torch.linspace(start_frame, end_frame, num_frames).round().long().tolist()
                sample_fps = round(num_frames / (end_frame - start_frame) * fps, 2)
                
                slow_video = vreader.get_batch(slow_frame_ids).asnumpy()
                slow_frame_seconds = [int(frame_id / fps) for frame_id in slow_frame_ids]
                slow_video = add_frame_numbers_batch(slow_video, slow_frame_seconds)

                H, W = slow_video.shape[-2:]
                resized_height, resized_width = get_video_hw(H, W, slow_video_resolution * 28 * 28)
                slow_video = transforms.functional.resize(
                    slow_video,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                answer_content.append({"type": "text", "text": "Subset zoom-in video clip [%d, %d]:" % (int(interval[0]), int(interval[1]))})
                answer_content.append({"type": "video", "video": slow_video})
                video_sample_fps_list.append(sample_fps)
        except Exception as e:
            print(e)
        
        answer_content.append({"type": "text", "text": prompt})
        answer_messages =[
            {"role": "system", "content": "You are a helpful assistant. The red numbers on each frame represent the timestamp in seconds and you can refer them during temporal grounding."},
            {
                    "role": "user",
                    "content": answer_content
                }]
        video_kwargs = {'fps': video_sample_fps_list}
        
        try:
            images, videos = process_vision_info([answer_messages])
            answer_prompts_text = [self.processing_class.apply_chat_template(answer_messages, add_generation_prompt=True) + prefix]
            answer_prompt_inputs = self.processing_class(
                text=answer_prompts_text,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **video_kwargs
            )
            answer_prompt_inputs = super()._prepare_inputs(answer_prompt_inputs)
            answer_prompt_ids, answer_prompt_mask = answer_prompt_inputs["input_ids"], answer_prompt_inputs["attention_mask"]

            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                answer_prompt_completion_ids = unwrapped_model.generate(
                    **answer_prompt_inputs, generation_config=self.generation_config, use_cache=True,
                )
            

            answer_prompt_length = answer_prompt_ids.size(1)
            answer_prompt_ids = answer_prompt_completion_ids[:, :answer_prompt_length]
            answer_completion_ids = answer_prompt_completion_ids[:, answer_prompt_length:]


            is_eos = answer_completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            answer_completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            answer_info = self.processing_class.batch_decode(answer_completion_ids, skip_special_tokens=True)[0]
            print(answer_info)
            if inputs[0]['tgt'] is None:
                if answer_info.startswith("I need to zoom in on the video.\n\n"):
                    mask_flag = 1.0
                    second_accuracy_reward = 0.0
                else:
                    if "\\boxed{" in answer_info:
                        temp_answer = answer_info.split("\\boxed{")[1][0]
                        second_accuracy_reward = 1.0 if temp_answer.lower() == gt.lower() else 0.0
                    else:
                        second_accuracy_reward = 0.0
            else:
                try:
                    answer = answer_info.split("\\boxed{")[1].split("}")[0]
                    if not answer.startswith('[') or not answer.endswith(']'):
                        assert False

                    zoomin_interval = answer[1:-1]
                    zoomin_interval = zoomin_interval.split(',')
                    zoomin_interval = [int(frame.strip()) for frame in zoomin_interval]
                    zoomin_interval = sorted(zoomin_interval)

                    max_time = min(duration / 2, 60)
                    if zoomin_interval[0] < 0 or zoomin_interval[1] < 0 or zoomin_interval[0] > (duration + 1) or zoomin_interval[1] > (duration + 1) or (zoomin_interval[1] - zoomin_interval[0] > max_time) or (zoomin_interval[1] - zoomin_interval[0] < 1):
                        assert False, "zoomin format error %s" % (answer)
                    
                    iou_with_exist_interval = [iou_1d(zoomin_interval, tgt) for tgt in answer_zoomin_intervals]
                    if sum([iou > 0.7 for iou in iou_with_exist_interval]) > 0:
                        assert False, "overlapping existing zoomin intervals"

                    second_accuracy_reward = 1.0 if sum([iou_1d(zoomin_interval, tgt) > 0 for tgt in inputs[0]['tgt']]) > 0 else 0.0
                    print(zoomin_interval, second_accuracy_reward)
                except Exception as e:
                    print(e)
                    second_format_reward = 0.0
            
        except Exception as e:
            print(e)
            second_accuracy_reward = 0.0

        if inputs[0]['tgt'] is None:
            if info.startswith("I need to zoom in on the video.\n\n"):
                first_accuracy_reward = second_accuracy_reward
        

        if mask_flag:
            first_accuracy_reward = 0.0
            second_accuracy_reward = 0.0
        
        if first_format_reward == 0:
            first_accuracy_reward = 0.0
        
        if second_format_reward == 0:
            second_accuracy_reward = 0.0



        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        answer_attention_mask = torch.cat([answer_prompt_mask, answer_completion_mask], dim=1)

        if mask_flag:
            completion_mask[:] = 0
            answer_completion_mask[:] = 0


        multimodal_keywords = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        answer_multimodal_inputs = {k: answer_prompt_inputs[k] if k in answer_prompt_inputs else None for k in multimodal_keywords}
        with torch.no_grad():


            old_per_token_logps = None
            answer_old_per_token_logps = self._get_per_token_logps(
                self.model, answer_prompt_completion_ids, answer_attention_mask, **answer_multimodal_inputs
            )
            answer_old_per_token_logps = answer_old_per_token_logps[:, answer_prompt_length - 1:]

            if self.beta == 0.0:
                ref_per_token_logps = None
                answer_ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                answer_ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, answer_prompt_completion_ids, answer_attention_mask, **answer_multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    answer_ref_per_token_logps = self._get_per_token_logps(
                        self.model, answer_prompt_completion_ids, answer_attention_mask, **answer_multimodal_inputs
                    )
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            answer_ref_per_token_logps = answer_ref_per_token_logps[:, answer_prompt_length - 1:]

        first_rewards_per_func = torch.zeros(len(inputs), len(self.reward_funcs), device=device)
        second_rewards_per_func = torch.zeros(len(inputs), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):

            if reward_func_name == 'accuracy_reward':

                output_reward_func = [first_accuracy_reward]
            else:
                output_reward_func = [first_format_reward]

            first_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            if reward_func_name == 'accuracy_reward':

                output_reward_func = [second_accuracy_reward]
            else:
                output_reward_func = [second_format_reward]

            second_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)



        first_rewards_per_func = gather(first_rewards_per_func)
        second_rewards_per_func = gather(second_rewards_per_func)
        is_mask_reward = self.accelerator.gather_for_metrics(torch.tensor(mask_flag, device=device)).bool().view(-1, self.num_generations)


        first_rewards = (first_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        first_rewards = first_rewards.view(-1, self.num_generations)
        second_rewards = (second_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        second_rewards = second_rewards.view(-1, self.num_generations)


        first_mean_grouped_rewards = first_rewards.mean(dim=1)
        first_std_grouped_rewards = first_rewards.std(dim=1)
        second_mean_grouped_rewards = second_rewards.mean(dim=1)
        second_std_grouped_rewards = second_rewards.std(dim=1)
        

        index = self.accelerator.process_index // self.num_generations
        is_mask_reward = is_mask_reward[index]
        if torch.sum(~is_mask_reward) < self.num_generations and torch.sum(~is_mask_reward) > 1:
            first_mask_mean = torch.mean(first_rewards[index][~is_mask_reward])
            first_mask_std = torch.std(first_rewards[index][~is_mask_reward])
            first_mean_grouped_rewards[index] = first_mask_mean
            first_std_grouped_rewards[index] = first_mask_std

            second_mask_mean = torch.mean(second_rewards[index][~is_mask_reward])
            second_mask_std = torch.std(second_rewards[index][~is_mask_reward])
            second_mean_grouped_rewards[index] = second_mask_mean
            second_std_grouped_rewards[index] = second_mask_std

        elif torch.sum(~is_mask_reward) == 1:
            first_mean_grouped_rewards[index] = first_rewards[index][~is_mask_reward]
            first_std_grouped_rewards[index] = 1

            second_mean_grouped_rewards[index] = second_rewards[index][~is_mask_reward]
            second_std_grouped_rewards[index] = 1


        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )


        first_mean_grouped_rewards = first_mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        first_std_grouped_rewards = first_std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        first_rewards = first_rewards.flatten()
        first_advantages = first_rewards - first_mean_grouped_rewards
        if self.scale_rewards:
            first_advantages = first_advantages / (first_std_grouped_rewards + 1e-4)
        first_advantages = first_advantages[process_slice]

        second_mean_grouped_rewards = second_mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        second_std_grouped_rewards = second_std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        second_rewards = second_rewards.flatten()
        second_advantages = second_rewards - second_mean_grouped_rewards
        if self.scale_rewards:
            second_advantages = second_advantages / (second_std_grouped_rewards + 1e-4)
        second_advantages = second_advantages[process_slice]


        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        is_sample = self.accelerator.gather_for_metrics(torch.tensor([is_sample,], dtype=torch.float, device=device))
        self._metrics[mode]["is_sample"].append(is_sample.float().mean().item())

        agg_completion_mask = self.accelerator.gather_for_metrics(attention_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())


        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:

            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())


        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(first_rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(first_rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(first_mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(first_std_grouped_rewards.mean().item())


        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(first_rewards_per_func[:, i].tolist())
        is_mask = self.accelerator.gather_for_metrics(torch.tensor(mask_flag, device=device))
        self._metrics[mode]["is_mask"].append(is_mask.float().mean().item())
        if vreader is not None:
            vreader.seek(0)
            del vreader
        print(info)
        if inputs[0]['tgt'] is None:
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = {
                    "prompt_ids": answer_prompt_ids,
                    "prompt_mask": answer_prompt_mask,
                    "completion_ids": answer_completion_ids,
                    "completion_mask": answer_completion_mask,
                    "advantages": first_advantages,
                    "old_per_token_logps": answer_old_per_token_logps,
                    "ref_per_token_logps": answer_ref_per_token_logps,
                    "multimodal_inputs": answer_multimodal_inputs,
                }
        else:
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = {
                    "prompt_ids": answer_prompt_ids,
                    "prompt_mask": answer_prompt_mask,
                    "completion_ids": answer_completion_ids,
                    "completion_mask": answer_completion_mask,
                    "advantages": second_advantages,
                    "old_per_token_logps": answer_old_per_token_logps,
                    "ref_per_token_logps": answer_ref_per_token_logps,
                    "multimodal_inputs": answer_multimodal_inputs,
                }
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": first_advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "multimodal_inputs": multimodal_inputs,
        }

    def compute_liger_loss(self, model, inputs):

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)


        last_hidden_state = self._get_last_hidden_state(model, input_ids, attention_mask, logits_to_keep)
        unwrapped_model = self.accelerator.unwrap_model(model)

        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
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
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
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
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())



















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

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
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
