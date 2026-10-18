# Copyright 2024. All rights reserved.
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

from qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()

import os
import re
import json
import random
import uuid
import time
import multiprocessing as mp
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import safe_globals, ParallelMode, set_rng_state_for_device
import numpy as np
import transformers
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)
from trainer import process_vision_info, get_video_hw
from typing import Optional, List, Dict, Any, Tuple
from torch.utils.data import Dataset
import wandb
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
import copy
from dataclasses import dataclass, field
from clip_worker import worker_main as clip_worker_main






def _new_load_rng_state(self, resume_from_checkpoint):
    if checkpoint is None:
        return

    if self.args.world_size > 1:
        process_index = self.args.process_index
        rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
        if not os.path.isfile(rng_file):
            return
    else:
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            return

    with safe_globals():
        checkpoint_rng_state = torch.load(rng_file)
    random.setstate(checkpoint_rng_state["python"])
    np.random.set_state(checkpoint_rng_state["numpy"])
    torch.random.set_rng_state(checkpoint_rng_state["cpu"])

    is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
    if torch.cuda.is_available():
        set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)


transformers.Trainer._load_rng_state = _new_load_rng_state






@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

@dataclass
class SFTScriptArguments(ScriptArguments):
    data_folder: str = field(
        default="./",
        metadata={"help": "root folder of video files"},
    )
    video_sft: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using video_sft mode"},
    )

    clip_model_name: str = field(
        default="openai/clip-vit-large-patch14",
        metadata={"help": "CLIP model path or HuggingFace name for retrieval"},
    )
    num_clips: int = field(
        default=3,
        metadata={"help": "number of zoom-in clips to retrieve via pseudo_queries"},
    )
    clip_duration: float = field(
        default=30.0,
        metadata={"help": "duration (seconds) of each zoom-in clip"},
    )
    clip_sim_threshold: float = field(
        default=0.20,
        metadata={"help": "minimum CLIP similarity score to keep a clip"},
    )
    verification_num_clips: int = field(
        default=3,
        metadata={"help": "number of verification clips to retrieve via clue"},
    )
    verification_clip_duration: float = field(
        default=10.0,
        metadata={"help": "duration (seconds) of each verification clip"},
    )






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
        text_width  = text_bbox[2] - text_bbox[0] + 10
        text_height = text_bbox[3] - text_bbox[1] + 10

        x = width  - text_width
        y = height - text_height - text_height / 3

        draw.text((x, y), text, font=font, fill='red')
        results.append(torch.tensor(np.array(numbered_frame)))

    results = torch.stack(results).permute(0, 3, 1, 2)
    return results




def extract_hires_video(
    vreader: VideoReader,
    fps: float,
    duration: float,
    clips: List[Dict],
    slow_max_frames: int = 32,
    slow_video_resolution: int = 256,
    factor: int = 1,
) -> Tuple[List[torch.Tensor], List[float]]:
    sorted_clips = sorted(clips, key=lambda x: x["start"])
    if not sorted_clips:
        return [], []

    total_frames = len(vreader)
    durations = [c["end"] - c["start"] for c in sorted_clips]
    total_dur = sum(durations)


    frames_per_clip = [max(2, int(d / total_dur * slow_max_frames)) if total_dur > 0 else 2
                       for d in durations]
    cur_total = sum(frames_per_clip)
    if cur_total > slow_max_frames:
        scale = slow_max_frames / cur_total
        frames_per_clip = [max(2, int(f * scale)) for f in frames_per_clip]

    result_tensors, result_fps = [], []
    for clip, num_f in zip(sorted_clips, frames_per_clip):
        start_f = max(0, int(clip["start"] * fps))
        end_f   = min(total_frames - 1, int(clip["end"] * fps))
        frame_ids = torch.linspace(start_f, end_f, num_f).round().long().tolist()
        clip_fps  = round(num_f / max(1, end_f - start_f) * fps, 2)

        raw   = vreader.get_batch(frame_ids).asnumpy()

        if raw.ndim == 3:
            raw = np.stack([raw] * 3, axis=-1)
        raw = raw.astype(np.uint8)
        secs  = [int(fid / fps) for fid in frame_ids]
        tensor = add_frame_numbers_batch(raw, secs)

        H, W = tensor.shape[-2:]
        rh, rw = get_video_hw(H, W, slow_video_resolution * 28 * 28 * factor)
        tensor = transforms.functional.resize(
            tensor.float(), [rh, rw],
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
        result_tensors.append(tensor)
        result_fps.append(clip_fps)

    return result_tensors, result_fps















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






def parse_final_cot(final_cot: str) -> dict:
    think_list     = [t.strip() for t in re.findall(r'<think>(.*?)</think>', final_cot, re.DOTALL)]
    pq_m           = re.search(r'<pseudo_queries>(.*?)</pseudo_queries>', final_cot, re.DOTALL)
    pseudo_queries = pq_m.group(1).strip() if pq_m else ""
    da_m           = re.search(r'<draft_answer>(.*?)</draft_answer>', final_cot, re.DOTALL)
    draft_answer   = da_m.group(1).strip() if da_m else ""
    clue_m         = re.search(r'<clue>(.*?)</clue>', final_cot, re.DOTALL)
    clue           = clue_m.group(1).strip() if clue_m else ""
    fa_m           = re.search(r'<final_answer>(.*?)</final_answer>', final_cot, re.DOTALL)
    final_answer   = fa_m.group(1).strip() if fa_m else ""

    return {
        "num_turns":      3 if pseudo_queries else 2,
        "think_list":     think_list,
        "pseudo_queries": pseudo_queries,
        "draft_answer":   draft_answer,
        "clue":           clue,
        "final_answer":   final_answer,
    }






class LazySupervisedDataset(Dataset):

    def __init__(self, data_path: str, script_args: SFTScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args


        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                self.list_data_dict = [json.loads(line) for line in f if line.strip()]
        else:
            with open(data_path, "r") as f:
                self.list_data_dict = json.load(f)


        self._clip_request_queue  = None
        self._clip_response_queue = None
        self._clip_worker_process = None
        self._start_clip_worker()





    def _start_clip_worker(self):
        import queue as _queue
        if self._clip_worker_process is not None:
            return
        _rank = os.environ.get("RANK", "?")
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        ctx = mp.get_context("spawn")
        self._clip_request_queue  = ctx.Queue()
        self._clip_response_queue = ctx.Queue()
        self._clip_worker_process = ctx.Process(
            target=clip_worker_main,
            args=(self._clip_request_queue, self._clip_response_queue,
                  self.script_args.clip_model_name, _local_rank),
            daemon=False,
        )
        self._clip_worker_process.start()
        print(f"[Dataset rank={_rank}] CLIP worker spawned pid={self._clip_worker_process.pid}, device=cuda:{_local_rank}, waiting for ready...", flush=True)


        try:
            msg = self._clip_response_queue.get(timeout=300)
            assert isinstance(msg, dict) and msg.get("ready"), f"unexpected msg: {msg}"
            print(f"[Dataset rank={_rank}] *** CLIP worker READY *** pid={msg['pid']}", flush=True)
        except _queue.Empty:
            raise RuntimeError(
                f"[Dataset rank={_rank}] CLIP worker did not become ready within 300s. "
                f"alive={self._clip_worker_process.is_alive()}"
            )

    def _stop_clip_worker(self):

        if hasattr(self, "_clip_request_queue") and self._clip_request_queue is not None:
            self._clip_request_queue.put(None)
        if hasattr(self, "_clip_worker_process") and self._clip_worker_process is not None:
            self._clip_worker_process.join(timeout=5)
            if self._clip_worker_process.is_alive():
                self._clip_worker_process.terminate()

    def _retrieve_clips_via_worker(
        self,
        video_path: str,
        queries: List[str],
        n_clips: int,
        clip_duration: float,
        mode: str = "zoom",
        question: str = "",
        sim_threshold: float = 0.15,
        num_fast_frames: int = 768,
    ) -> List[Dict]:

        if self._clip_worker_process is None or not self._clip_worker_process.is_alive():
            self._clip_worker_process = None
            self._clip_request_queue  = None
            self._clip_response_queue = None
            self._start_clip_worker()

        req_id = str(uuid.uuid4())
        self._clip_request_queue.put({
            "req_id":          req_id,
            "video_path":      video_path,
            "queries":         queries,
            "n_clips":         n_clips,
            "clip_duration":   clip_duration,
            "mode":            mode,
            "question":        question,
            "sim_threshold":   sim_threshold,
            "num_fast_frames": num_fast_frames,
        })

        while True:
            msg = self._clip_response_queue.get(timeout=600)
            if msg.get("ready"):
                continue
            if msg["req_id"] != req_id:
                continue
            if not msg["ok"]:
                raise RuntimeError(msg["error"])
            return msg["result"]

    def __del__(self):
        try:
            self._stop_clip_worker()
        except Exception:
            pass

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        num_base_retries = 3
        video_path = os.path.join(self.script_args.data_folder, self.list_data_dict[i]["video"])
        print(f'[__getitem__] index={i}, video_path={video_path}')
        try:
            return self._get_item(i)
        except Exception as e:
            import traceback
            print(f'[__getitem__] FAILED index={i}, video_path={video_path}')
            print(f'[__getitem__] Exception: {e}')
            traceback.print_exc()

        for attempt_idx in range(num_base_retries):
            sample_idx = random.choice(range(len(self)))
            retry_path = os.path.join(self.script_args.data_folder, self.list_data_dict[sample_idx]["video"])
            print(f'[try other #{attempt_idx}] index={sample_idx}, video_path={retry_path}')
            try:
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                import traceback
                print(f'[try other #{attempt_idx}] FAILED index={sample_idx}, video_path={retry_path}')
                print(f'[try other #{attempt_idx}] Exception: {e}')
                traceback.print_exc()


        raise RuntimeError(f"Failed to fetch any valid sample after {num_base_retries} retries starting from index {i}")

    def _get_item(self, i):
        _rank = os.environ.get("RANK", "?")
        _t0   = time.time()
        source = copy.deepcopy(self.list_data_dict[i])
        print(f"[_get_item rank={_rank}] START i={i} video={source['video']}", flush=True)




        vreader          = VideoReader(os.path.join(self.script_args.data_folder, source["video"]), ctx=cpu(0))
        duration         = len(vreader) / vreader.get_avg_fps()
        fps              = vreader.get_avg_fps()
        total_num_frames = len(vreader)
        print(f"[_get_item rank={_rank}] i={i} video loaded: dur={duration:.1f}s fps={fps:.2f} frames={total_num_frames} elapsed={time.time()-_t0:.1f}s", flush=True)


        num_fast_frames       = 512
        fast_video_resolution = 32
        slow_max_frames       = 32
        slow_video_resolution = 256
        factor = 1




        start_frame = 0
        end_frame   = len(vreader) - 1
        num_fast    = min(int(duration * 2), num_fast_frames)
        fast_frame_ids  = torch.linspace(start_frame, end_frame, num_fast).round().long().tolist()
        fast_sample_fps = round(num_fast / (end_frame - start_frame) * fps, 2)

        fast_video_raw     = vreader.get_batch(fast_frame_ids).asnumpy()

        if fast_video_raw.ndim == 3:
            fast_video_raw = np.stack([fast_video_raw] * 3, axis=-1)
        fast_video_raw = fast_video_raw.astype(np.uint8)
        fast_frame_seconds = [int(fid / fps) for fid in fast_frame_ids]
        fast_video         = add_frame_numbers_batch(fast_video_raw, fast_frame_seconds)

        H, W = fast_video.shape[-2:]
        rh, rw = get_video_hw(H, W, fast_video_resolution * 28 * 28 * factor)
        fast_video = transforms.functional.resize(
            fast_video, [rh, rw],
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        ).float()
        print(f"[_get_item rank={_rank}] i={i} fast frames sampled: num={num_fast} shape={fast_video.shape} elapsed={time.time()-_t0:.1f}s", flush=True)





        question = re.sub(r'<image>\s*', '', source['original_question']).strip()
        parsed   = parse_final_cot(source['final_cot'])
        num_turns   = parsed['num_turns']
        think_list  = parsed['think_list']




        draft_letter = ""
        da_m = re.search(r'\\boxed\{(\w)\}', parsed['draft_answer'])
        if da_m:
            draft_letter = da_m.group(1).upper()
        gt_letter = ""
        gt_m = re.search(r'^([A-E])', source.get('original_answer', '').strip())
        if gt_m:
            gt_letter = gt_m.group(1).upper()
        draft_correct = (draft_letter == gt_letter and draft_letter != "")



        mask_draft_turn = -1
        if not draft_correct:
            if num_turns == 2:
                mask_draft_turn = 0
            else:
                mask_draft_turn = 1

        print(f"[_get_item rank={_rank}] i={i} parsed final_cot: num_turns={num_turns} len(think_list)={len(think_list)} has_clue={bool(parsed['clue'])} has_pq={bool(parsed['pseudo_queries'])} draft_correct={draft_correct} elapsed={time.time()-_t0:.1f}s", flush=True)




        messages              = []
        video_sample_fps_list = []

        messages.append({"role": "system", "content": SYSTEM_PROMPT})




        turn1_user_text = TURN1_USER_SUFFIX.format(
            duration=duration, fps=fps,
            total_frames=total_num_frames, question=question,
        )
        turn1_user_content = [
            {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
            {"type": "video", "video": fast_video},
            {"type": "text",  "text":  turn1_user_text},
        ]
        video_sample_fps_list.append(fast_sample_fps)
        messages.append({"role": "user", "content": turn1_user_content})
        if num_turns == 2:





            messages.append({"role": "assistant", "content": (
                f"<think>\n{think_list[0]}\n</think>\n\n"
                f"<draft_answer>\n{parsed['draft_answer']}\n</draft_answer>\n\n"
                f"<clue>\n{parsed['clue']}\n</clue>"
            )})





            video_path = os.path.join(self.script_args.data_folder, source["video"])
            print(f"[_get_item rank={_rank}] i={i} PatternA: calling verify CLIP elapsed={time.time()-_t0:.1f}s", flush=True)
            _result = self._retrieve_clips_via_worker(
                video_path=video_path,
                queries=[parsed['clue']],
                n_clips=self.script_args.verification_num_clips,
                clip_duration=self.script_args.verification_clip_duration,
                mode="verify",
                sim_threshold=self.script_args.clip_sim_threshold,
            )
            verify_clips = [{"start": c["start"], "end": c["end"], "similarity": c["similarity"]}
                            for c in _result["clips"] if c["similarity"] >= self.script_args.clip_sim_threshold]
            print(f"[_get_item rank={_rank}] i={i} PatternA: verify CLIP done, clips={len(verify_clips)} elapsed={time.time()-_t0:.1f}s", flush=True)
            verify_clips = sorted(verify_clips, key=lambda x: x["start"])
            verify_tensors, verify_fps_list = extract_hires_video(
                vreader, fps, duration, verify_clips,
                slow_max_frames=slow_max_frames,
                slow_video_resolution=slow_video_resolution,
                factor=factor,
            )
            turn2_user_content = [
                {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
                {"type": "video", "video": fast_video},
                {"type": "text", "text": "Verification clips (retrieved via clue similarity):"},
            ]
            for clip_info, clip_tensor in zip(verify_clips, verify_tensors):
                turn2_user_content.append({
                    "type": "text",
                    "text": f"[{clip_info['start']:.1f}s – {clip_info['end']:.1f}s]",
                })

                turn2_user_content.append({"type": "video", "video": clip_tensor})
            turn2_user_content.append({"type": "text", "text": TURN_VERIFY_USER})
            video_sample_fps_list.append(fast_sample_fps)
            video_sample_fps_list.extend(verify_fps_list)
            messages.append({"role": "user", "content": turn2_user_content})


            messages.append({"role": "assistant", "content": (
                f"<think>\n{think_list[1]}\n</think>\n\n"
                f"<final_answer>\n{parsed['final_answer']}\n</final_answer>"
            )})

        else:





            messages.append({"role": "assistant", "content": (
                f"<think>\n{think_list[0]}\n</think>\n\n"
                f"<pseudo_queries>\n{parsed['pseudo_queries']}\n</pseudo_queries>"
            )})






            pq_lines = [re.sub(r'^\d+\.\s*', '', l).strip()
                        for l in parsed['pseudo_queries'].splitlines() if l.strip()]

            video_path = os.path.join(self.script_args.data_folder, source["video"])
            print(f"[_get_item rank={_rank}] i={i} PatternB: calling zoom CLIP pq_lines={pq_lines} elapsed={time.time()-_t0:.1f}s", flush=True)
            _result = self._retrieve_clips_via_worker(
                video_path=video_path,
                queries=pq_lines,
                n_clips=self.script_args.num_clips,
                clip_duration=self.script_args.clip_duration,
                mode="zoom",
                question=question,
                sim_threshold=self.script_args.clip_sim_threshold,
            )
            zoom_clips = [{"start": c["start"], "end": c["end"], "similarity": c["similarity"]}
                          for c in _result["clips"] if c["similarity"] >= self.script_args.clip_sim_threshold]
            print(f"[_get_item rank={_rank}] i={i} PatternB: zoom CLIP done, clips={len(zoom_clips)} elapsed={time.time()-_t0:.1f}s", flush=True)
            zoom_clips = sorted(zoom_clips, key=lambda x: x["start"])
            zoom_tensors, zoom_fps_list = extract_hires_video(
                vreader, fps, duration, zoom_clips,
                slow_max_frames=slow_max_frames,
                slow_video_resolution=slow_video_resolution,
                factor=factor,
            )
            turn2_user_text = TURN2_B_USER_SUFFIX.format(question=question)
            turn2_user_content = [
                {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
                {"type": "video", "video": fast_video},
                {"type": "text", "text": "Retrieved zoom-in clips (selected via pseudo-query similarity):"},
            ]

            for clip_info, clip_tensor in zip(zoom_clips, zoom_tensors):
                turn2_user_content.append({
                    "type": "text",
                    "text": f"[{clip_info['start']:.1f}s – {clip_info['end']:.1f}s]",
                })
                turn2_user_content.append({"type": "video", "video": clip_tensor})
            turn2_user_content.append({"type": "text", "text": turn2_user_text})
            video_sample_fps_list.append(fast_sample_fps)
            video_sample_fps_list.extend(zoom_fps_list)
            messages.append({"role": "user", "content": turn2_user_content})


            messages.append({"role": "assistant", "content": (
                f"<think>\n{think_list[1]}\n</think>\n\n"
                f"<draft_answer>\n{parsed['draft_answer']}\n</draft_answer>\n\n"
                f"<clue>\n{parsed['clue']}\n</clue>"
            )})




            print(f"[_get_item rank={_rank}] i={i} PatternB: calling verify CLIP elapsed={time.time()-_t0:.1f}s", flush=True)
            _result = self._retrieve_clips_via_worker(
                video_path=video_path,
                queries=[parsed['clue']],
                n_clips=self.script_args.verification_num_clips,
                clip_duration=self.script_args.verification_clip_duration,
                mode="verify",
                sim_threshold=self.script_args.clip_sim_threshold,
            )
            verify_clips = [{"start": c["start"], "end": c["end"], "similarity": c["similarity"]}
                            for c in _result["clips"] if c["similarity"] >= self.script_args.clip_sim_threshold]
            print(f"[_get_item rank={_rank}] i={i} PatternB: verify CLIP done, clips={len(verify_clips)} elapsed={time.time()-_t0:.1f}s", flush=True)
            verify_clips = sorted(verify_clips, key=lambda x: x["start"])
            verify_tensors, verify_fps_list = extract_hires_video(
                vreader, fps, duration, verify_clips,
                slow_max_frames=slow_max_frames,
                slow_video_resolution=slow_video_resolution,
                factor=factor,
            )
            turn3_user_content = [
                {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
                {"type": "video", "video": fast_video},
                {"type": "text", "text": "Verification clips (retrieved via clue similarity):"},
            ]
            for clip_info, clip_tensor in zip(verify_clips, verify_tensors):
                turn3_user_content.append({
                    "type": "text",
                    "text": f"[{clip_info['start']:.1f}s – {clip_info['end']:.1f}s]",
                })
                turn3_user_content.append({"type": "video", "video": clip_tensor})
            turn3_user_content.append({"type": "text", "text": TURN_VERIFY_USER})
            video_sample_fps_list.append(fast_sample_fps)
            video_sample_fps_list.extend(verify_fps_list)
            messages.append({"role": "user", "content": turn3_user_content})


            messages.append({"role": "assistant", "content": (
                f"<think>\n{think_list[2]}\n</think>\n\n"
                f"<final_answer>\n{parsed['final_answer']}\n</final_answer>"
            )})




        vreader.seek(0)
        del vreader

        video_kwargs = {'fps': video_sample_fps_list}
        print(f"[_get_item rank={_rank}] i={i} process_vision_info start elapsed={time.time()-_t0:.1f}s", flush=True)
        image_inputs, video_inputs = process_vision_info(messages)
        print(f"[_get_item rank={_rank}] i={i} DONE num_videos={len(video_inputs) if video_inputs else 0} elapsed={time.time()-_t0:.1f}s", flush=True)

        return {
            'image_inputs':     image_inputs,
            'video_inputs':     video_inputs,
            'messages':         messages,
            'video_kwargs':     video_kwargs,
            'mask_draft_turn':  mask_draft_turn,
        }






def find_assistant_spans(input_ids, start_tokens=[151644, 77091, 198], end_token=151645):
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("input_ids must be a torch.Tensor")

    device       = input_ids.device
    seq_len      = input_ids.shape[0]
    spans        = []
    start_tensor = torch.tensor(start_tokens, device=device)
    start_len    = len(start_tensor)

    i = 0
    while i <= seq_len - start_len:
        if torch.equal(input_ids[i:i + start_len], start_tensor):
            j = i + start_len
            while j < seq_len:
                if input_ids[j].item() == end_token:
                    spans.append((i + start_len, j + 1))
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return spans






class VideoSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        torch.cuda.empty_cache()
        return loss






def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    texts        = []
    image_inputs = []
    video_inputs = []
    mask_draft_turns = []

    for example in examples:
        texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
        image_inputs = example["image_inputs"]
        video_inputs = example["video_inputs"]
        video_kwargs = example["video_kwargs"]
        mask_draft_turns.append(example.get("mask_draft_turn", -1))

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        **video_kwargs
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    for visual_token_id in [151652, 151653, 151656, 151655]:
        labels[labels == visual_token_id] = -100


    for batch_idx, mask_turn in enumerate(mask_draft_turns):
        if mask_turn < 0:
            continue

        spans = find_assistant_spans(inputs["input_ids"][batch_idx])
        if mask_turn < len(spans):
            start, end = spans[mask_turn]
            labels[batch_idx, start:end] = -100

    inputs["labels"] = labels
    return inputs











if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )

    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )


    dataset = LazySupervisedDataset(
        script_args.dataset_name, script_args,
    )

    if model_config.freeze_vision_modules:
        print("Freezing vision modules...")
        for n, p in model.named_parameters():
            if any(kw in n for kw in ['visual']):
                p.requires_grad = False

    total_trainable_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:

            total_trainable_params += p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
    print("total_trainable_params:", total_trainable_params)

    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    trainer = VideoSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
