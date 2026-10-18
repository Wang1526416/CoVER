
from __future__ import annotations

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Qwen2VLProcessor,
)

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from .vision_process import process_vision_info, get_video_hw
from ...smp import get_rank_and_world_size, get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY


from .model_cover_infer import (
    add_frame_numbers_batch,
    sample_frames_uniform,
    encode_clip_frames,
    gaussian_smoothing,
    tcs_select_clips_with_gaussian,
    select_clips_from_cached_features,
    extract_hires_video,
    SYSTEM_PROMPT,
    TURN1_USER_SUFFIX,
    TURN2_B_USER_SUFFIX,
)

logger = logging.getLogger(__name__)


class Qwen2VLChatCoVERDraftInfer(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE  = True
    VIDEO_LLM   = True

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        top_p: float = 0.001,
        top_k: int = 1,
        temperature: float = 0.3,
        repetition_penalty: float = 1.0,
        use_custom_prompt: bool = True,
        system_prompt: Optional[str] = None,
        verbose: bool = True,

        fast_max_frames: int = 768,
        fast_video_resolution: int = 32,

        slow_max_frames: int = 32,
        slow_video_resolution: int = 256,

        clip_model_name: str = "./Model/clip14",
        num_clips: int = 3,
        clip_duration: float = 30.0,
        clip_sim_threshold: float = 0.15,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)

        self.max_new_tokens          = max_new_tokens
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt            = system_prompt
        self.verbose                  = verbose
        self.fast_max_frames          = fast_max_frames
        self.fast_video_resolution    = fast_video_resolution
        self.slow_max_frames          = slow_max_frames
        self.slow_video_resolution    = slow_video_resolution
        self.clip_model_name          = clip_model_name
        self.num_clips                = num_clips
        self.clip_duration            = clip_duration
        self.clip_sim_threshold       = clip_sim_threshold

        self.fps    = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)


        self._cached_clip_frame_features: Optional[torch.Tensor] = None
        self._cached_clip_frame_ids: Optional[List[int]]         = None


        assert model_path is not None
        self.model_path = model_path
        logger.info(f"Loading model from: {model_path}")

        if listinstr(["2.5", "2_5", "qwen25"], model_path.lower()):
            MODEL_CLS      = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            MODEL_CLS      = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        self.model = MODEL_CLS.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        torch.cuda.empty_cache()


        self._clip_model     = None
        self._clip_processor = None





    @property
    def clip_model(self):
        if self._clip_model is None:
            self._load_clip()
        return self._clip_model

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            self._load_clip()
        return self._clip_processor

    def _load_clip(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info(f"Loading CLIP model: {self.clip_model_name}")
            self._clip_model     = CLIPModel.from_pretrained(
                self.clip_model_name, use_safetensors=True
            ).to("cuda")
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self._clip_model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}. Will use uniform fallback.")
            self._clip_model     = None
            self._clip_processor = None





    def generate_inner(self, message, dataset=None):
        video_path, question = self._parse_input(message, dataset)
        if self.verbose:
            logger.info(f"[Input] video={video_path}, question={question}")




        vreader          = VideoReader(video_path, ctx=cpu(0))
        fps              = vreader.get_avg_fps()
        total_num_frames = len(vreader)
        duration         = total_num_frames / fps

        factor   = 2 if duration < 120 else 1
        num_fast = min(int(duration * 2), self.fast_max_frames)
        num_fast = min(num_fast, total_num_frames)

        fast_frame_ids  = torch.linspace(0, total_num_frames - 1, num_fast).round().long().tolist()
        fast_sample_fps = round(num_fast / max(1, total_num_frames - 1) * fps, 2)

        fast_video_raw = vreader.get_batch(fast_frame_ids).asnumpy()
        if fast_video_raw.ndim == 3:
            fast_video_raw = np.stack([fast_video_raw] * 3, axis=-1)
        fast_video_raw  = fast_video_raw.astype(np.uint8)
        fast_frame_secs = [int(fid / fps) for fid in fast_frame_ids]
        fast_video      = add_frame_numbers_batch(fast_video_raw, fast_frame_secs)

        H, W = fast_video.shape[-2:]
        rh, rw = get_video_hw(H, W, self.fast_video_resolution * 28 * 28 * factor)
        fast_video = transforms.functional.resize(
            fast_video.float(), [rh, rw],
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )

        logger.info(
            f"[Stage 0] Sampled {num_fast} frames at {rh}x{rw}, "
            f"duration={duration:.1f}s, fps={fps:.2f}"
        )


        if self.clip_model is not None:
            logger.info("[Stage 0] Encoding CLIP frame features (one-time cache)...")
            self._cached_clip_frame_features = encode_clip_frames(
                fast_video_raw, fast_frame_ids,
                self.clip_model, self.clip_processor,
                device="cuda", batch_size=32,
            )
            self._cached_clip_frame_ids = fast_frame_ids
        else:
            self._cached_clip_frame_features = None
            self._cached_clip_frame_ids      = None




        turn1_user_text = TURN1_USER_SUFFIX.format(
            duration=duration, fps=fps,
            total_frames=total_num_frames, question=question,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text":  f"Full video [0, {int(duration)}]:"},
                    {"type": "video", "video": fast_video},
                    {"type": "text",  "text":  turn1_user_text},
                ],
            },
        ]

        turn1_output = self._run_model(
            messages, max_new_tokens=self.max_new_tokens, temperature=0.3
        )
        logger.info(f"[Turn 1] FULL OUTPUT:\n{turn1_output}")
        messages.append({"role": "assistant", "content": turn1_output})




        pseudo_queries_text = self._extract_tag(turn1_output, "pseudo_queries")
        draft_answer_t1     = self._extract_tag(turn1_output, "draft_answer")
        clue_t1             = self._extract_tag(turn1_output, "clue")

        is_pattern_b = bool(pseudo_queries_text.strip()) and not bool(draft_answer_t1.strip())

        if is_pattern_b:



            logger.info("[Branch] Pattern B (insufficient evidence, 2 LLM calls for draft)")

            pq_lines = [
                re.sub(r'^\d+\.\s*', '', l).strip()
                for l in pseudo_queries_text.splitlines()
                if l.strip()
            ]
            logger.info(f"[Turn 1] pseudo_queries: {pq_lines}")


            zoom_clips = self._locate_zoom_clips(fps, pq_lines, question=question)
            logger.info(f"[CLIP Zoom] Located {len(zoom_clips)} zoom-in clips")

            zoom_tensors, zoom_fps_list = extract_hires_video(
                vreader, fps, duration, zoom_clips,
                slow_max_frames=self.slow_max_frames,
                slow_video_resolution=self.slow_video_resolution,
                factor=factor,
            )


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
            turn2_user_content.append({
                "type": "text",
                "text": TURN2_B_USER_SUFFIX.format(question=question),
            })
            messages.append({"role": "user", "content": turn2_user_content})


            turn2_output = self._run_model(
                messages, max_new_tokens=self.max_new_tokens, temperature=0.3
            )
            logger.info(f"[Turn 2 PatB] FULL OUTPUT:\n{turn2_output}")

            draft_answer = self._extract_draft_answer(turn2_output)
            clue         = self._extract_tag(turn2_output, "clue")
            pattern      = "B"

            logger.info(f"[Turn 2 PatB] draft_answer={repr(draft_answer)}, clue={repr(clue[:100])}")

        else:



            logger.info("[Branch] Pattern A (sufficient evidence, 1 LLM call for draft)")

            draft_answer = self._extract_draft_answer(turn1_output)
            clue         = clue_t1
            pattern      = "A"

            logger.info(f"[Turn 1 PatA] draft_answer={repr(draft_answer)}, clue={repr(clue[:100])}")





        logger.info(f"[Result] pattern={pattern}, draft_answer={draft_answer}")
        del vreader
        return draft_answer





    def _locate_zoom_clips(self, fps: float, pq_lines: List[str], question: str = "") -> List[Dict]:
        if (self.clip_model is not None and
                self._cached_clip_frame_features is not None and
                self._cached_clip_frame_ids is not None):
            clips = select_clips_from_cached_features(
                cached_frame_features=self._cached_clip_frame_features,
                cached_frame_ids=self._cached_clip_frame_ids,
                fps=fps,
                queries=pq_lines,
                clip_model=self.clip_model,
                clip_processor=self.clip_processor,
                num_clips=self.num_clips,
                clip_duration=self.clip_duration,
                sim_threshold=self.clip_sim_threshold,
                question=question,
                question_weight=0.7,
            )
        else:
            duration = (self._cached_clip_frame_ids[-1] / fps
                        if self._cached_clip_frame_ids else 30.0)
            clips = []
            step = duration / max(1, self.num_clips)
            for i in range(self.num_clips):
                start = i * step
                end   = min(start + self.clip_duration, duration)
                clips.append({
                    "start": round(start, 2), "end": round(end, 2),
                    "query": pq_lines[i] if i < len(pq_lines) else "",
                    "similarity": 0.0,
                })
        return sorted(clips, key=lambda x: x["start"])





    def _run_model(self, messages: list, max_new_tokens: int = 1024, temperature: float = 0.3) -> str:
        text_input = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ]
        images, videos = self._extract_vision_inputs(messages)

        inputs = self.processor(
            text=text_input,
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.01),
            )

        input_len   = inputs["input_ids"].shape[1]
        new_ids     = generated_ids[:, input_len:]
        output_text = self.processor.tokenizer.decode(
            new_ids[0], skip_special_tokens=True
        )
        return output_text

    def _extract_vision_inputs(self, messages: list) -> Tuple[Optional[list], Optional[list]]:
        try:
            images, videos = process_vision_info([messages])
            return images, videos
        except Exception:
            pass
        images, videos = [], []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            img = item.get("image")
                            if img is not None:
                                images.append(img)
                        elif item.get("type") == "video":
                            vid = item.get("video")
                            if vid is not None:
                                videos.append(vid)
        return images or None, videos or None





    def _parse_input(self, message: list, dataset: Optional[str] = None) -> Tuple[str, str]:
        video_path = None
        for msg in message:
            if msg.get("type") == "video":
                video_path = msg["value"]
                break

        if dataset == "Video-MME":
            question = message[-1]["value"].split("Question: ")[-1].split("Answer:")[0].strip()
        elif dataset == "MVBench":
            question = message[-3]["value"].split("Question: ")[-1].strip()
        elif dataset in ("MLVU_MCQ", "MLVU"):
            question = message[-2]["value"].split("Question: ")[-1].strip()
        elif dataset == "LVBench":
            question = message[-1]["value"].split("Question: ")[-1].split("Answer:")[0].strip()
        else:
            question = " ".join(
                m["value"] for m in message if m.get("type") == "text"
            ).strip()

        assert video_path is not None, "No video path found in message"
        return video_path, question





    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_boxed(text: str) -> str:
        m = re.search(r"\\boxed\{\{(.*?)\}\}", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _extract_draft_answer(self, output_text: str) -> str:
        draft_tag = self._extract_tag(output_text, "draft_answer")
        if draft_tag:
            boxed = self._extract_boxed(draft_tag)
            if boxed:
                return boxed
            return draft_tag.strip()

        boxed = self._extract_boxed(output_text)
        if boxed:
            return boxed
        return output_text.strip()

