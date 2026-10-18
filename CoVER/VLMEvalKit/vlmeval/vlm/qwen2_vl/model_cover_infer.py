
from __future__ import annotations

import os
import re

os.environ.setdefault('DECORD_EOF_RETRY_MAX', '20480')
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

logger = logging.getLogger(__name__)





def add_frame_numbers_batch(frame_batch, frame_numbers):
    results = []
    for frame, frame_number in zip(frame_batch, frame_numbers):
        numbered_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(numbered_frame)

        width, height = numbered_frame.size
        font_size = int(min(width, height) / 8)
        try:
            font = ImageFont.truetype("../Arial.Unicode.ttf", size=font_size)
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


def sample_frames_uniform(
    vreader: VideoReader,
    num_frames: int,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[np.ndarray, List[int]]:
    total = len(vreader)
    if end_frame is None:
        end_frame = total - 1
    end_frame = min(end_frame, total - 1)
    num_frames = min(num_frames, end_frame - start_frame + 1)
    frame_ids = (
        torch.linspace(start_frame, end_frame, num_frames)
        .round().long().tolist()
    )
    frames = vreader.get_batch(frame_ids).asnumpy()
    return frames, frame_ids






def encode_clip_frames(
    frames: np.ndarray,
    frame_ids: List[int],
    clip_model,
    clip_processor,
    device: str = "cuda",
    batch_size: int = 32,
) -> torch.Tensor:
    clip_model.eval()
    frame_features_list = []
    for i in range(0, len(frames), batch_size):
        batch = [Image.fromarray(f.astype(np.uint8)) for f in frames[i: i + batch_size]]
        inputs = clip_processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(inputs.pixel_values)
            feats = F.normalize(feats, p=2, dim=-1)
        frame_features_list.append(feats)
    frame_features = torch.cat(frame_features_list, dim=0)
    logger.info(f"[CLIP] Encoded {len(frames)} frames, feature shape: {frame_features.shape}")
    return frame_features






def gaussian_smoothing(scores: torch.Tensor, radius: int = 4, sigma: float = 1.0) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=scores.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    scores_3d = scores.unsqueeze(0).unsqueeze(0)
    kernel_3d = kernel.view(1, 1, -1)
    smoothed  = F.conv1d(scores_3d, kernel_3d, padding=radius)
    return smoothed.squeeze(0).squeeze(0)


def tcs_select_clips_with_gaussian(
    sim_matrix: torch.Tensor,
    frame_ids: List[int],
    fps: float,
    num_clips: int = 3,
    clip_duration: float = 30.0,
    alpha: float = 0.5,
    smooth_radius: int = 4,
    smooth_sigma: float = 1.0,
) -> List[Dict]:

    scores = sim_matrix.mean(dim=0)


    smoothed = gaussian_smoothing(scores, radius=smooth_radius, sigma=smooth_sigma)


    mu        = smoothed.mean().item()
    sigma_val = smoothed.std().item()
    threshold = mu + alpha * sigma_val


    peaks = []
    for i in range(1, len(smoothed) - 1):
        if (smoothed[i] > threshold and
                smoothed[i] > smoothed[i - 1] and
                smoothed[i] > smoothed[i + 1]):
            peaks.append((i, smoothed[i].item()))


    candidate_clips = []
    for peak_idx, peak_val in peaks:
        center_time = frame_ids[peak_idx] / fps
        start_t = max(0.0, center_time - clip_duration / 2)
        end_t   = center_time + clip_duration / 2
        candidate_clips.append({"start": start_t, "end": end_t, "peak_val": peak_val})


    candidate_clips.sort(key=lambda x: x["start"])
    merged_clips = []
    if candidate_clips:
        cur_start     = candidate_clips[0]["start"]
        cur_end       = candidate_clips[0]["end"]
        cur_peak_vals = [candidate_clips[0]["peak_val"]]

        for clip in candidate_clips[1:]:
            if clip["start"] <= cur_end:
                cur_end = max(cur_end, clip["end"])
                cur_peak_vals.append(clip["peak_val"])
            else:
                merged_clips.append({
                    "start":      round(cur_start, 2),
                    "end":        round(cur_end, 2),
                    "query":      "tcs_aggregated",
                    "similarity": round(sum(cur_peak_vals) / len(cur_peak_vals), 4),
                })
                cur_start     = clip["start"]
                cur_end       = clip["end"]
                cur_peak_vals = [clip["peak_val"]]

        merged_clips.append({
            "start":      round(cur_start, 2),
            "end":        round(cur_end, 2),
            "query":      "tcs_aggregated",
            "similarity": round(sum(cur_peak_vals) / len(cur_peak_vals), 4),
        })


    merged_clips.sort(key=lambda x: x["similarity"], reverse=True)
    clips = merged_clips[:num_clips]

    logger.info(
        f"[TCS Gaussian] {len(candidate_clips)} peaks → "
        f"{len(merged_clips)} merged → top {len(clips)} (threshold={threshold:.4f})"
    )
    return clips


def select_clips_from_cached_features(
    cached_frame_features: torch.Tensor,
    cached_frame_ids: List[int],
    fps: float,
    queries: List[str],
    clip_model,
    clip_processor,
    num_clips: int = 3,
    clip_duration: float = 30.0,
    sim_threshold: float = 0.10,
    question: str = "",
    question_weight: float = 0.7,
) -> List[Dict]:
    if not queries:
        logger.warning("[CLIP TCS] No queries provided, returning empty clips")
        return []

    device = next(clip_model.parameters()).device
    clip_model.eval()

    def encode_text(text: str) -> torch.Tensor:

        inputs = clip_processor(
            text=[text], return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        ).to(device)
        with torch.no_grad():
            feat = clip_model.get_text_features(**inputs)
            feat = F.normalize(feat, p=2, dim=-1)
        return feat


    pq_features_list = [encode_text(q) for q in queries]
    pq_features = torch.cat(pq_features_list, dim=0)
    sim_pq = torch.matmul(pq_features, cached_frame_features.T).mean(dim=0, keepdim=True)


    if question.strip():
        q_feat = encode_text(question)
        sim_q  = torch.matmul(q_feat, cached_frame_features.T)

        sim_weighted = question_weight * sim_q + (1.0 - question_weight) * sim_pq
        logger.info(f"[CLIP TCS] question_weight={question_weight}, pq_weight={1.0 - question_weight:.2f}")
    else:
        sim_weighted = sim_pq
        logger.info("[CLIP TCS] No question provided, using pseudo_queries only")

    clips = tcs_select_clips_with_gaussian(
        sim_matrix=sim_weighted,
        frame_ids=cached_frame_ids,
        fps=fps,
        num_clips=num_clips,
        clip_duration=clip_duration,
        alpha=1.0,
        smooth_radius=4,
        smooth_sigma=1.0,
    )
    clips = [c for c in clips if c["similarity"] >= sim_threshold]
    logger.info(f"[CLIP TCS] Selected {len(clips)} clips from {len(queries)} pseudo_queries (cached)")
    return clips


def select_verify_clips_from_cached_features(
    cached_frame_features: torch.Tensor,
    cached_frame_ids: List[int],
    fps: float,
    clue: str,
    clip_model,
    clip_processor,
    num_clips: int = 3,
    clip_duration: float = 10.0,
    sim_threshold: float = 0.10,
) -> List[Dict]:
    if not clue:
        logger.warning("[CLIP Verify] No clue provided, returning empty clips")
        return []

    device = next(clip_model.parameters()).device
    clip_model.eval()

    inputs = clip_processor(
        text=[clue], return_tensors="pt", padding=True,
        truncation=True, max_length=77,
    ).to(device)
    with torch.no_grad():
        clue_feature = clip_model.get_text_features(**inputs)
        clue_feature = F.normalize(clue_feature, p=2, dim=-1)


    sim_matrix = torch.matmul(clue_feature, cached_frame_features.T)

    clips = tcs_select_clips_with_gaussian(
        sim_matrix=sim_matrix,
        frame_ids=cached_frame_ids,
        fps=fps,
        num_clips=num_clips,
        clip_duration=clip_duration,
        alpha=1.0,
        smooth_radius=4,
        smooth_sigma=1.0,
    )
    clips = [c for c in clips if c["similarity"] >= sim_threshold]

    logger.info(
        f"[CLIP Verify TCS] Selected {len(clips)} clips from clue "
        f"(threshold={sim_threshold})"
    )
    return clips


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
    durations    = [c["end"] - c["start"] for c in sorted_clips]
    total_dur    = sum(durations)

    frames_per_clip = [
        max(2, int(d / total_dur * slow_max_frames)) if total_dur > 0 else 2
        for d in durations
    ]
    cur_total = sum(frames_per_clip)
    if cur_total > slow_max_frames:
        scale = slow_max_frames / cur_total
        frames_per_clip = [max(2, int(f * scale)) for f in frames_per_clip]

    result_tensors, result_fps = [], []
    for clip, num_f in zip(sorted_clips, frames_per_clip):
        start_f   = max(0, int(clip["start"] * fps))
        end_f     = min(total_frames - 1, int(clip["end"] * fps))
        frame_ids = torch.linspace(start_f, end_f, num_f).round().long().tolist()
        clip_fps  = round(num_f / max(1, end_f - start_f) * fps, 2)

        raw = vreader.get_batch(frame_ids).asnumpy()
        if raw.ndim == 3:
            raw = np.stack([raw] * 3, axis=-1)
        raw    = raw.astype(np.uint8)
        secs   = [int(fid / fps) for fid in frame_ids]
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






class Qwen2VLChatCoVERInfer(Qwen2VLPromptMixin, BaseModel):

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
        repetition_penalty: float = 1.3,
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
        verification_num_clips: int = 3,
        verification_clip_duration: float = 10.0,
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
        self.verification_num_clips   = verification_num_clips
        self.verification_clip_duration = verification_clip_duration


        self.fps    = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)


        self._cached_clip_frame_features: Optional[torch.Tensor] = None
        self._cached_clip_frame_ids: Optional[List[int]]         = None




        assert model_path is not None
        self.model_path = model_path
        logger.info(f"Loading model from: {model_path}")


        _is_qwen25 = listinstr(["2.5", "2_5", "qwen25"], model_path.lower())
        if not _is_qwen25:
            try:
                import json as _json
                with open(os.path.join(model_path, "config.json")) as _f:
                    _cfg = _json.load(_f)
                if "qwen2_5" in _cfg.get("model_type", "").lower() or \
                   "qwen2.5" in _cfg.get("_name_or_path", "").lower():
                    _is_qwen25 = True
            except Exception:
                pass
        if _is_qwen25:
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

        try:
            fast_video_raw = vreader.get_batch(fast_frame_ids).asnumpy()
        except Exception as e:
            logger.warning(f"[Stage 0] Failed to read video frames from {video_path}: {e}, skipping.")
            return ''
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



            logger.info("[Branch] Pattern B (insufficient evidence, 3 turns, 3 LLM calls)")


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
            messages.append({"role": "assistant", "content": turn2_output})


            clue         = self._extract_tag(turn2_output, "clue")
            draft_answer = self._extract_tag(turn2_output, "draft_answer")
            logger.info(f"[Turn 2 PatB] draft_answer={repr(draft_answer[:80])}, clue={repr(clue)}")














            verify_clips = self._locate_verify_clips(fps, clue,
                                                     video_path=video_path, question=question,
                                                     turn1_output=turn1_output,
                                                     turn2_output=turn2_output,
                                                     draft_answer=draft_answer)
            logger.info(f"[CLIP Verify] Located {len(verify_clips)} verification clips")

            verify_tensors, verify_fps_list = extract_hires_video(
                vreader, fps, duration, verify_clips,
                slow_max_frames=self.slow_max_frames,
                slow_video_resolution=self.slow_video_resolution,
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
            messages.append({"role": "user", "content": turn3_user_content})


            turn3_output = self._run_model(
                messages, max_new_tokens=self.max_new_tokens, temperature=0.3
            )
            logger.info(f"[Turn 3 PatB] Output[:200]: {turn3_output[:200]}")

            final_answer = self._extract_final_answer(turn3_output)
            pattern      = "B"

        else:



            logger.info("[Branch] Pattern A (sufficient evidence, 2 turns, 2 LLM calls)")

            clue         = clue_t1
            draft_answer = draft_answer_t1
            logger.info(f"[Turn 1 PatA] draft_answer={draft_answer[:80]}, clue={clue[:100]}")














            verify_clips = self._locate_verify_clips(fps, clue,
                                                     video_path=video_path, question=question,
                                                     turn1_output=turn1_output,
                                                     draft_answer=draft_answer)
            logger.info(f"[CLIP Verify] Located {len(verify_clips)} verification clips")

            verify_tensors, verify_fps_list = extract_hires_video(
                vreader, fps, duration, verify_clips,
                slow_max_frames=self.slow_max_frames,
                slow_video_resolution=self.slow_video_resolution,
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
            messages.append({"role": "user", "content": turn2_user_content})


            turn2_output = self._run_model(
                messages, max_new_tokens=self.max_new_tokens, temperature=0.3
            )
            logger.info(f"[Turn 2 PatA] Output[:200]: {turn2_output[:200]}")

            final_answer = self._extract_final_answer(turn2_output)
            pattern      = "A"





        logger.info(f"[Result] pattern={pattern}, final_answer={final_answer}")
        del vreader
        return final_answer





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
            reasons = []
            if self.clip_model is None:
                reasons.append("clip_model=None (CLIP model not loaded)")
            if self._cached_clip_frame_features is None:
                reasons.append("_cached_clip_frame_features=None (no cached frame features)")
            if self._cached_clip_frame_ids is None:
                reasons.append("_cached_clip_frame_ids=None (no cached frame ids)")
            logger.warning(
                f"[CLIP Zoom] falling back to uniform split, reasons: {'; '.join(reasons)}"
            )
            duration = (self._cached_clip_frame_ids[-1] / fps
                        if self._cached_clip_frame_ids else 30.0)
            clips = []
            step = duration / max(1, self.num_clips)
            for i in range(self.num_clips):
                start = i * step
                end   = min(start + self.clip_duration, duration)
                clips.append({
                    "start":      round(start, 2),
                    "end":        round(end, 2),
                    "query":      pq_lines[i] if i < len(pq_lines) else "",
                    "similarity": 0.0,
                })

        return sorted(clips, key=lambda x: x["start"])

    def _locate_verify_clips(self, fps: float, clue: str,
                             video_path: str = "", question: str = "",
                             **cot_kwargs) -> List[Dict]:
        if (self.clip_model is not None and
                self._cached_clip_frame_features is not None and
                self._cached_clip_frame_ids is not None and
                clue):
            clips = select_verify_clips_from_cached_features(
                cached_frame_features=self._cached_clip_frame_features,
                cached_frame_ids=self._cached_clip_frame_ids,
                fps=fps,
                clue=clue,
                clip_model=self.clip_model,
                clip_processor=self.clip_processor,
                num_clips=self.verification_num_clips,
                clip_duration=self.verification_clip_duration,
                sim_threshold=self.clip_sim_threshold,
            )
        else:
            reasons = []
            if self.clip_model is None:
                reasons.append("clip_model=None (CLIP model not loaded)")
            if self._cached_clip_frame_features is None:
                reasons.append("_cached_clip_frame_features=None (no cached frame features)")
            if self._cached_clip_frame_ids is None:
                reasons.append("_cached_clip_frame_ids=None (no cached frame ids)")
            if not clue:
                reasons.append(f"empty clue (clue={repr(clue)})")
            reason_str = "; ".join(reasons)
            logger.warning(
                f"[CLIP Verify] falling back to uniform split, reasons: {reason_str}"
            )

            duration = (self._cached_clip_frame_ids[-1] / fps
                        if self._cached_clip_frame_ids else 30.0)
            clips = []
            step = duration / max(1, self.verification_num_clips)
            for i in range(self.verification_num_clips):
                start = i * step
                end   = min(start + self.verification_clip_duration, duration)
                clips.append({
                    "start":      round(start, 2),
                    "end":        round(end, 2),
                    "query":      clue,
                    "similarity": 0.0,
                })

        return sorted(clips, key=lambda x: x["start"])





    def _run_model(
        self,
        messages: list,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
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





    def _extract_vision_inputs(
        self, messages: list
    ) -> Tuple[Optional[list], Optional[list]]:
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





    def _parse_input(
        self, message: list, dataset: Optional[str] = None
    ) -> Tuple[str, str]:
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





    def _retry_clue(
        self,
        messages_before_gen: list,
        initial_output: str,
        max_retries: int = 3,
        temperature: float = 0.5,
    ) -> Tuple[str, str]:
        for attempt in range(1, max_retries + 1):
            retry_output = self._run_model(
                messages_before_gen,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
            )
            retry_clue = self._extract_tag(retry_output, "clue")
            logger.info(
                f"[Clue Retry {attempt}/{max_retries}] "
                f"clue={repr(retry_clue)}, output[:80]={retry_output[:80]}"
            )
            if retry_clue:
                logger.info(f"[Clue Retry] clue extracted on attempt {attempt}, stopping retries")
                return retry_clue, retry_output

        logger.warning(
            f"[Clue Retry] no clue produced in {max_retries} attempts, falling back to uniform split"
        )
        return "", initial_output

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

    def _extract_final_answer(self, output_text: str) -> str:
        boxed = self._extract_boxed(output_text)
        if boxed:
            return boxed
        final_answer_tag = self._extract_tag(output_text, "final_answer")
        if final_answer_tag:
            inner_boxed = self._extract_boxed(final_answer_tag)
            return inner_boxed if inner_boxed else final_answer_tag
        return output_text.strip()

