
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
        text_width = text_bbox[2] - text_bbox[0] + 10
        text_height = text_bbox[3] - text_bbox[1] + 10

        x = width - text_width
        y = height - text_height - text_height / 3

        draw.text((x, y), text, font=font, fill='red')
        results.append(torch.tensor(np.array(numbered_frame)))
    results = torch.stack(results).permute(0, 3, 1, 2)
    return results


def sample_frames_uniform(
    vreader: VideoReader,
    num_frames: int,
    start_frame: int = 0,
    end_frame: Optional[int] = None
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



    scores = scores.unsqueeze(0).unsqueeze(0)
    kernel = kernel.view(1, 1, -1)
    smoothed = F.conv1d(scores, kernel, padding=radius)
    return smoothed.squeeze(0).squeeze(0)


def tcs_select_clips_with_gaussian(
    sim_matrix: torch.Tensor,
    frame_ids: List[int],
    fps: float,
    num_clips: int = 3,
    clip_duration: float = 4.0,
    alpha: float = 0.5,
    smooth_radius: int = 4,
    smooth_sigma: float = 1.0,
) -> List[Dict]:


    scores = sim_matrix.mean(dim=0)



    smoothed = gaussian_smoothing(scores, radius=smooth_radius, sigma=smooth_sigma)



    mu = smoothed.mean().item()
    sigma = smoothed.std().item()
    threshold = mu + alpha * sigma



    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > threshold and smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            peaks.append((i, smoothed[i].item()))


    peaks.sort(key=lambda x: x[1], reverse=True)


    candidate_clips = []
    for peak_idx, peak_val in peaks:
        center_time = frame_ids[peak_idx] / fps
        start_t = max(0.0, center_time - clip_duration / 2)
        end_t = center_time + clip_duration / 2
        candidate_clips.append({
            "start": start_t,
            "end": end_t,
            "peak_val": peak_val,
            "peak_idx": peak_idx,
        })


    candidate_clips.sort(key=lambda x: x["start"])


    merged_clips = []
    if candidate_clips:
        current_start = candidate_clips[0]["start"]
        current_end = candidate_clips[0]["end"]
        current_peak_vals = [candidate_clips[0]["peak_val"]]

        for clip in candidate_clips[1:]:

            if clip["start"] <= current_end:
                current_end = max(current_end, clip["end"])
                current_peak_vals.append(clip["peak_val"])
            else:

                avg_sim = sum(current_peak_vals) / len(current_peak_vals)
                merged_clips.append({
                    "start": round(current_start, 2),
                    "end": round(current_end, 2),
                    "query": "tcs_aggregated",
                    "similarity": round(avg_sim, 4),
                })

                current_start = clip["start"]
                current_end = clip["end"]
                current_peak_vals = [clip["peak_val"]]


        if current_peak_vals:
            avg_sim = sum(current_peak_vals) / len(current_peak_vals)
            merged_clips.append({
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "query": "tcs_aggregated",
                "similarity": round(avg_sim, 4),
            })


    merged_clips.sort(key=lambda x: x["similarity"], reverse=True)
    clips = merged_clips[:num_clips]

    peak_vals = [f"{p[1]:.4f}" for p in peaks[:num_clips]]
    logger.info(f"[TCS Gaussian] Merged from {len(candidate_clips)} peaks to {len(merged_clips)} clips, selected top {len(clips)} (threshold={threshold:.4f})")
    return clips


def select_clips_from_cached_features(
    cached_frame_features: torch.Tensor,
    cached_frame_ids: List[int],
    fps: float,
    queries: List[str],
    clip_model,
    clip_processor,
    num_clips: int = 3,
    clip_duration: float = 4.0,
    sim_threshold: float = 0.20,
) -> List[Dict]:
    device = next(clip_model.parameters()).device
    clip_model.eval()


    if not queries:
        logger.warning("[CLIP TCS] No queries provided, returning empty clips")
        return []


    query_features_list = []
    for q in queries:
        inputs = clip_processor(text=[q], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = clip_model.get_text_features(**inputs)
            feat = F.normalize(feat, p=2, dim=-1)
        query_features_list.append(feat)
    query_features = torch.cat(query_features_list, dim=0)


    sim_matrix = torch.matmul(query_features, cached_frame_features.T)


    clips = tcs_select_clips_with_gaussian(
        sim_matrix=sim_matrix,
        frame_ids=cached_frame_ids,
        fps=fps,
        num_clips=num_clips,
        clip_duration=clip_duration,
        alpha=0.5,
        smooth_radius=4,
        smooth_sigma=1.0,
    )


    clips = [c for c in clips if c["similarity"] >= sim_threshold]

    logger.info(f"[CLIP TCS] Selected {len(clips)} clips from {len(queries)} queries (using cached features)")
    return clips


def cosine_select_clips(
    frames: np.ndarray,
    frame_ids: List[int],
    fps: float,
    queries: List[str],
    clip_model,
    clip_processor,
    num_clips: int = 3,
    clip_duration: float = 4.0,
    sim_threshold: float = 0.20,
) -> List[Dict]:
    device = next(clip_model.parameters()).device
    clip_model.eval()


    batch_size = 32
    frame_features_list = []
    for i in range(0, len(frames), batch_size):
        batch = [Image.fromarray(f.astype(np.uint8)) for f in frames[i: i + batch_size]]
        inputs = clip_processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(inputs.pixel_values)
            feats = F.normalize(feats, p=2, dim=-1)
        frame_features_list.append(feats)
    frame_features = torch.cat(frame_features_list, dim=0)


    query_features_list = []
    for q in queries:
        inputs = clip_processor(text=[q], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = clip_model.get_text_features(**inputs)
            feat = F.normalize(feat, p=2, dim=-1)
        query_features_list.append(feat)
    query_features = torch.cat(query_features_list, dim=0)


    sim_matrix = torch.matmul(query_features, frame_features.T)


    clips = tcs_select_clips_with_gaussian(
        sim_matrix=sim_matrix,
        frame_ids=frame_ids,
        fps=fps,
        num_clips=num_clips,
        clip_duration=clip_duration,
        alpha=0.5,
        smooth_radius=4,
        smooth_sigma=1.0,
    )


    clips = [c for c in clips if c["similarity"] >= sim_threshold]

    logger.info(f"[CLIP TCS] Selected {len(clips)} clips from {len(queries)} queries")
    return clips






class Qwen2VLChatSampleFrameThinkVerify(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True





    STAGE1_SYSTEM = (
        "You are an expert video analyst. "
        "Your task is to determine if the visual evidence from low-resolution frames "
        "is sufficient to confidently answer the given question."
    )

    STAGE1_USER = """\
Low-resolution overview frames from the full video are provided.
Each frame is labeled with its timestamp in seconds.

Video:
- Duration: {duration:.1f}s
- FPS: {fps:.2f}
- Total frames: {total_frames}

Question: {question}

Task:
Decide whether the current visual evidence is sufficient to answer the question confidently.

Rules:
- "sufficient=yes" only if the key evidence needed for the answer is clearly visible.
- "sufficient=no" if critical objects, actions, or temporal relations are unclear or missing.
- In <key_observations>, list only question-relevant visible facts.
- In <missing_information>, list only question-relevant missing visual evidence.
- If sufficient=no, generate exactly 3 pseudo-queries for retrieval.

Pseudo-query rules:
- Each query must check ONE specific visual fact.
- Focus on object identity, object state, human action, hand-object interaction, or before/after relation.
- Do NOT ask vague questions like "What is happening?" or "What does the person do?"
- Write each query as a direct English question ending with a question mark.

Output EXACTLY in this format:

<evidence_check>
<sufficient>yes or no</sufficient>
<key_observations>
- ...
- ...
</key_observations>
<missing_information>
- ...
- ...
</missing_information>
</evidence_check>

If sufficient=yes:
\\boxed{{answer}}

If sufficient=no:
<pseudo_queries>
1. ...
2. ...
3. ...
</pseudo_queries>
"""





    STAGE23_SYSTEM = (
        "You are an expert video analyst and hypothesis generator. "
        "Your task is to: (1) generate a candidate answer based on evidence, "
        "(2) convert it into a verifiable hypothesis, "
        "(3) extract a discriminative clue, and (4) generate verification queries."
    )

    STAGE23_USER_PATH_B = """\
You are given a question and HIGH-RESOLUTION frames from the most relevant video clips,
selected to resolve missing visual evidence.

Missing information from the global overview:
{missing_info}

Question: {question}

The high-resolution clips below are ordered by relevance:
{clip_descriptions}

Do the following:

1. Briefly explain your reasoning before giving the answer.
   - The reasoning must directly address the missing information.
   - The reasoning must match the question condition, especially any temporal relation such as before / after / during / next.
   - The reasoning must explain why the chosen answer is supported better than other plausible options.
   - Keep it short: 1 to 3 sentences.

2. Generate the best candidate answer based only on visible evidence.

3. Write a verifiable hypothesis.
   - A hypothesis means: if the answer is correct, what observable facts should be visible in the video?
   - Write 2 to 4 short observable facts.
   - These facts must be consistent with your reasoning and answer.

4. Write one discriminative clue.
   - A clue is the single most useful visual check for later verification.
   - It must be short, specific, and visually checkable.
   - It must be derived from the hypothesis and help distinguish this answer from competing options.

Rules:
- Use only visible evidence from the high-resolution frames.
- If the question is multiple-choice, choose the best-supported option.
- Do not write vague statements.
- The reasoning, answer, hypothesis, and clue must all support the same conclusion.

Output EXACTLY in this format:

<answer_reasoning>
[1-3 short sentences explaining why this answer resolves the missing information and matches the question condition]
</answer_reasoning>

<candidate_answer>
[best-supported answer]
</candidate_answer>

<hypothesis>
1. [observable fact 1]
2. [observable fact 2]
3. [observable fact 3]
</hypothesis>

<clue>
[one short discriminative visual check]
</clue>
"""

    STAGE23_USER_PATH_A = """\
You are given a question and have performed an initial global analysis.
Complete the following three tasks sequentially:

Question: {question}

Initial observations: {initial_observations}

Candidate answer from Stage 1: {candidate_answer}

--------------------
Part 1: Hypothesis Generation

Based on the candidate answer above, transform it into a VERIFIABLE hypothesis.
A good hypothesis describes what MUST be visible in the video if the answer is correct.

<hypothesis>
If the current answer is correct, the video should show:
1. [Specific observable fact 1]
2. [Specific observable fact 2]
3. [Specific observable fact 3]
...
</hypothesis>

--------------------
Part 2: Clue Generation

Extract a SINGLE, concise discriminative clue from the hypothesis.

<clue>
[One specific, short, actionable visual check]
</clue>
"""






    STAGE345_SYSTEM = (
        "You are an expert hypothesis generator, clue extractor, "
        "and verification query designer. "
        "Your task is to: (1) convert a candidate answer into a verifiable hypothesis, "
        "(2) extract a discriminative clue, and (3) generate verification queries."
    )

    STAGE345_USER = """\
You are given a question and a candidate answer based on preliminary visual evidence.
Complete the following three tasks sequentially:

--------------------
Part 1: Hypothesis Generation

Transform the candidate answer into a VERIFIABLE hypothesis.
A good hypothesis describes what MUST be visible in the video if the answer is correct.

Format the hypothesis as a sequence of observable facts:
- Entity/object presence or absence
- Specific actions or state changes
- Temporal ordering (before/after/during)
- Causal relationships

<hypothesis>
If the current answer is correct, the video should show:
1. [Specific observable fact 1]
2. [Specific observable fact 2]
3. [Specific observable fact 3]
...
</hypothesis>

--------------------
Part 2: Clue Generation

Extract a SINGLE, concise discriminative clue from the hypothesis.
The clue should be short and actionable for retrieval.

Examples of good clues:
- "Check if the person is holding an object with a narrow neck"
- "Verify the color of the object immediately after the action"
- "Confirm the spatial position of the object (left/right hand)"

<clue>
[One specific, short, actionable visual check]
</clue>
"""





    STAGE56_SYSTEM = (
        "You are an expert video verifier and final decision maker. "
        "Your task is to: (1) determine the verification state based on visual evidence, "
        "and (2) integrate all evidence to produce the final answer."
    )

    STAGE56_USER = """\
You are given a question, an initial answer, a hypothesis, a clue, and high-resolution verification clips.

Question: {question}

Initial Candidate Answer: {initial_answer}

Hypothesis:
{hypothesis}

Clue:
{clue}

Initial Evidence Summary:
{initial_evidence_summary}

Task:
1. Determine the verification state from the verification clips.
2. Briefly explain whether the initial answer should be kept or revised.
3. Produce the final answer.

Verification state definition:
- VERIFIED: the clue is clearly supported by visual evidence.
- PARTIAL: the clue is only weakly or ambiguously supported.
- NOT_VERIFIED: the clue is not supported, or is contradicted by the visual evidence.

Decision rule:
- If VERIFIED: keep the current answer.
- If PARTIAL: keep it only if it is still the best-supported choice.
- If NOT_VERIFIED: revise to a better-supported answer.

Requirements:
- Keep the explanation short: 1 to 2 sentences.
- You must give a definite answer.
- Do not answer with "unclear", "cannot determine", or "need more information".
- If this is multiple-choice, choose exactly one option.
- Use \\boxed{{...}} for the final answer.

Output EXACTLY in this format:

<verification_result>
<verification_state>VERIFIED / PARTIAL / NOT_VERIFIED</verification_state>
<decision_reason>
[brief reason for keeping or revising the answer]
</decision_reason>
</verification_result>

<final_answer>
\\boxed{{answer}}
</final_answer>
"""


    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        top_p: float = 0.001,
        top_k: int = 1,
        temperature: float = 0.01,
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
        clip_sim_threshold: float = 0.20,

        verification_num_clips: int = 3,
        verification_clip_duration: float = 10.0,

        enable_verification_retry: bool = False,
        enable_hypothesis_revision: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)

        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fast_max_frames = fast_max_frames
        self.fast_video_resolution = fast_video_resolution
        self.slow_max_frames = slow_max_frames
        self.slow_video_resolution = slow_video_resolution
        self.clip_model_name = clip_model_name
        self.num_clips = num_clips
        self.clip_duration = clip_duration
        self.clip_sim_threshold = clip_sim_threshold
        self.verification_num_clips = verification_num_clips
        self.verification_clip_duration = verification_clip_duration
        self.enable_verification_retry = enable_verification_retry
        self.enable_hypothesis_revision = enable_hypothesis_revision


        self._cached_clip_frame_features: Optional[torch.Tensor] = None
        self._cached_clip_frame_ids: Optional[List[int]] = None


        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)


        assert model_path is not None
        self.model_path = model_path
        logger.info(f"Loading model from: {model_path}")

        if listinstr(["2.5", "2_5", "qwen25"], model_path.lower()):
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        self.model = MODEL_CLS.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",

        )
        self.model.eval()
        torch.cuda.empty_cache()


        self._clip_model = None
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
            self._clip_model = CLIPModel.from_pretrained(
                self.clip_model_name,
                use_safetensors=True
            )
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self._clip_model.eval()
            self._clip_model.to("cuda")
            logger.info("CLIP model loaded successfully with safetensors")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}. Will use timestamp fallback.")
            self._clip_model = None
            self._clip_processor = None




    def generate_inner(self, message, dataset=None):
        video_path, question = self._parse_input(message, dataset)
        if self.verbose:
            logger.info(f"[Input] video={video_path}, question={question}")




        vreader = VideoReader(video_path, ctx=cpu(0))
        fps = vreader.get_avg_fps()
        total_frames = len(vreader)
        duration = total_frames / fps


        factor = 2 if duration < 60 else 1
        num_fast = min(int(duration * 2), self.fast_max_frames)
        num_fast = min(num_fast, total_frames)

        frame_ids = (
            torch.linspace(0, total_frames - 1, num_fast)
            .round().long().tolist()
        )
        frames = vreader.get_batch(frame_ids).asnumpy()
        timestamps = [fid / fps for fid in frame_ids]

        frames_tensor = add_frame_numbers_batch(frames, timestamps)

        H, W = frames_tensor.shape[-2], frames_tensor.shape[-1]
        target_pixels = self.fast_video_resolution * 28 * 28 * factor
        resized_h, resized_w = get_video_hw(H, W, target_pixels)

        fast_video = transforms.functional.resize(
            frames_tensor.float(),
            [resized_h, resized_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        logger.info(
            f"[Stage 0] Sampled {num_fast} frames at {resized_h}x{resized_w}, "
            f"duration={duration:.1f}s, fps={fps:.2f}"
        )




        if self.clip_model is not None:
            logger.info("[Stage 0] Encoding CLIP frame features (one-time cache)...")
            self._cached_clip_frame_features = encode_clip_frames(
                frames, frame_ids,
                self.clip_model,
                self.clip_processor,
                device="cuda",
                batch_size=32,
            )
            self._cached_clip_frame_ids = frame_ids
        else:
            self._cached_clip_frame_features = None
            self._cached_clip_frame_ids = None




        stage1_result = self._stage1_evidence_check(
            fast_video, duration, fps, total_frames, question
        )
        is_sufficient = stage1_result["sufficient"]
        key_observations = stage1_result["key_observations"]
        missing_info = stage1_result["missing_info"]
        candidate_answer_step1 = stage1_result.get("candidate_answer")
        pseudo_queries = stage1_result.get("pseudo_queries", [])




        initial_answer = ""
        initial_evidence_summary = ""
        initial_clips = []
        initial_hires_frames = []

        if is_sufficient:

            logger.info("[Branch] Evidence sufficient. Path A (3 calls total: Stage 1 + Stage 23 + Stage 56).")
            if candidate_answer_step1:
                initial_answer = candidate_answer_step1
                initial_evidence_summary = key_observations
            else:
                logger.warning("No boxed answer found in Stage 1, using key_observations.")
                initial_answer = key_observations
                initial_evidence_summary = key_observations


            stage23_result = self._stage23_generate_all(
                question=question,
                lowres_video=fast_video,
                is_path_a=True,
                initial_observations=key_observations,
                candidate_answer=initial_answer,
            )
            hypothesis = stage23_result["hypothesis"]
            clue = stage23_result["clue"]
            answer_reasoning = stage23_result["answer_reasoning"]

            logger.info(f"[Stage 23] Clue: {clue[:100]}...")
        else:

            logger.info("[Branch] Evidence insufficient. Path B (3 calls total: Stage 1 + Stage 23 + Stage 56).")
            logger.info(f"[Stage 1] Generated {len(pseudo_queries)} pseudo-queries")


            initial_clips = self._locate_clips(
                frames, frame_ids, fps, pseudo_queries,
                num_clips=self.num_clips,
                clip_duration=self.clip_duration
            )
            logger.info(f"[Stage 2b] Located {len(initial_clips)} initial clips")


            initial_hires_frames, initial_clips = self._extract_hires_frames(
                vreader, fps, duration, initial_clips
            )


            stage23_result = self._stage23_generate_all(
                question=question,
                missing_info=missing_info,
                clips=initial_clips,
                hires_frames_per_clip=initial_hires_frames,
                is_path_a=False,
            )
            initial_answer = stage23_result["answer"]
            initial_evidence_summary = stage23_result.get("evidence_summary", "")
            hypothesis = stage23_result["hypothesis"]
            clue = stage23_result["clue"]
            answer_reasoning = stage23_result["answer_reasoning"]

            logger.info(f"[Stage 23] Answer generated, Clue: {clue[:100]}...")




        verification_clips, verification_hires_frames = self._stage5_retrieve(
            clue,
            vreader, fps, duration,
            raw_frames=frames,
            raw_frame_ids=frame_ids,
        )
        logger.info(f"[Stage 5] Retrieved {len(verification_clips)} verification clips")




        stage56_result = self._stage56_verify_and_integrate(
            question, initial_answer, hypothesis, clue,
            initial_evidence_summary or "",
            verification_clips, verification_hires_frames,
        )

        final_answer = stage56_result["final_answer"]
        verification_state = stage56_result["verification_state"]
        reasoning = stage56_result["reasoning"]

        logger.info(f"[Stage 56] Verification state: {verification_state}, Final answer: {final_answer}")






        if self.enable_verification_retry and verification_state == "PARTIAL":
            logger.info("[Multi-Reflection] Verification retry triggered...")

            verification_clips, verification_hires_frames = self._stage5_retrieve(
                clue,
                vreader, fps, duration,
                raw_frames=frames,
                raw_frame_ids=frame_ids,
            )

            stage56_result = self._stage56_verify_and_integrate(
                question, initial_answer, hypothesis, clue,
                initial_evidence_summary or "",
                verification_clips, verification_hires_frames,
            )
            final_answer = stage56_result["final_answer"]
            verification_state = stage56_result["verification_state"]
            reasoning = stage56_result["reasoning"]


        if self.enable_hypothesis_revision and verification_state == "NOT_VERIFIED":
            logger.info("[Multi-Reflection] Hypothesis revision triggered...")

            stage345_result = self._stage345_generate_hypothesis_clue_and_queries(
                question, initial_answer, initial_evidence_summary,
                revise=True
            )
            hypothesis = stage345_result["hypothesis"]
            clue = stage345_result["clue"]


            verification_clips, verification_hires_frames = self._stage5_retrieve(
                clue,
                vreader, fps, duration,
                raw_frames=frames,
                raw_frame_ids=frame_ids,
            )

            stage56_result = self._stage56_verify_and_integrate(
                question, initial_answer, hypothesis, clue,
                initial_evidence_summary or "",
                verification_clips, verification_hires_frames,
            )
            final_answer = stage56_result["final_answer"]
            verification_state = stage56_result["verification_state"]
            reasoning = stage56_result["reasoning"]


        logger.info(f"[Stage 56] Final answer: {final_answer}")





        return final_answer




    def _stage1_evidence_check(
        self,
        fast_video: torch.Tensor,
        duration: float,
        fps: float,
        total_frames: int,
        question: str,
    ) -> dict:
        user_text = self.STAGE1_USER.format(
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            question=question,
        )
        messages = [
            {"role": "system", "content": self.STAGE1_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Full video [0, {int(duration)}]:"},
                    {"type": "video", "video": fast_video},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        output_text = self._run_model(messages, max_new_tokens=256, temperature=0.3)


        sufficient_str = self._extract_tag(output_text, "sufficient").strip().lower()
        sufficient = sufficient_str == "yes"
        key_observations = self._extract_tag(output_text, "key_observations")
        missing_info = self._extract_tag(output_text, "missing_information")


        candidate_answer = None
        pseudo_queries = []

        if sufficient:
            candidate_answer = self._extract_boxed(output_text)
            if not candidate_answer:
                candidate_answer = key_observations
        else:

            pseudo_block = self._extract_tag(output_text, "pseudo_queries")
            for line in pseudo_block.split("\n"):
                line = line.strip()
                m = re.match(r"^\d+[\.\)]\s*(.+)", line)
                if m:
                    pseudo_queries.append(m.group(1).strip())

        return {
            "sufficient": sufficient,
            "key_observations": key_observations,
            "missing_info": missing_info,
            "candidate_answer": candidate_answer,
            "pseudo_queries": pseudo_queries,
        }








    def _stage23_generate_all(
        self,
        question: str,
        missing_info: str = "",
        clips: Optional[List[Dict]] = None,
        hires_frames_per_clip: Optional[List[torch.Tensor]] = None,
        lowres_video: Optional[torch.Tensor] = None,
        is_path_a: bool = False,
        initial_observations: str = "",
        candidate_answer: str = "",
    ) -> dict:
        if is_path_a:

            user_text = self.STAGE23_USER_PATH_A.format(
                question=question,
                initial_observations=initial_observations,
                candidate_answer=candidate_answer,
            )

            content = []
            if lowres_video is not None:
                content.append({
                    "type": "text",
                    "text": "Low-resolution overview frames:",
                })
                content.append({"type": "video", "video": lowres_video})

            content.append({"type": "text", "text": user_text})

            messages = [
                {"role": "system", "content": self.STAGE23_SYSTEM},
                {"role": "user", "content": content},
            ]

            output_text = self._run_model(messages, max_new_tokens=384, temperature=0.2)
        else:

            clip_descriptions = "\n".join(
                f"  Clip {i+1} [{c['start']:.1f}s – {c['end']:.1f}s] "
                f"(query: \"{c['query']}\", CLIP sim={c['similarity']:.3f})"
                for i, c in enumerate(clips)
            )

            user_text = self.STAGE23_USER_PATH_B.format(
                question=question,
                missing_info=missing_info,
                clip_descriptions=clip_descriptions,
            )

            content = []

            content.append({
                "type": "text",
                "text": "\nHigh-resolution evidence clips (for detailed examination):",
            })
            for i, (clip, frames_tensor) in enumerate(zip(clips, hires_frames_per_clip)):
                content.append({
                    "type": "text",
                    "text": f"\n--- Clip {i+1} [{clip['start']:.1f}s – {clip['end']:.1f}s] ---\n"
                            f"Query: \"{clip['query']}\"\n",
                })
                content.append({"type": "video", "video": frames_tensor})

            content.append({"type": "text", "text": user_text})

            messages = [
                {"role": "system", "content": self.STAGE23_SYSTEM},
                {"role": "user", "content": content},
            ]

            output_text = self._run_model(messages, max_new_tokens=384, temperature=0.2)


        if is_path_a:

            answer = candidate_answer
        else:

            answer = self._extract_tag(output_text, "candidate_answer")
            if not answer:
                answer = output_text.split("</hypothesis>")[0].strip() if "</hypothesis>" in output_text else output_text

        hypothesis = self._extract_tag(output_text, "hypothesis")
        clue = self._extract_tag(output_text, "clue")
        answer_reasoning = self._extract_tag(output_text, "answer_reasoning")


        return {
            "answer": answer,
            "hypothesis": hypothesis,
            "clue": clue,
            "answer_reasoning": answer_reasoning,
            "verification_queries": [],
            "evidence_summary": initial_observations if is_path_a else "",
        }





    def _stage345_generate_hypothesis_clue_and_queries(
        self,
        question: str,
        initial_answer: str,
        initial_evidence_summary: str,
        lowres_video: Optional[torch.Tensor] = None,
        hires_frames: Optional[List[torch.Tensor]] = None,
        revise: bool = False,
    ) -> dict:
        user_text = self.STAGE345_USER.format(
            question=question,
            candidate_answer=initial_answer,
            initial_evidence_summary=initial_evidence_summary,
        )

        content = []

        if revise:
            content.append({
                "type": "text",
                "text": "REVISED HYPOTHESIS GENERATION (Previous hypothesis was NOT verified):",
            })

        if lowres_video is not None:
            content.append({
                "type": "text",
                "text": "\nLow-resolution overview frames:",
            })
            content.append({"type": "video", "video": lowres_video})

        if hires_frames:
            content.append({
                "type": "text",
                "text": "\nHigh-resolution evidence frames:",
            })
            for i, frames_tensor in enumerate(hires_frames):
                content.append({
                    "type": "text",
                    "text": f"\n--- Evidence Clip {i+1} ---",
                })
                content.append({"type": "video", "video": frames_tensor})

        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": self.STAGE345_SYSTEM},
            {"role": "user", "content": content},
        ]

        output_text = self._run_model(messages, max_new_tokens=320, temperature=0.2)


        hypothesis = self._extract_tag(output_text, "hypothesis")


        clue = self._extract_tag(output_text, "clue")



        return {
            "hypothesis": hypothesis,
            "clue": clue,
            "verification_queries": [],
        }




    def _stage5_retrieve(
        self,
        clue: str,
        vreader: VideoReader,
        fps: float,
        duration: float,
        raw_frames: np.ndarray,
        raw_frame_ids: List[int],
    ) -> Tuple[List[Dict], List[torch.Tensor]]:

        if self.clip_model is not None and clue:

            device = next(self.clip_model.parameters()).device
            self.clip_model.eval()


            if (self._cached_clip_frame_features is not None and
                self._cached_clip_frame_ids is not None):
                frame_features = self._cached_clip_frame_features
                frame_ids = self._cached_clip_frame_ids
                logger.info("[Stage 5] Using cached CLIP frame features")
            else:
                logger.info("[Stage 5] No cache, encoding frame features")
                batch_size = 32
                frame_features_list = []
                for i in range(0, len(raw_frames), batch_size):
                    batch = [Image.fromarray(f.astype(np.uint8)) for f in raw_frames[i: i + batch_size]]
                    inputs = self.clip_processor(images=batch, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        feats = self.clip_model.get_image_features(inputs.pixel_values)
                        feats = F.normalize(feats, p=2, dim=-1)
                    frame_features_list.append(feats)
                frame_features = torch.cat(frame_features_list, dim=0)
                frame_ids = raw_frame_ids


            inputs = self.clip_processor(text=[clue], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                clue_feature = self.clip_model.get_text_features(**inputs)
                clue_feature = F.normalize(clue_feature, p=2, dim=-1)


            similarities = torch.matmul(clue_feature, frame_features.T).squeeze(0)



            clip_model = self.clip_model
            num_clips = self.verification_num_clips
            clip_duration = self.verification_clip_duration
            sim_threshold = self.clip_sim_threshold


            sorted_indices = similarities.argsort(descending=True)
            verification_clips = []
            for idx in sorted_indices[:num_clips * 2]:
                sim = similarities[idx].item()
                if sim >= sim_threshold and len(verification_clips) < num_clips:
                    center_time = frame_ids[idx] / fps
                    start_t = max(0.0, center_time - clip_duration / 2)
                    end_t = center_time + clip_duration / 2
                    verification_clips.append({
                        "start": round(start_t, 2),
                        "end": round(end_t, 2),
                        "query": clue,
                        "similarity": round(sim, 4),
                    })
                if len(verification_clips) >= num_clips:
                    break

            logger.info(f"[Stage 5 CLIP] Selected {len(verification_clips)} clips using clue (threshold={sim_threshold})")
        else:

            verification_clips = []
            step = duration / max(1, self.verification_num_clips)
            for i in range(self.verification_num_clips):
                start = i * step
                end = min(start + self.verification_clip_duration, duration)
                verification_clips.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "query": clue if clue else "",
                    "similarity": 0.0,
                })


        for c in verification_clips:
            c["end"] = min(c["end"], duration)
        verification_clips = sorted(verification_clips, key=lambda x: x["start"])


        verification_hires_frames, verification_clips = self._extract_hires_frames(
            vreader, fps, duration, verification_clips
        )

        return verification_clips, verification_hires_frames




    def _stage56_verify_and_integrate(
        self,
        question: str,
        initial_answer: str,
        hypothesis: str,
        clue: str,
        initial_evidence_summary: str,
        clips: List[Dict],
        hires_frames: List[torch.Tensor],
    ) -> dict:
        clip_descriptions = "\n".join(
            f"  Clip {i+1} [{c['start']:.1f}s – {c['end']:.1f}s] "
            f"(query: \"{c['query']}\", CLIP sim={c['similarity']:.3f})"
            for i, c in enumerate(clips)
        )

        user_text = self.STAGE56_USER.format(
            question=question,
            initial_answer=initial_answer,
            hypothesis=hypothesis,
            clue=clue,
            initial_evidence_summary=initial_evidence_summary,
        )

        content = []

        content.append({
            "type": "text",
            "text": "\nHigh-resolution verification clips:",
        })
        for i, (clip, frames_tensor) in enumerate(zip(clips, hires_frames)):
            content.append({
                "type": "text",
                "text": f"\n--- Verification Clip {i+1} [{clip['start']:.1f}s – {clip['end']:.1f}s] ---\n"
                        f"Query: \"{clip['query']}\"\n",
            })
            content.append({"type": "video", "video": frames_tensor})

        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": self.STAGE56_SYSTEM},
            {"role": "user", "content": content},
        ]

        output_text = self._run_model(messages, max_new_tokens=384, temperature=0.1)


        verification_state = self._extract_tag(output_text, "verification_state")
        decision_reason = self._extract_tag(output_text, "decision_reason")
        final_answer_tag = self._extract_tag(output_text, "final_answer")


        boxed_answer = self._extract_boxed(output_text)
        if boxed_answer:
            final_answer = boxed_answer
        elif final_answer_tag:
            final_answer = final_answer_tag
        else:
            final_answer = initial_answer

        return {
            "final_answer": final_answer,
            "verification_state": verification_state,
            "reasoning": decision_reason,
        }




    def _locate_clips(
        self,
        frames: np.ndarray,
        frame_ids: List[int],
        fps: float,
        queries: List[str],
        num_clips: int = 3,
        clip_duration: float = 30.0,
    ) -> List[Dict]:
        if self.clip_model is not None:

            if (self._cached_clip_frame_features is not None and
                self._cached_clip_frame_ids is not None):
                logger.info("[Stage 2b] Using cached CLIP frame features")
                clips = select_clips_from_cached_features(
                    cached_frame_features=self._cached_clip_frame_features,
                    cached_frame_ids=self._cached_clip_frame_ids,
                    fps=fps,
                    queries=queries,
                    clip_model=self.clip_model,
                    clip_processor=self.clip_processor,
                    num_clips=num_clips,
                    clip_duration=clip_duration,
                    sim_threshold=self.clip_sim_threshold,
                )
            else:
                logger.info("[Stage 2b] No cache, encoding frame features")
                clips = cosine_select_clips(
                    frames=frames,
                    frame_ids=frame_ids,
                    fps=fps,
                    queries=queries,
                    clip_model=self.clip_model,
                    clip_processor=self.clip_processor,
                    num_clips=num_clips,
                    clip_duration=clip_duration,
                    sim_threshold=self.clip_sim_threshold,
                )
        else:
            logger.warning("CLIP unavailable – using uniform clip fallback")
            duration = frame_ids[-1] / fps if frame_ids else 30.0
            clips = []
            step = duration / max(1, num_clips)
            for i in range(num_clips):
                start = i * step
                end = min(start + clip_duration, duration)
                clips.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "query": queries[i] if i < len(queries) else "",
                    "similarity": 0.0,
                })

        return clips

    def _extract_hires_frames(
        self,
        vreader: VideoReader,
        fps: float,
        duration: float,
        clips: List[Dict],
    ) -> Tuple[List[torch.Tensor], List[Dict]]:

        sorted_clips = sorted(clips, key=lambda x: x["start"])

        factor = 2 if duration < 60 else 1
        result = []
        total_frames = len(vreader)


        num_clips = len(sorted_clips)
        if num_clips == 0:
            return result, sorted_clips


        clip_durations = [c["end"] - c["start"] for c in sorted_clips]
        total_duration = sum(clip_durations)
        max_total_frames = self.slow_max_frames


        frames_per_clip = []
        for dur in clip_durations:
            if total_duration > 0:
                proportion = dur / total_duration
                num_f = max(2, int(proportion * max_total_frames))
            else:
                num_f = 2
            frames_per_clip.append(num_f)


        current_total = sum(frames_per_clip)
        if current_total > max_total_frames:
            scale = max_total_frames / current_total
            frames_per_clip = [max(2, int(f * scale)) for f in frames_per_clip]

        for clip, num_f in zip(sorted_clips, frames_per_clip):
            start_f = max(0, int(clip["start"] * fps))
            end_f = min(total_frames - 1, int(clip["end"] * fps))

            frames, frame_ids = sample_frames_uniform(vreader, num_f, start_f, end_f)
            timestamps = [fid / fps for fid in frame_ids]

            tensor = add_frame_numbers_batch(frames, timestamps)

            H, W = tensor.shape[-2], tensor.shape[-1]
            target_pixels = self.slow_video_resolution * 28 * 28 * factor
            resized_h, resized_w = get_video_hw(H, W, target_pixels)

            tensor = transforms.functional.resize(
                tensor.float(),
                [resized_h, resized_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

            result.append(tensor)

        actual_total_frames = sum([t.shape[0] for t in result])
        logger.info(f"[Hires Frames] Extracted {actual_total_frames} total frames from {num_clips} clips (max: {max_total_frames})")
        return result, sorted_clips

    def _parse_numbered_list(self, text: str) -> List[str]:
        items = []
        for line in text.split("\n"):
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                items.append(m.group(1).strip())
        return items




    def _run_model(
        self,
        messages: list,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
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

        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
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

        images = []
        videos = []
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




    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_boxed(text: str) -> str:

        m = re.search(r"\\boxed\{\{(.*?)\}\}", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
        return m.group(1).strip() if m else ""

