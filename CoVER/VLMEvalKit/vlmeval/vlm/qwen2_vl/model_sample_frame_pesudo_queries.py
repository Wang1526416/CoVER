
from __future__ import annotations

import os
import re
import json
import math
import logging
import warnings
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


    best_sims, best_frame_idxs = sim_matrix.max(dim=1)


    order = best_sims.argsort(descending=True)
    clips = []
    used_centers: List[float] = []

    for qi in order:
        if len(clips) >= num_clips:
            break
        sim_val = best_sims[qi].item()
        if sim_val < sim_threshold:
            logger.debug(f"[CLIP TCS] query {qi} skipped, similarity {sim_val:.4f} < threshold {sim_threshold}")
            continue

        fi = best_frame_idxs[qi].item()
        center_time = frame_ids[fi] / fps


        if any(abs(center_time - c) < clip_duration * 0.5 for c in used_centers):

            continue

        start_t = max(0.0, center_time - clip_duration / 2)
        end_t = center_time + clip_duration / 2
        clips.append({
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "query": queries[qi],
            "similarity": round(sim_val, 4),
        })
        used_centers.append(center_time)

    logger.info(f"[CLIP] Selected {len(clips)} clips from {len(queries)} queries")
    print(f"[clips] {clips}")
    return clips






class Qwen2VLChatSampleFramePesudoQueries(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True






    STEP1_SYSTEM = (
        "You are an expert video analyst. "
        "Your task is to determine if the visual evidence from low-resolution frames "
        "is sufficient to confidently answer the given question."
    )

    STEP1_USER = """\
The following are uniformly-sampled frames from the entire video (low resolution overview).
Each frame is labeled with its timestamp in seconds.

Video metadata:
- Duration   : {duration:.1f} s
- FPS        : {fps:.2f}
- Total frames: {total_frames}

Question: {question}

─────────────────────────────────────────
Instructions:
1. Watch all frames carefully.
2. Determine if the visual evidence is SUFFICIENT to produce a confident candidate answer.

IMPORTANT: You should only output "sufficient: yes" if:
- You can clearly see the key objects, actions, or events relevant to the question
- The visual details are clear enough to distinguish between different options
- You have enough temporal information to answer sequence-related questions

If the frames are too blurry, too few, or missing critical moments, output "sufficient: no".

Please respond in the following format EXACTLY:

<evidence_check>
<sufficient>yes or no</sufficient>
<key_observations>
[Bullet-point list of what you CAN clearly observe from these frames]
</key_observations>
<missing_information>
[Bullet-point list of specific visual information you CANNOT determine.
 Write "none" if evidence is fully sufficient.]
</missing_information>
</evidence_check>

IF AND ONLY IF the evidence is sufficient (sufficient: yes), you MUST also provide:
- A candidate answer to the question
- For multiple-choice questions, select ONE specific option (A, B, C, or D)
- Format your answer as: \\boxed{{your answer here}}

Example for sufficient case:
<evidence_check>
<sufficient>yes</sufficient>
<key_observations>
- The person is wearing a red shirt
- They are holding a blue cup in their right hand
- The action takes place in a kitchen
</key_observations>
<missing_information>none</missing_information>
</evidence_check>

Final_answer:\\boxed{{B}}
"""


    STEP2A_SYSTEM = (
        "You are a video retrieval expert. "
        "Generate targeted queries to help locate missing visual evidence."
    )

    STEP2A_USER = """\
Based on the question and the missing information identified, generate targeted pseudo-queries
to help locate the missing visual evidence in the video.

Question: {question}

Missing information:
{missing_info}

Requirements:
- Generate 3 to 5 pseudo-queries.
- Each pseudo-query MUST be written as a direct English question ending with a question mark.
- Focus only on observable visual evidence, such as:
  - object presence or absence
  - object state or appearance
  - human action or motion
  - hand-object interaction
  - spatial relationship
  - before/after event change
- Do NOT use vague meta-questions like "what is happening?" or "why is the person doing this?"
- Do NOT combine multiple checks into one question.
- Make the questions specific enough to support zoom-in retrieval.

Respond EXACTLY in this format:

<pseudo_queries>
1. [question]
2. [question]
3. [question]
...
</pseudo_queries>
"""


    STEP2B_SYSTEM = (
        "You are an expert video analyst. "
        "Based on the high-resolution evidence frames, provide a candidate answer to the question."
    )

    STEP2B_USER = """\
You are now given HIGH-RESOLUTION frames from the most relevant video clips,
selected specifically to address the missing visual information.

Question: {question}

Missing information that prompted retrieval:
{missing_info}

The high-resolution clips below are ordered by relevance:
{clip_descriptions}

Please carefully examine the high-resolution frames above and provide your candidate answer.

Requirements:
- Base your answer ONLY on visible evidence from the high-resolution frames.
- If the question is multiple-choice, identify the best-supported option.
- Provide a clear, concise answer.

Respond EXACTLY in this format:

<candidate_answer>
[Your answer based on the high-resolution evidence]
</candidate_answer>

<evidence_summary>
[Brief description of the key visual evidence that supports your answer]
</evidence_summary>
"""


    STEP3_SYSTEM = (
        "You are a critical reasoning expert. "
        "Your task is to reflect on a candidate answer and determine if it is correct and confident."
    )

    STEP3_USER = """\
You are given a question, a candidate answer, and visual evidence from the video.
Your task is to reflect critically on whether the candidate answer is correct and confident.

Question: {question}

Candidate Answer: {candidate_answer}

Visual Evidence Summary:
{evidence_summary}

─────────────────────────────────────────
Reflection Instructions:

1. Evidence Verification:
   - Does the visual evidence actually support the candidate answer?
   - Are there any contradictions or ambiguities in the evidence?
   - Identify specific visual details that either support or contradict the answer.

2. Alternative Consideration:
   - Could other options also be plausible given the evidence?
   - Is there evidence that clearly rules out competing options?
   - If multiple-choice, explain why the chosen option is more likely than others.

3. Confidence Assessment:
   - Is this a high-confidence answer that is clearly supported by strong visual evidence?
   - Or is there remaining uncertainty that suggests the answer might need revision?

4. Decision Rationale (CRITICAL):
   - IF you decide to KEEP the candidate answer:
     * Explain WHY the visual evidence strongly supports this answer
     * Explain WHY there is no significant evidence suggesting a different answer
     * Be specific about the visual evidence that confirms this choice

   - IF you decide to REJECT the candidate answer:
     * Explain WHY the visual evidence contradicts or fails to support this answer
     * Explain WHAT visual evidence suggests a different answer instead
     * Explain WHY the alternative answer is better supported

5. FINAL ANSWER REQUIREMENT (CRITICAL):
   - You MUST provide a definite answer. Do NOT say "cannot determine", "unclear", "need more information", etc.
   - If this is a multiple-choice question, you MUST select ONE specific option (e.g., "(A)", "(B)", "(C)", "(D)") that you are most confident about.
   - Your answer MUST be in the format \\boxed{{answer}} where answer is your definite choice.

Please respond in the following format EXACTLY:

<reflection>
<is_high_confidence>yes or no</is_high_confidence>
<evidence_support>
[Explain how the evidence supports or contradicts the candidate answer. Be specific about what you see in the video.]
</evidence_support>
<competing_options>
[If multiple-choice: explain why other options are less likely. If not multiple-choice: write "N/A"]
</competing_options>
<decision_rationale>
[CRITICAL: Explain your decision in detail.

If you ARE keeping the candidate answer, write:
"I am keeping the current answer because [explain why the visual evidence strongly supports it]. The evidence shows [specific visual details], and there is no significant evidence suggesting a different answer."

If you ARE NOT keeping the candidate answer, write:
"I am not using the current answer because [explain why the evidence contradicts it]. Instead, I choose [alternative answer] because [explain why the visual evidence supports this alternative better]."]
</decision_rationale>
<remaining_uncertainty>
[Any remaining doubts or missing evidence, or write "none" if fully confident]
</remaining_uncertainty>
</reflection>

<final_answer>
[Provide your final definite answer in the format: \\boxed{{(A) exact answer text}} for multiple-choice questions, or \\boxed{{exact answer}} for other questions. You MUST choose ONE specific option if multiple-choice.]
</final_answer>"""


    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
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
            attn_implementation="flash_attention_2",
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




        step1_result = self._step1_evidence_check(
            vreader, fps, total_frames, duration, question
        )
        is_sufficient = step1_result["sufficient"]
        key_observations = step1_result["key_observations"]
        missing_info = step1_result["missing_info"]
        candidate_answer_step1 = step1_result.get("candidate_answer")
        raw_frames_np = step1_result.get("raw_frames_np")
        raw_frame_ids = step1_result.get("raw_frame_ids")
        fast_video = step1_result.get("fast_video")

        if self.verbose:
            logger.info(f"[Step 1] Sufficient: {is_sufficient}")
            logger.info(f"[Step 1] Missing info: {missing_info}")




        candidate_answer = ""
        evidence_summary = ""
        clips = []
        hires_frames_per_clip = []
        pseudo_queries = []
        is_high_conf = None
        decision_rationale = None




        if is_sufficient:

            logger.info("[Branch] Evidence sufficient. Using extracted boxed answer.")
            if candidate_answer_step1:
                candidate_answer = candidate_answer_step1
                evidence_summary = key_observations
            else:

                logger.warning("No boxed answer found in Step 1, using key_observations.")
                candidate_answer = key_observations
                evidence_summary = key_observations
        else:

            logger.info("[Branch] Evidence insufficient. Starting targeted retrieval...")


            pseudo_queries = self._step2a_generate_pseudo_queries(
                question, missing_info,
                lowres_video=fast_video
            )
            if self.verbose:
                logger.info(f"[Step 2a] Pseudo-queries: {pseudo_queries}")


            clips = self._locate_clips(
                vreader, fps, total_frames, pseudo_queries,
                search_frames=raw_frames_np,
                search_frame_ids=raw_frame_ids,
            )

            if self.verbose:
                logger.info(f"[Step 2b] Located clips: {clips}")


            hires_frames_per_clip = self._extract_hires_frames(
                vreader, fps, duration, clips
            )


            candidate_answer, evidence_summary = self._step2d_generate_candidate_answer(
                question, missing_info, clips, hires_frames_per_clip,
                lowres_video=fast_video
            )
            if self.verbose:
                logger.info(f"[Step 2d] Candidate answer: {candidate_answer}")





        final_answer, is_high_conf, decision_rationale = self._step3_reflect(
            question, candidate_answer, evidence_summary,
            lowres_video=fast_video,
            hires_frames_per_clip=hires_frames_per_clip if hires_frames_per_clip else None,
            clips=clips if clips else None,
        )


        return final_answer




    def _step1_evidence_check(
        self,
        vreader: VideoReader,
        fps: float,
        total_frames: int,
        duration: float,
        question: str,
    ) -> dict:
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
            f"[Step 1] Sampled {num_fast} frames at {resized_h}x{resized_w} "
            f"(target_pixels={target_pixels}, factor={factor})"
        )

        user_text = self.STEP1_USER.format(
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            question=question,
        )
        messages = [
            {"role": "system", "content": self.STEP1_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Full video [0, {int(duration)}]:"},
                    {"type": "video", "video": fast_video},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        output_text = self._run_model(messages, max_new_tokens=1024, temperature=0.3)


        sufficient_str = self._extract_tag(output_text, "sufficient").strip().lower()
        sufficient = sufficient_str == "yes"
        key_observations = self._extract_tag(output_text, "key_observations")
        missing_info = self._extract_tag(output_text, "missing_information")


        candidate_answer = None
        if sufficient:
            candidate_answer = self._extract_boxed(output_text)
            if not candidate_answer:

                candidate_answer = key_observations

        return {
            "sufficient": sufficient,
            "key_observations": key_observations,
            "missing_info": missing_info,
            "candidate_answer": candidate_answer,
            "raw_frames_np": frames,
            "raw_frame_ids": frame_ids,
            "fast_video": fast_video,
            "num_fast_frames": num_fast,
        }




    def _step2a_generate_pseudo_queries(
        self,
        question: str,
        missing_info: str,
        lowres_video: Optional[torch.Tensor] = None,
    ) -> List[str]:
        user_text = self.STEP2A_USER.format(
            question=question,
            missing_info=missing_info,
        )


        content = []


        if lowres_video is not None:
            content.append({
                "type": "text",
                "text": "Low-resolution overview frames from the video:",
            })
            content.append({"type": "video", "video": lowres_video})


        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": self.STEP2A_SYSTEM},
            {"role": "user", "content": content},
        ]
        output_text = self._run_model(messages, max_new_tokens=512, temperature=0.5)

        block = self._extract_tag(output_text, "pseudo_queries")
        queries = []
        for line in block.split("\n"):
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                queries.append(m.group(1).strip())

        if not queries:
            for line in missing_info.split("\n"):
                line = re.sub(r"^[-*•]\s*", "", line).strip()
                if line:
                    queries.append(line)

        logger.info(f"Generated {len(queries)} pseudo-queries")
        return queries[:5]




    def _locate_clips(
        self,
        vreader: VideoReader,
        fps: float,
        total_frames: int,
        queries: List[str],
        search_frames: Optional[np.ndarray] = None,
        search_frame_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        duration = total_frames / fps

        if search_frames is not None and search_frame_ids is not None:
            frames = search_frames
            frame_ids = search_frame_ids
            logger.info(f"[CLIP] Reusing {len(frames)} frames from Step 1 for CLIP search")
        else:
            n_search = min(self.fast_max_frames, total_frames)
            frames, frame_ids = sample_frames_uniform(vreader, n_search)

        if self.clip_model is not None:
            clips = cosine_select_clips(
                frames=frames,
                frame_ids=frame_ids,
                fps=fps,
                queries=queries,
                clip_model=self.clip_model,
                clip_processor=self.clip_processor,
                num_clips=self.num_clips,
                clip_duration=self.clip_duration,
                sim_threshold=self.clip_sim_threshold,
            )
        else:
            logger.warning("CLIP unavailable – using uniform clip fallback")
            clips = []
            step = duration / max(1, self.num_clips)
            for i in range(self.num_clips):
                start = i * step
                end = min(start + self.clip_duration, duration)
                clips.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "query": queries[i] if i < len(queries) else "",
                    "similarity": 0.0,
                })

        for c in clips:
            c["end"] = min(c["end"], duration)

        return clips




    def _extract_hires_frames(
        self,
        vreader: VideoReader,
        fps: float,
        duration: float,
        clips: List[Dict],
    ) -> List[torch.Tensor]:
        factor = 2 if duration < 60 else 1
        result = []
        total_frames = len(vreader)

        for clip in clips:
            start_f = max(0, int(clip["start"] * fps))
            end_f = min(total_frames - 1, int(clip["end"] * fps))
            clip_dur = clip["end"] - clip["start"]

            num_f = min(int(clip_dur * 2), self.slow_max_frames)
            num_f = max(num_f, 2)

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

            logger.info(
                f"[Step 2c] Clip [{clip['start']:.1f}s-{clip['end']:.1f}s]: "
                f"{num_f} frames at {resized_h}x{resized_w}"
            )
            result.append(tensor)

        return result




    def _step2d_generate_candidate_answer(
        self,
        question: str,
        missing_info: str,
        clips: List[Dict],
        hires_frames_per_clip: List[torch.Tensor],
        lowres_video: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str]:
        clip_descriptions = "\n".join(
            f"  Clip {i+1} [{c['start']:.1f}s – {c['end']:.1f}s] "
            f"(query: \"{c['query']}\", CLIP sim={c['similarity']:.3f})"
            for i, c in enumerate(clips)
        )

        user_text = self.STEP2B_USER.format(
            question=question,
            missing_info=missing_info,
            clip_descriptions=clip_descriptions,
        )

        content = []


        if lowres_video is not None:
            content.append({
                "type": "text",
                "text": "Low-resolution overview frames (for context):",
            })
            content.append({"type": "video", "video": lowres_video})


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
            {"role": "system", "content": self.STEP2B_SYSTEM},
            {"role": "user", "content": content},
        ]

        output_text = self._run_model(messages, max_new_tokens=1024, temperature=0.2)

        candidate_answer = self._extract_tag(output_text, "candidate_answer")
        evidence_summary = self._extract_tag(output_text, "evidence_summary")

        return candidate_answer or output_text, evidence_summary




    def _step3_reflect(
        self,
        question: str,
        candidate_answer: str,
        evidence_summary: str,
        lowres_video: Optional[torch.Tensor] = None,
        hires_frames_per_clip: Optional[List[torch.Tensor]] = None,
        clips: Optional[List[Dict]] = None,
    ) -> Tuple[str, str, str]:
        user_text = self.STEP3_USER.format(
            question=question,
            candidate_answer=candidate_answer,
            evidence_summary=evidence_summary,
        )


        content = []


        if lowres_video is not None:
            content.append({
                "type": "text",
                "text": "Low-resolution overview frames:",
            })
            content.append({"type": "video", "video": lowres_video})


        if hires_frames_per_clip and clips:
            content.append({
                "type": "text",
                "text": "\nHigh-resolution evidence clips:",
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
            {"role": "system", "content": self.STEP3_SYSTEM},
            {"role": "user", "content": content},
        ]

        output_text = self._run_model(messages, max_new_tokens=1024, temperature=0.1)
        print(f"[question] {question}")
        print(f"[candidate_answer] {candidate_answer}")
        print(f"[evidence_summary] {evidence_summary}")
        print(f"[reflection] {output_text}")


        boxed_answer = self._extract_boxed(output_text)
        if boxed_answer:
            final_answer = boxed_answer
        else:

            final_answer_tag = self._extract_tag(output_text, "final_answer")
            final_answer = final_answer_tag if final_answer_tag else candidate_answer

        is_high_conf = self._extract_tag(output_text, "is_high_confidence")
        decision_rationale = self._extract_tag(output_text, "decision_rationale")

        print(f"[Reflect] High confidence: {is_high_conf}")
        print(f"[Reflect] Boxed answer: {boxed_answer}")
        print(f"[Reflect] Final answer: {final_answer}")
        print(f"[Reflect] Decision rationale: {decision_rationale}")

        return final_answer, is_high_conf, decision_rationale




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


