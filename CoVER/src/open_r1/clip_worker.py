#!/usr/bin/env python











import os
import sys
import traceback
import uuid
import multiprocessing as mp
import torch
import torch.nn.functional as F
from typing import List, Dict
from PIL import Image
from decord import VideoReader, cpu
from transformers import CLIPModel, CLIPProcessor





CLIP_MODEL_PATH = "./Model/Clip-14"
NUM_FAST_FRAMES = 768






def build_uniform_fallback(total_frames, fps, queries, n_clips, clip_duration):
    clips = []
    step = (total_frames / fps) / max(1, n_clips)
    for i in range(n_clips):
        start_t = i * step
        end_t = min(start_t + clip_duration, total_frames / fps)
        clips.append({
            "start": round(start_t, 2),
            "end": round(end_t, 2),
            "query": queries[min(i, len(queries) - 1)] if queries else "",
            "similarity": 0.0,
        })
    return clips


def encode_frames(clip_model, clip_processor, raw_frames, device, batch_size=32):
    model_dtype = next(clip_model.parameters()).dtype
    features = []
    for i in range(0, len(raw_frames), batch_size):
        batch = [Image.fromarray(f.astype("uint8")) for f in raw_frames[i:i + batch_size]]
        inp = clip_processor(images=batch, return_tensors="pt", padding=True)
        inp = {k: (v.to(device=device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
               for k, v in inp.items()}
        with torch.inference_mode():
            feats = clip_model.get_image_features(inp["pixel_values"])
            feats = F.normalize(feats.float(), p=2, dim=-1)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)


def encode_text(clip_model, clip_processor, text, device):
    model_dtype = next(clip_model.parameters()).dtype
    inp = clip_processor(
        text=[text], return_tensors="pt", padding=True,
        truncation=True, max_length=77,
    )
    inp = {k: (v.to(device=device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
           for k, v in inp.items()}
    with torch.inference_mode():
        feat = clip_model.get_text_features(**inp)
        feat = F.normalize(feat.float(), p=2, dim=-1)
    return feat.cpu()


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

    print(f"[CLIP Worker][TCS] peaks={len(peaks)} → merged={len(merged_clips)} → top={len(clips)} threshold={threshold:.4f}")
    return clips







def retrieve_zoom_clips(
    clip_model,
    clip_processor,
    frame_features: torch.Tensor,
    frame_ids: List[int],
    fps: float,
    queries: List[str],
    question: str,
    n_clips: int,
    clip_duration: float,
    sim_threshold: float,
    question_weight: float = 0.7,
    device: str = "cuda",
) -> List[Dict]:
    if not queries:
        return []


    pq_feats = torch.cat([encode_text(clip_model, clip_processor, q, device) for q in queries], dim=0)
    sim_pq = torch.matmul(pq_feats, frame_features.T).mean(dim=0, keepdim=True)


    if question.strip():
        q_feat = encode_text(clip_model, clip_processor, question, device)
        sim_q  = torch.matmul(q_feat, frame_features.T)
        sim_weighted = question_weight * sim_q + (1.0 - question_weight) * sim_pq
        print(f"[CLIP Worker][Zoom] question_weight={question_weight}, pq_weight={1.0-question_weight:.2f}")
    else:
        sim_weighted = sim_pq

    clips = tcs_select_clips_with_gaussian(
        sim_matrix=sim_weighted,
        frame_ids=frame_ids,
        fps=fps,
        num_clips=n_clips,
        clip_duration=clip_duration,
        alpha=0.5,
        smooth_radius=4,
        smooth_sigma=1.0,
    )
    clips = [c for c in clips if c["similarity"] >= sim_threshold]
    print(f"[CLIP Worker][Zoom] Final clips after threshold={sim_threshold}: {len(clips)}")
    return clips







def retrieve_verify_clips(
    clip_model,
    clip_processor,
    frame_features: torch.Tensor,
    frame_ids: List[int],
    fps: float,
    clue: str,
    n_clips: int,
    clip_duration: float,
    sim_threshold: float,
    device: str = "cuda",
) -> List[Dict]:
    if not clue:
        return []

    clue_feat = encode_text(clip_model, clip_processor, clue, device)
    similarities = torch.matmul(clue_feat, frame_features.T).squeeze(0)


    sorted_indices = similarities.argsort(descending=True)
    candidate_clips = []
    for idx in sorted_indices[: n_clips * 5]:
        sim = similarities[idx].item()
        if sim >= sim_threshold:
            center_time = frame_ids[idx] / fps
            start_t = max(0.0, center_time - clip_duration / 2)
            end_t   = center_time + clip_duration / 2
            candidate_clips.append({
                "start":      round(start_t, 2),
                "end":        round(end_t, 2),
                "query":      clue,
                "similarity": round(sim, 4),
            })


    candidate_clips.sort(key=lambda x: x["start"])
    merged_clips = []
    if candidate_clips:
        cur_start = candidate_clips[0]["start"]
        cur_end   = candidate_clips[0]["end"]
        cur_sims  = [candidate_clips[0]["similarity"]]

        for clip in candidate_clips[1:]:
            if clip["start"] <= cur_end:
                cur_end = max(cur_end, clip["end"])
                cur_sims.append(clip["similarity"])
            else:
                merged_clips.append({
                    "start":      round(cur_start, 2),
                    "end":        round(cur_end, 2),
                    "query":      clue,
                    "similarity": round(sum(cur_sims) / len(cur_sims), 4),
                })
                cur_start = clip["start"]
                cur_end   = clip["end"]
                cur_sims  = [clip["similarity"]]

        merged_clips.append({
            "start":      round(cur_start, 2),
            "end":        round(cur_end, 2),
            "query":      clue,
            "similarity": round(sum(cur_sims) / len(cur_sims), 4),
        })


    merged_clips.sort(key=lambda x: x["similarity"], reverse=True)
    clips = merged_clips[:n_clips]

    print(f"[CLIP Worker][Verify] candidates={len(candidate_clips)} → merged={len(merged_clips)} → top={len(clips)} threshold={sim_threshold}")
    return clips






def retrieve_with_clip(
    clip_model,
    clip_processor,
    video_path: str,
    queries: List[str],
    n_clips: int,
    clip_duration: float,
    mode: str = "zoom",
    question: str = "",
    sim_threshold: float = 0.15,
    num_fast_frames: int = NUM_FAST_FRAMES,
) -> Dict:
    device = next(clip_model.parameters()).device
    print(f"[CLIP Worker] mode={mode} video={video_path} queries={queries} n_clips={n_clips} clip_duration={clip_duration}")

    vreader     = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vreader)
    fps          = float(vreader.get_avg_fps())
    duration     = total_frames / fps

    num_frames_global = min(int(duration * 2), num_fast_frames)
    fast_frame_ids = torch.linspace(0, total_frames - 1, num_frames_global).round().long().tolist()
    print(f"[CLIP Worker] Sampling {len(fast_frame_ids)} frames, dur={duration:.1f}s fps={fps:.2f}")

    raw_frames = vreader.get_batch(fast_frame_ids).asnumpy()
    frame_features = encode_frames(clip_model, clip_processor, raw_frames, device)
    print(f"[CLIP Worker] Frame features: {frame_features.shape}")

    if mode == "zoom":
        clips = retrieve_zoom_clips(
            clip_model=clip_model, clip_processor=clip_processor,
            frame_features=frame_features, frame_ids=fast_frame_ids,
            fps=fps, queries=queries, question=question,
            n_clips=n_clips, clip_duration=clip_duration,
            sim_threshold=sim_threshold, device=str(device),
        )
    else:
        clue = queries[0] if queries else ""
        clips = retrieve_verify_clips(
            clip_model=clip_model, clip_processor=clip_processor,
            frame_features=frame_features, frame_ids=fast_frame_ids,
            fps=fps, clue=clue,
            n_clips=n_clips, clip_duration=clip_duration,
            sim_threshold=sim_threshold, device=str(device),
        )


    if not clips:
        print(f"[CLIP Worker] No clips found, falling back to uniform sampling")
        clips = build_uniform_fallback(total_frames, fps, queries, n_clips, clip_duration)

    del frame_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[CLIP Worker] Returning {len(clips)} clips")
    return {"clips": clips, "fps": fps, "total_frames": total_frames}






def worker_main(request_queue: mp.Queue, response_queue: mp.Queue, clip_path: str, device_id: int = 0):
    import time
    start_time = time.time()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP Worker] Starting, pid={os.getpid()}, device={device}")

    load_start = time.time()
    clip_model = CLIPModel.from_pretrained(
        clip_path, use_safetensors=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    clip_processor = CLIPProcessor.from_pretrained(clip_path)
    clip_model.eval()
    clip_model.to(device)
    print(f"[CLIP Worker] Model loaded in {time.time()-load_start:.2f}s, device={device}")

    w = clip_model.vision_model.embeddings.patch_embedding.weight
    assert w.ndim == 4 and w.shape[0] > 0
    print(f"[CLIP Worker] patch_embedding shape={w.shape}")

    response_queue.put({"ready": True, "pid": os.getpid()})

    request_count = 0
    while True:
        msg = request_queue.get()
        if msg is None:
            break

        request_count += 1
        req_id = msg["req_id"]
        print(f"[CLIP Worker] Request #{request_count}: req_id={req_id} mode={msg.get('mode','zoom')}")

        try:
            result = retrieve_with_clip(
                clip_model=clip_model,
                clip_processor=clip_processor,
                video_path=msg["video_path"],
                queries=msg["queries"],
                n_clips=msg["n_clips"],
                clip_duration=msg["clip_duration"],
                mode=msg.get("mode", "zoom"),
                question=msg.get("question", ""),
                sim_threshold=msg.get("sim_threshold", 0.15),
                num_fast_frames=msg.get("num_fast_frames", NUM_FAST_FRAMES),
            )
            response_queue.put({"req_id": req_id, "ok": True, "result": result})
        except Exception as e:
            print(f"[CLIP Worker] ERROR: {e}")
            response_queue.put({
                "req_id": req_id, "ok": False,
                "error": str(e), "traceback": traceback.format_exc(),
            })

    print(f"[CLIP Worker] Stopped, total requests={request_count}, time={time.time()-start_time:.1f}s")
