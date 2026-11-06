import os
import sys
import platform

# 1. 自动处理 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
import csv
import re
import hashlib
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import librosa
import cv2
from PIL import Image
import torchvision.transforms as T
from transformers import WhisperProcessor, WhisperModel, Wav2Vec2Processor, Wav2Vec2Model, AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm

# Import original components
try:
    from school_binary.audio_dataset import SchoolAudioDataset, SchoolVideoDataset
    from school_binary.config import get_school_paths
    from video_binary.video import CLIPVideoFeatureExtractor
    from video_binary.audio import _read_wav, get_audio_segment_indices
    from video_binary.train import (
        _setup_distributed, 
        _dist_is_initialized, 
        _all_reduce_mean, 
        _all_gather_concat_cpu,
        QuestionAttentionBinary,
        extract_clip_tokens,
        extract_whisper_tokens,
        extract_wav2vec_tokens,
        set_seed,
        _epoch_auc,
        _infer_patch_size_from_vision_model,
        _patch_background_mask_from_frames,
    )
except ImportError:
    from audio_dataset import SchoolAudioDataset, SchoolVideoDataset
    from config import get_school_paths
    from video import CLIPVideoFeatureExtractor
    from video_binary.audio import _read_wav, get_audio_segment_indices
    from train import (
        _setup_distributed, 
        _dist_is_initialized, 
        _all_reduce_mean, 
        _all_gather_concat_cpu,
        QuestionAttentionBinary,
        extract_clip_tokens,
        extract_whisper_tokens,
        extract_wav2vec_tokens,
        set_seed,
        _epoch_auc,
        _infer_patch_size_from_vision_model,
        _patch_background_mask_from_frames,
    )

# 强制设置共享策略
try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

def _silence_ratio(wav: np.ndarray, thr: float) -> float:
    if wav is None:
        return 1.0
    w = np.asarray(wav).reshape(-1)
    if w.size == 0:
        return 1.0
    return float((np.abs(w) < float(thr)).mean())

class MultimodalSSLDataset(Dataset):
    """
    多模态自监督学习数据集：同时加载同一问题的音频和视频。
    """
    def __init__(
        self,
        audio_root,
        video_root,
        label_path,
        allowed_ids_path=None,
        split=None,
        num_frames=4,
        sample_rate=16000,
        seconds=8,
        mask_background=False,
        face_expand=0.25,
        rotate180: bool = True,
        segment_threshold: float = 0.005,
        use_highpass: bool = True,
        patch_mask_background: bool = False,
        patch_mask_var_thr: float = 1e-6,
        reliability_mode: str = "none",
        asr_cache_path: str = "",
        asr_logprob_lo: float = -3.0,
        asr_logprob_hi: float = -0.5,
        silence_thr: float = 0.01,
        prefer_cache_silence: bool = True,
        cache_frames: bool = True,
        frame_cache_dir: str = "",
    ):
        self.audio_ds = SchoolAudioDataset(
            audio_root=audio_root,
            label_path=label_path,
            allowed_ids_path=allowed_ids_path,
            split=split,
            sample_rate=sample_rate,
            seconds=seconds,
            return_segment_mask=False,
            segment_threshold=segment_threshold,
            use_highpass=use_highpass,
        )
        self.video_ds = SchoolVideoDataset(
            video_root=video_root,
            label_path=label_path,
            allowed_ids_path=allowed_ids_path,
            split=split,
            num_frames=num_frames,
            mask_background=mask_background,
            face_expand=face_expand,
            rotate180=rotate180,
        )
        self.sample_rate = int(sample_rate)
        self.seconds = int(seconds)
        self.target_len = int(sample_rate) * int(seconds)
        self.use_highpass = bool(use_highpass)
        self.segment_threshold = float(segment_threshold)
        self.patch_mask_background = bool(patch_mask_background)
        self.patch_mask_var_thr = float(patch_mask_var_thr)
        self.reliability_mode = str(reliability_mode)
        self.asr_cache_path = str(asr_cache_path or "")
        self.asr_logprob_lo = float(asr_logprob_lo)
        self.asr_logprob_hi = float(asr_logprob_hi)
        self.silence_thr = float(silence_thr)
        self.prefer_cache_silence = bool(prefer_cache_silence)
        self.cache_frames = bool(cache_frames)
        if frame_cache_dir:
            self.frame_cache_dir = str(frame_cache_dir)
        else:
            self.frame_cache_dir = os.path.join(str(video_root), "_cached_frames")
        self._asr_avg_logprob = {}
        self._asr_silence_ratio = {}
        
        # 对齐样本：按学生对齐（Q=3），保证聚合方式能与 video_binary/train.py 一致
        self.samples = []
        audio_map = {sid: paths for sid, paths, _ in self.audio_ds.samples}
        video_map = {sid: paths for sid, paths, _ in self.video_ds.samples}

        common_sids = sorted(list(set(audio_map.keys()) & set(video_map.keys())))
        for sid in common_sids:
            self.samples.append(
                {
                    "sid": sid,
                    "audio_paths": audio_map[sid],
                    "video_paths": video_map[sid],
                    "label": self.audio_ds.label_map.get(sid, 0),
                }
            )

        print(f"[MultimodalSSLDataset] Found {len(self.samples)} aligned students (Q={len(getattr(self.audio_ds, 'selected_folders', ())) or 3}).")
        self.reload_reliability_cache()

    def __len__(self):
        return len(self.samples)

    def reload_reliability_cache(self):
        self._asr_avg_logprob = {}
        self._asr_silence_ratio = {}
        p = self.asr_cache_path
        if not p or not os.path.exists(p):
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return
            for k, v in obj.items():
                if k == "_cfg_hash":
                    continue
                if not isinstance(v, dict):
                    continue
                if "avg_logprob" in v and v["avg_logprob"] is not None:
                    try:
                        self._asr_avg_logprob[str(k)] = float(v["avg_logprob"])
                    except Exception:
                        pass
                if "silence_ratio" in v and v["silence_ratio"] is not None:
                    try:
                        self._asr_silence_ratio[str(k)] = float(v["silence_ratio"])
                    except Exception:
                        pass
        except Exception:
            return

    def _asr_norm_logprob(self, lp):
        if lp is None:
            return 0.0
        lo = self.asr_logprob_lo
        hi = self.asr_logprob_hi
        if hi <= lo:
            return 0.0
        return float(np.clip((float(lp) - lo) / (hi - lo), 0.0, 1.0))

    def _compute_reliability(self, audio_path: str, wav: np.ndarray):
        mode = str(self.reliability_mode or "none").lower()
        if mode == "none":
            return 1.0

        sr_cached = self._asr_silence_ratio.get(audio_path, None) if self.prefer_cache_silence else None
        if sr_cached is None:
            sr_cached = _silence_ratio(wav, thr=self.silence_thr)
        q_sil = 1.0 - float(sr_cached)

        lp = self._asr_avg_logprob.get(audio_path, None)
        q_asr = self._asr_norm_logprob(lp)

        if mode in ("silence", "sil"):
            return float(np.clip(q_sil, 0.0, 1.0))
        if mode in ("asr", "logprob"):
            return float(np.clip(q_asr, 0.0, 1.0))
        if mode in ("asr+silence", "both", "asr_silence"):
            return float(np.clip(0.5 * q_asr + 0.5 * q_sil, 0.0, 1.0))
        return 1.0

    def _fix_len(self, wav: np.ndarray) -> np.ndarray:
        w = np.asarray(wav).reshape(-1).astype(np.float32)
        L = int(self.target_len)
        if w.size == 0:
            return np.zeros((L,), dtype=np.float32)
        if w.size >= L:
            start = int((w.size - L) // 2)
            return w[start : start + L].astype(np.float32)
        pad = L - w.size
        return np.concatenate([w, np.zeros((pad,), dtype=np.float32)], axis=0).astype(np.float32)

    def _load_video_tensor_with_audio(self, vpath: str, wav: np.ndarray, sr: int) -> torch.Tensor:
        if not os.path.exists(vpath):
            return torch.zeros(self.video_ds.num_frames, 3, 224, 224, dtype=torch.float32)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            return torch.zeros(self.video_ds.num_frames, 3, 224, 224, dtype=torch.float32)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return torch.zeros(self.video_ds.num_frames, 3, 224, 224, dtype=torch.float32)

        init_bbox = None
        if self.video_ds.mask_background:
            init_bbox = self.video_ds._detect_face_bbox_multi_frames(cap, frame_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 默认三等分
        t1 = frame_count // 3
        t2 = (2 * frame_count) // 3

        # 使用音频分段对齐（与 video_binary/video_dataset.py 的策略一致）
        try:
            s_idx, e_idx = get_audio_segment_indices(wav, sr, threshold=self.segment_threshold)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                t1 = int((s_idx / sr) * fps)
                t2 = int((e_idx / sr) * fps)
                t1 = max(0, min(t1, frame_count))
                t2 = max(t1, min(t2, frame_count))
        except Exception:
            pass

        if self.video_ds.num_frames == 1:
            pick_idx = 0
            if t1 > 0:
                pick_idx = max(0, min(frame_count - 1, (t1 - 1) // 2))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pick_idx))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                if getattr(self.video_ds, "rotate180", False):
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame, _ = self.video_ds._apply_face_mask_rgb(frame, init_bbox)
            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            return transform(Image.fromarray(frame)).unsqueeze(0)

        n_thinking = self.video_ds.num_frames // 3
        n_speaking = self.video_ds.num_frames // 3
        n_post = self.video_ds.num_frames - n_thinking - n_speaking

        th_idx = [0 + i for i in self.video_ds._uniform_indices(t1, n_thinking)]
        sp_idx = [t1 + i for i in self.video_ds._uniform_indices(max(t2 - t1, 0), n_speaking)]
        po_idx = [t2 + i for i in self.video_ds._uniform_indices(max(frame_count - t2, 0), n_post)]
        indices = th_idx + sp_idx + po_idx

        frames = []
        collected = 0
        target_set = set(indices)
        idx = 0
        prev_bbox = init_bbox
        while collected < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in target_set:
                if frame is not None:
                    if getattr(self.video_ds, "rotate180", False):
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame, prev_bbox = self.video_ds._apply_face_mask_rgb(frame, prev_bbox)
                    frames.append(Image.fromarray(frame))
                else:
                    frames.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
                collected += 1
            idx += 1
        cap.release()

        while len(frames) < self.video_ds.num_frames:
            frames.append(frames[-1] if frames else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return torch.stack([transform(f) for f in frames])

    def _frame_cache_path(self, sid: str, q_idx: int, audio_path: str, video_path: str) -> str:
        key_obj = {
            "ap": str(audio_path),
            "vp": str(video_path),
            "num_frames": int(self.video_ds.num_frames),
            "mask_background": bool(self.video_ds.mask_background),
            "face_expand": float(getattr(self.video_ds, "face_expand", 0.0)),
            "rotate180": bool(getattr(self.video_ds, "rotate180", False)),
            "segment_threshold": float(self.segment_threshold),
            "sample_rate": int(self.sample_rate),
            "seconds": int(self.seconds),
        }
        h = hashlib.sha1(json.dumps(key_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        sub = os.path.join(self.frame_cache_dir, str(sid), f"Q{int(q_idx)+1}")
        return os.path.join(sub, f"{h}.pt")

    def _load_video_tensor_cached(self, sid: str, q_idx: int, audio_path: str, video_path: str, wav_full: np.ndarray, sr: int) -> torch.Tensor:
        if not self.cache_frames:
            return self._load_video_tensor_with_audio(video_path, wav_full, sr)
        cache_p = self._frame_cache_path(sid=sid, q_idx=q_idx, audio_path=audio_path, video_path=video_path)
        try:
            if os.path.exists(cache_p):
                obj = torch.load(cache_p, map_location="cpu")
                if isinstance(obj, dict) and "frames" in obj:
                    x = obj["frames"]
                else:
                    x = obj
                if torch.is_tensor(x):
                    x = x.to(dtype=torch.float32)
                    if x.dim() == 4 and x.size(1) == 3:
                        return x
        except Exception:
            pass

        x = self._load_video_tensor_with_audio(video_path, wav_full, sr)
        try:
            os.makedirs(os.path.dirname(cache_p), exist_ok=True)
            tmp = cache_p + ".tmp"
            torch.save({"frames": x.to(device="cpu", dtype=torch.float16)}, tmp)
            os.replace(tmp, cache_p)
        except Exception:
            pass
        return x

    def __getitem__(self, idx):
        item = self.samples[idx]
        sid = str(item["sid"])
        a_paths = list(item.get("audio_paths") or [])
        v_paths = list(item.get("video_paths") or [])
        Q = max(len(a_paths), len(v_paths), 3)
        a_paths = (a_paths + [None] * Q)[:Q]
        v_paths = (v_paths + [None] * Q)[:Q]

        waves = []
        videos = []
        q_mask = []
        reliability_list = []

        num_frames = int(self.video_ds.num_frames)
        if num_frames == 1:
            segmask_out = torch.tensor([0], dtype=torch.long)
        else:
            n = num_frames // 3
            segmask_out = torch.tensor([0] * n + [1] * n + [2] * (num_frames - 2 * n), dtype=torch.long)

        for qi, (ap, vp) in enumerate(zip(a_paths, v_paths)):
            ok = 0
            if ap and os.path.exists(ap) and vp and os.path.exists(vp):
                ok = 1

            if ok == 1:
                wav_full, sr = _read_wav(ap, target_sr=self.sample_rate, use_highpass=self.use_highpass)
                wav_fixed = self._fix_len(wav_full)
                video = self._load_video_tensor_cached(sid=sid, q_idx=qi, audio_path=ap, video_path=vp, wav_full=wav_full, sr=sr)
                rel = self._compute_reliability(str(ap), wav_full)
                reliability_list.append(rel)
            else:
                wav_fixed = np.zeros((self.target_len,), dtype=np.float32)
                video = torch.zeros(self.video_ds.num_frames, 3, 224, 224, dtype=torch.float32)
                rel = 0.0

            waves.append(wav_fixed)
            videos.append(video)
            q_mask.append(ok)

        reliability = float(np.mean(reliability_list)) if len(reliability_list) > 0 else 0.0

        return {
            "id": sid,
            "audio": torch.from_numpy(np.stack(waves, axis=0)).float(),  # (Q, L)
            "video": torch.stack(videos, dim=0),  # (Q, T, C, H, W)
            "q_mask": torch.tensor(q_mask, dtype=torch.long),  # (Q,)
            "segmask": segmask_out,
            "label": torch.tensor(item["label"]).long(),
            "reliability": torch.tensor(reliability).float(),
        }

def _parse_csv_list(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _post_process_text(text: str, max_chars: int, blacklist):
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        return ""
    for kw in blacklist:
        if kw and kw in text:
            return ""
    return text

def _infer_qkey_from_sample(sample):
    sid = str(sample.get("id", ""))
    m = re.search(r"_Q(\d+)$", sid)
    if not m:
        return ""
    try:
        q = int(m.group(1))
    except Exception:
        return ""
    if q <= 0:
        return ""
    return f"A{q}"

def _make_initial_prompt(qkey: str):
    question_prompts = {
        "A1": "问题：你最近感觉怎么样？",
        "A2": "问题：你感觉你现在和之前相比有什么不一样吗？",
        "A3": "问题：你觉得自己是个什么样的人？请对自己各方面进行评价？",
        "A4": "问题：你最近心情怎么样？",
        "A5": "问题：你最近的睡眠怎么样？",
        "A6": "问题：你最近身体有哪些不舒服或疼痛的异常表现？",
        "A7": "问题：你是为什么来就医？",
        "A8": "问题：你与家人和周围人的关系怎么样？",
    }
    topic_prompt = question_prompts.get(qkey, "")
    return f"""
以下是一段受试者的简短口头回答。
{topic_prompt}
请仅转录明确听到的内容，不要添加或猜测。
""".strip()

class WhisperASRGate:
    def __init__(self, model_dir: str, device: torch.device, torch_dtype: torch.dtype, language: str, task: str):
        self.model_dir = model_dir
        self.device = device
        self.torch_dtype = torch_dtype
        self.language = language
        self.task = task
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_dir)

    @torch.no_grad()
    def transcribe(self, audio_path: str, initial_prompt: str, temperature: float = 0.0):
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        sr_ratio = _silence_ratio(wav, thr=0.01)
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

        prompt_ids = None
        try:
            if initial_prompt:
                prompt_ids = self.processor.tokenizer.get_prompt_ids(initial_prompt)
        except Exception:
            prompt_ids = None

        generate_kwargs = {
            "language": self.language,
            "task": self.task,
            "temperature": temperature,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if prompt_ids is not None and len(prompt_ids) > 0:
            generate_kwargs["prompt_ids"] = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)

        out = self.model.generate(input_features, **generate_kwargs)
        sequences = out.sequences
        text = self.processor.batch_decode(sequences, skip_special_tokens=True)[0]

        avg_logprob = None
        try:
            scores = out.scores
            if scores is not None and len(scores) > 0:
                transition = self.model.compute_transition_scores(
                    sequences,
                    scores,
                    normalize_logits=True
                )
                finite = transition[torch.isfinite(transition)]
                if finite.numel() > 0:
                    avg_logprob = float(finite.mean().item())
        except Exception:
            avg_logprob = None

        return text, avg_logprob, sr_ratio

def _asr_gate_filter_samples(
    samples,
    gate: WhisperASRGate,
    max_chars: int,
    blacklist,
    min_avg_logprob: float,
    cache_path: str,
    cache_cfg_hash: str,
    hard_filter: bool,
    rank: int,
    log_print,
):
    cache = {}
    if cache_path:
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if isinstance(cache, dict) and "_cfg_hash" in cache and str(cache["_cfg_hash"]) != str(cache_cfg_hash):
                    cache = {"_cfg_hash": str(cache_cfg_hash)}
        except Exception:
            cache = {}
    if isinstance(cache, dict) and "_cfg_hash" not in cache:
        cache["_cfg_hash"] = str(cache_cfg_hash)

    kept = []
    stats = {
        "total": 0,
        "kept": 0,
        "drop_empty": 0,
        "drop_long_or_blacklist": 0,
        "drop_low_conf": 0,
        "drop_error": 0,
    }

    for s in samples:
        stats["total"] += 1
        audio_paths = []
        ap0 = str(s.get("audio_path", "")).strip()
        if ap0:
            audio_paths = [ap0]
        else:
            aps = s.get("audio_paths", None)
            if isinstance(aps, (list, tuple)):
                audio_paths = [str(p) for p in aps if p]
        audio_paths = [p for p in audio_paths if os.path.exists(p)]
        if len(audio_paths) == 0:
            stats["drop_error"] += 1
            if not hard_filter:
                kept.append(s)
                stats["kept"] += 1
            continue

        sample_ok = False
        any_checked = False
        for qi, ap in enumerate(audio_paths):
            cached = cache.get(ap, None)
            if isinstance(cached, dict) and "ok" in cached:
                any_checked = True
                if cached.get("ok", False):
                    sample_ok = True
                else:
                    reason = str(cached.get("reason", ""))
                    if reason == "empty":
                        stats["drop_empty"] += 1
                    elif reason == "long_or_blacklist":
                        stats["drop_long_or_blacklist"] += 1
                    elif reason == "low_conf":
                        stats["drop_low_conf"] += 1
                    else:
                        stats["drop_error"] += 1
                continue

            try:
                qkey = ""
                if "audio_paths" in s and len(audio_paths) <= 3:
                    qkey = f"A{qi + 1}"
                if not qkey:
                    qkey = _infer_qkey_from_sample(s)
                initial_prompt = _make_initial_prompt(qkey) if qkey else ""
                raw_text, avg_logprob, sr_ratio = gate.transcribe(ap, initial_prompt=initial_prompt, temperature=0.0)
                text = _post_process_text(raw_text, max_chars=max_chars, blacklist=blacklist)

                any_checked = True
                if not text:
                    cache[ap] = {"ok": False, "reason": "long_or_blacklist" if (raw_text and str(raw_text).strip()) else "empty", "avg_logprob": avg_logprob, "silence_ratio": sr_ratio}
                    if raw_text and str(raw_text).strip():
                        stats["drop_long_or_blacklist"] += 1
                    else:
                        stats["drop_empty"] += 1
                    continue

                if min_avg_logprob is not None and min_avg_logprob > -1e9:
                    if avg_logprob is None or avg_logprob < float(min_avg_logprob):
                        cache[ap] = {"ok": False, "reason": "low_conf", "avg_logprob": avg_logprob, "silence_ratio": sr_ratio, "text": text}
                        stats["drop_low_conf"] += 1
                        continue

                cache[ap] = {"ok": True, "avg_logprob": avg_logprob, "silence_ratio": sr_ratio, "text": text}
                sample_ok = True
            except Exception as e:
                cache[ap] = {"ok": False, "reason": "error", "error": str(e)}
                stats["drop_error"] += 1

        if not any_checked:
            stats["drop_error"] += 1
            if not hard_filter:
                kept.append(s)
                stats["kept"] += 1
            continue

        if not sample_ok:
            if hard_filter:
                continue
            kept.append(s)
            stats["kept"] += 1
            continue

        kept.append(s)
        stats["kept"] += 1

    if cache_path and rank == 0:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log_print(f"[ASR-Gate] Failed to write cache: {e}")

    return kept if hard_filter else kept, stats

class MultimodalSSLModel(nn.Module):
    """
    多模态对齐模型：包含两个 Backbone 和映射头
    """
    def __init__(self, audio_dim, video_dim, projection_dim=512, pooling_type="mean"):
        super().__init__()
        self.pooling_type = pooling_type
        # 映射头将不同维度的特征映射到统一的对比空间
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        if pooling_type == "attention":
            self.a_pool = nn.Sequential(nn.Linear(audio_dim, 1), nn.Softmax(dim=1))
            self.v_pool = nn.Sequential(nn.Linear(video_dim, 1), nn.Softmax(dim=1))

    def forward(self, audio_x, video_x):
        if audio_x.dim() == 3:
            if self.pooling_type == "attention":
                a_weights = self.a_pool(audio_x)
                a_feat = (audio_x * a_weights).sum(dim=1)
            elif self.pooling_type == "max":
                a_feat = audio_x.max(dim=1)[0]
            else:
                a_feat = audio_x.mean(dim=1)
        else:
            a_feat = audio_x

        if video_x.dim() == 3:
            if self.pooling_type == "attention":
                v_weights = self.v_pool(video_x)
                v_feat = (video_x * v_weights).sum(dim=1)
            elif self.pooling_type == "max":
                v_feat = video_x.max(dim=1)[0]
            else:
                v_feat = video_x.mean(dim=1)
        else:
            v_feat = video_x

        z_a = F.normalize(self.audio_proj(a_feat), dim=-1)
        z_v = F.normalize(self.video_proj(v_feat), dim=-1)
        return z_a, z_v


class VideoPatchMHAPool(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        heads = int(num_heads) if (int(num_heads) > 0 and d_model % int(num_heads) == 0) else 1
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 5:
            tokens = tokens.squeeze(1)
        if tokens.dim() != 4:
            raise ValueError(f"Expected video tokens with shape (B,T,P,D) or (B,1,T,P,D), got {tuple(tokens.shape)}")
        B, T, P, D = tokens.shape
        if P <= 1:
            return tokens[:, :, 0, :]
        patches = tokens[:, :, 1:, :]
        n = B * T
        x = patches.reshape(n, patches.size(2), D)
        q = self.query.to(device=x.device, dtype=x.dtype).view(1, 1, D).expand(n, 1, D)
        out, _ = self.mha(q, x, x, need_weights=False)
        return out.squeeze(1).view(B, T, D)

def contrastive_loss(z_a, z_v, temperature=0.07, weights: torch.Tensor = None):
    """
    InfoNCE 对齐损失：让同一对音频视频特征互为正样本，其他为负样本。
    """
    batch_size = z_a.size(0)
    # 计算相似度矩阵 (B, B)
    logits = torch.matmul(z_a, z_v.T) / temperature
    
    # 正样本在对角线上
    labels = torch.arange(batch_size).to(z_a.device)
    
    # 双向对比：Audio -> Video 和 Video -> Audio
    loss_a2v = F.cross_entropy(logits, labels, reduction="none")
    loss_v2a = F.cross_entropy(logits.T, labels, reduction="none")
    
    if weights is None:
        return 0.5 * (loss_a2v.mean() + loss_v2a.mean())

    w = weights.to(device=z_a.device, dtype=loss_a2v.dtype).view(-1)
    if w.numel() != batch_size:
        raise ValueError(f"Expected weights with shape ({batch_size},), got {tuple(w.shape)}")
    w = w.detach()
    denom = w.sum().clamp_min(1e-8)
    return 0.5 * ((w * loss_a2v).sum() / denom + (w * loss_v2a).sum() / denom)

def _extract_fused_features_from_head(
    head: QuestionAttentionBinary,
    tokens: torch.Tensor,
    q_mask: torch.Tensor,
    segmask: torch.Tensor = None,
    patch_bg_mask: torch.Tensor = None,
) -> torch.Tensor:
    if getattr(head, "audio_layer", "last") == "all" and tokens.dim() == 5:
        B, Q, Layers, T, D = tokens.shape
        w = torch.softmax(head.layer_weights[:Layers], dim=0)
        tokens = (tokens * w.view(1, 1, Layers, 1, 1)).sum(dim=2)

    if getattr(head, "pool_mode", "") == "residual" and tokens.dim() == 5:
        q_feat = head._pool_residual(tokens, patch_bg_mask=patch_bg_mask)
    else:
        if tokens.dim() == 5:
            tokens = head.pool_video_patches(tokens, patch_bg_mask=patch_bg_mask)
        tokens = head.projector(tokens)
        B, Q, L, D = tokens.shape

        def pool_token(t: torch.Tensor, mode: str) -> torch.Tensor:
            if t.size(2) == 0:
                return torch.zeros(B, Q, D, device=t.device, dtype=t.dtype)
            if "mean" in mode:
                return t.mean(dim=2)
            if "max" in mode:
                return t.max(dim=2)[0]
            if "lstm" in mode:
                curr_T = t.size(2)
                t_reshaped = t.view(-1, curr_T, D)
                _, (h_n, _) = head.token_lstm(t_reshaped)
                pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
                return pooled.view(B, Q, D)
            w = torch.softmax(head.token_attn(t), dim=2)
            return (t * w).sum(dim=2)

        if L <= 3 and not str(head.pool_mode).startswith("segment_"):
            q_feat = tokens.squeeze(2)
        elif str(head.pool_mode).startswith("segment_"):
            if segmask is not None:
                m = segmask[0]
                idx1 = (m == 0).nonzero(as_tuple=True)[0]
                idx2 = (m == 1).nonzero(as_tuple=True)[0]
                idx3 = (m == 2).nonzero(as_tuple=True)[0]
                t1 = tokens[:, :, idx1, :] if len(idx1) > 0 else tokens[:, :, :0, :]
                t2 = tokens[:, :, idx2, :] if len(idx2) > 0 else tokens[:, :, :0, :]
                t3 = tokens[:, :, idx3, :] if len(idx3) > 0 else tokens[:, :, :0, :]
                f1 = pool_token(t1, head.pool_mode)
                f2 = pool_token(t2, head.pool_mode)
                f3 = pool_token(t3, head.pool_mode)
                segments = torch.stack([f1, f2, f3], dim=2)
                seg_w = torch.softmax(head.segment_attn(segments), dim=2)
                q_feat = (segments * seg_w).sum(dim=2)
            elif L >= 8:
                l1 = L // 3
                l2 = L // 3
                t1 = tokens[:, :, :l1, :]
                t2 = tokens[:, :, l1 : l1 + l2, :]
                t3 = tokens[:, :, l1 + l2 :, :]
                f1 = pool_token(t1, head.pool_mode)
                f2 = pool_token(t2, head.pool_mode)
                f3 = pool_token(t3, head.pool_mode)
                segments = torch.stack([f1, f2, f3], dim=2)
                seg_w = torch.softmax(head.segment_attn(segments), dim=2)
                q_feat = (segments * seg_w).sum(dim=2)
            else:
                q_feat = pool_token(tokens, head.pool_mode)
        else:
            q_feat = pool_token(tokens, head.pool_mode)

    if str(head.q_pool_mode) == "mean":
        mask = (q_mask > 0).to(dtype=q_feat.dtype, device=q_feat.device).unsqueeze(-1)
        fused = (q_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    elif str(head.q_pool_mode) == "max":
        mask = (q_mask > 0).to(dtype=torch.bool, device=q_feat.device).unsqueeze(-1)
        neg_inf = torch.finfo(q_feat.dtype).min
        q_feat_masked = q_feat.masked_fill(~mask, neg_inf)
        fused = q_feat_masked.max(dim=1)[0]
    else:
        q_logits = head.q_attn(q_feat).squeeze(-1)
        mask = (q_mask > 0).to(dtype=torch.bool, device=q_logits.device)
        neg_inf = torch.finfo(q_logits.dtype).min
        q_logits = q_logits.masked_fill(~mask, neg_inf)
        q_weights = torch.softmax(q_logits, dim=1).unsqueeze(-1)
        fused = (q_feat * q_weights).sum(dim=1)

    return fused

def train_one_epoch_multimodal_ssl(model, audio_head, video_head, clip_model, audio_model, audio_processor, loader, optimizer, device, **kwargs):
    model.train()
    audio_head.train()
    video_head.train()
    if audio_model: audio_model.train()
    if clip_model: clip_model.train()
    
    total_loss = 0
    step_count = 0
    grad_accum = kwargs.get("grad_accum", 1)
    
    pbar = tqdm(loader, desc="MM-SSL Training")
    for i, batch in enumerate(pbar):
        audio = batch["audio"].to(device) # (B, Q, L)
        video = batch["video"].to(device) # (B, Q, T, C, H, W)
        q_mask = batch["q_mask"].to(device)
        segmask = batch.get("segmask", None)
        if segmask is not None:
            segmask = segmask.to(device)
        reliability = batch.get("reliability", None)
        if video.dim() != 6:
            raise RuntimeError(f"Expected video with shape (B,Q,T,C,H,W), got {tuple(video.shape)}")
        if video.size(2) == 0:
            B = video.size(0)
            Q = video.size(1)
            video = torch.zeros(B, Q, 1, 3, 224, 224, device=device, dtype=video.dtype)
        
        # 1. 提取特征
        if kwargs.get("audio_type") == "whisper":
            a_tokens = extract_whisper_tokens(audio_model, audio_processor, audio, device, audio_layer=str(kwargs.get("audio_layer", "last")))
        else:
            a_tokens = extract_wav2vec_tokens(audio_model, audio_processor, audio, device, audio_layer=str(kwargs.get("audio_layer", "last")))
        
        v_tokens = extract_clip_tokens(clip_model, video)
        
        patch_bg_mask = None
        use_pm = bool(kwargs.get("patch_mask_background", False) and kwargs.get("mask_background", False))
        if use_pm and clip_model is not None:
            patch_size = kwargs.get("patch_size", None)
            if patch_size is None:
                patch_size = _infer_patch_size_from_vision_model(clip_model)
            patch_bg_mask = _patch_background_mask_from_frames(video, patch_size=int(patch_size), var_thr=float(kwargs.get("patch_mask_var_thr", 1e-6)))

        a_fused = _extract_fused_features_from_head(audio_head, a_tokens, q_mask=q_mask)
        v_fused = _extract_fused_features_from_head(video_head, v_tokens, q_mask=q_mask, segmask=segmask, patch_bg_mask=patch_bg_mask)

        z_a, z_v = model(a_fused, v_fused)
        
        # 3. 计算损失
        w = None
        if bool(kwargs.get("use_reliability", False)) and reliability is not None:
            rel = reliability.to(device=device, dtype=torch.float32).view(-1).clamp(0.0, 1.0)
            w_min = float(kwargs.get("rel_w_min", 0.1))
            w = (w_min + (1.0 - w_min) * rel).to(device=device)
        loss = contrastive_loss(z_a, z_v, weights=w)
        
        (loss / grad_accum).backward()
        
        if (i + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        step_count += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / max(1, step_count)

@torch.no_grad()
def evaluate_multimodal_ssl(model, audio_head, video_head, clip_model, audio_model, audio_processor, loader, device, **kwargs):
    model.eval()
    audio_head.eval()
    video_head.eval()
    if audio_model: audio_model.eval()
    if clip_model: clip_model.eval()
    
    total_loss = 0
    step_count = 0
    
    for batch in loader:
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        q_mask = batch["q_mask"].to(device)
        segmask = batch.get("segmask", None)
        if segmask is not None:
            segmask = segmask.to(device)
        reliability = batch.get("reliability", None)
        if video.dim() != 6:
            raise RuntimeError(f"Expected video with shape (B,Q,T,C,H,W), got {tuple(video.shape)}")
        if video.size(2) == 0:
            B = video.size(0)
            Q = video.size(1)
            video = torch.zeros(B, Q, 1, 3, 224, 224, device=device, dtype=video.dtype)
        
        if kwargs.get("audio_type") == "whisper":
            a_tokens = extract_whisper_tokens(audio_model, audio_processor, audio, device, audio_layer=str(kwargs.get("audio_layer", "last")))
        else:
            a_tokens = extract_wav2vec_tokens(audio_model, audio_processor, audio, device, audio_layer=str(kwargs.get("audio_layer", "last")))
        
        v_tokens = extract_clip_tokens(clip_model, video)

        patch_bg_mask = None
        use_pm = bool(kwargs.get("patch_mask_background", False) and kwargs.get("mask_background", False))
        if use_pm and clip_model is not None:
            patch_size = kwargs.get("patch_size", None)
            if patch_size is None:
                patch_size = _infer_patch_size_from_vision_model(clip_model)
            patch_bg_mask = _patch_background_mask_from_frames(video, patch_size=int(patch_size), var_thr=float(kwargs.get("patch_mask_var_thr", 1e-6)))

        a_fused = _extract_fused_features_from_head(audio_head, a_tokens, q_mask=q_mask)
        v_fused = _extract_fused_features_from_head(video_head, v_tokens, q_mask=q_mask, segmask=segmask, patch_bg_mask=patch_bg_mask)

        z_a, z_v = model(a_fused, v_fused)
        w = None
        if bool(kwargs.get("use_reliability", False)) and reliability is not None:
            rel = reliability.to(device=device, dtype=torch.float32).view(-1).clamp(0.0, 1.0)
            w_min = float(kwargs.get("rel_w_min", 0.1))
            w = (w_min + (1.0 - w_min) * rel).to(device=device)
        loss = contrastive_loss(z_a, z_v, weights=w)
        
        total_loss += loss.item()
        step_count += 1
            
    return total_loss / max(1, step_count)

def count_parameters(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] Total params: {total_params:,} | Trainable params: {trainable_params:,}")
    return total_params, trainable_params

def main():
    parser = argparse.ArgumentParser(description="Multimodal SSL (Audio-Video Alignment)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--audio_type", type=str, default="whisper", choices=["whisper", "wav2vec"])
    parser.add_argument("--video_type", type=str, default="siglip", choices=["clip", "siglip"])
    parser.add_argument("--pool_mode", type=str, default="lstm")
    parser.add_argument("--q_pool_mode", type=str, default="max", choices=["mean", "max", "attention"])
    parser.add_argument("--patch_pool_mode", type=str, default="attn", choices=["cls", "attn", "mean", "max"])
    parser.add_argument("--patch_attn_type", type=str, default="mha", choices=["mlp", "gated", "dot", "mha", "selfatt"])
    parser.add_argument("--patch_mha_heads", type=int, default=8)
    parser.add_argument("--patch_selfatt_layers", type=int, default=1)
    parser.add_argument("--audio_layer", type=str, default="last")
    parser.add_argument("--use_projector", action="store_true", default=False)
    parser.add_argument("--pooling_type", type=str, default="mean", choices=["mean", "attention", "max"])
    parser.add_argument("--mask_background", action="store_true", default=False)
    parser.add_argument("--patch_mask_background", action="store_true", default=False)
    parser.add_argument("--patch_mask_var_thr", type=float, default=1e-6)
    parser.add_argument("--face_expand", type=float, default=0.25)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--rotate180", action="store_true", default=True)
    parser.add_argument("--segment_threshold", type=float, default=0.005)
    parser.add_argument("--use_highpass", action="store_true", default=True)
    parser.add_argument("--unfreeze_last_k", type=int, default=2, help="Unfreeze last K layers of BOTH backbones")
    parser.add_argument("--num_workers", type=int, default=8 if platform.system().lower() == "windows" else 4)
    parser.add_argument("--out_dir", type=str, default="runs_multimodal_ssl")
    parser.add_argument("--asr_gate", action="store_true", default=False)
    parser.add_argument("--asr_hard_filter", action="store_true", default=False)
    parser.add_argument("--asr_gate_all_splits", action="store_true", default=False)
    parser.add_argument("--asr_model_dir", type=str, default="")
    parser.add_argument("--asr_language", type=str, default="zh")
    parser.add_argument("--asr_task", type=str, default="transcribe")
    parser.add_argument("--asr_max_chars", type=int, default=50)
    parser.add_argument("--asr_blacklist", type=str, default="大家好,我是,欢迎,打赏,订阅,明镜,栏目")
    parser.add_argument("--asr_min_avg_logprob", type=float, default=-1.0)
    parser.add_argument("--asr_cache_path", type=str, default="")
    parser.add_argument("--rel_mode", type=str, default="none", choices=["none", "silence", "asr", "asr+silence"])
    parser.add_argument("--rel_w_min", type=float, default=0.1)
    parser.add_argument("--rel_asr_logprob_lo", type=float, default=-3.0)
    parser.add_argument("--rel_asr_logprob_hi", type=float, default=-0.5)
    parser.add_argument("--rel_silence_thr", type=float, default=0.01)
    parser.add_argument("--rel_no_cache_silence", action="store_true", default=False)
    parser.add_argument("--no_cache_frames", action="store_true", default=False)
    parser.add_argument("--frame_cache_dir", type=str, default="")
    args = parser.parse_args()

    use_ddp, rank, world_size, local_rank, device = _setup_distributed()
    set_seed(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        log_file = open(os.path.join(run_dir, "train_log.txt"), "w", encoding="utf-8")
        def log_print(msg):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
        log_print(f"Project Root: {parent_dir}")
        log_print(f"Run Directory: {run_dir}")
        log_print(f"Args: {json.dumps(vars(args), indent=2)}")
    else:
        def log_print(msg): pass

    # 路径配置
    paths = get_school_paths(audio_type=args.audio_type, video_type=args.video_type)
    cfg = paths

    audio_root = cfg.get("audio_dir") or cfg.get("audio_root") or ""
    video_root = cfg.get("video_dir") or cfg.get("video_root") or ""
    label_path = cfg.get("csv_path") or cfg.get("label_path") or ""
    allowed_ids_path = cfg.get("allowed_ids_path")
    asr_cache_path = args.asr_cache_path.strip() if isinstance(args.asr_cache_path, str) else ""
    if not asr_cache_path:
        asr_cache_path = os.path.join(args.out_dir, "asr_gate_cache.json")
    if rank == 0:
        log_print("\n[Path Debug]")
        log_print(f"  - audio_root: {audio_root}")
        log_print(f"  - video_root: {video_root}")
        log_print(f"  - label_path: {label_path}")
        log_print(f"  - allowed_ids_path: {allowed_ids_path}")
        log_print(f"  - asr_cache_path: {asr_cache_path}")
    
    # Dataset
    train_ds = MultimodalSSLDataset(
        audio_root=audio_root, 
        video_root=video_root, 
        label_path=label_path,
        allowed_ids_path=allowed_ids_path,
        split="train",
        num_frames=args.num_frames,
        mask_background=args.mask_background,
        face_expand=args.face_expand,
        rotate180=bool(args.rotate180),
        segment_threshold=float(args.segment_threshold),
        use_highpass=bool(args.use_highpass),
        patch_mask_background=bool(args.patch_mask_background),
        patch_mask_var_thr=float(args.patch_mask_var_thr),
        reliability_mode=args.rel_mode,
        asr_cache_path=asr_cache_path,
        asr_logprob_lo=args.rel_asr_logprob_lo,
        asr_logprob_hi=args.rel_asr_logprob_hi,
        silence_thr=args.rel_silence_thr,
        prefer_cache_silence=(not args.rel_no_cache_silence),
        cache_frames=(not bool(args.no_cache_frames)),
        frame_cache_dir=str(args.frame_cache_dir or ""),
    )
    val_ds = MultimodalSSLDataset(
        audio_root=audio_root, 
        video_root=video_root, 
        label_path=label_path,
        allowed_ids_path=allowed_ids_path,
        split="val",
        num_frames=args.num_frames,
        mask_background=args.mask_background,
        face_expand=args.face_expand,
        rotate180=bool(args.rotate180),
        segment_threshold=float(args.segment_threshold),
        use_highpass=bool(args.use_highpass),
        patch_mask_background=bool(args.patch_mask_background),
        patch_mask_var_thr=float(args.patch_mask_var_thr),
        reliability_mode=args.rel_mode,
        asr_cache_path=asr_cache_path,
        asr_logprob_lo=args.rel_asr_logprob_lo,
        asr_logprob_hi=args.rel_asr_logprob_hi,
        silence_thr=args.rel_silence_thr,
        prefer_cache_silence=(not args.rel_no_cache_silence),
        cache_frames=(not bool(args.no_cache_frames)),
        frame_cache_dir=str(args.frame_cache_dir or ""),
    )

    if args.asr_gate:
        train_before = len(train_ds.samples)
        val_before = len(val_ds.samples)
        asr_model_dir = args.asr_model_dir.strip() if isinstance(args.asr_model_dir, str) else ""
        if not asr_model_dir:
            asr_model_dir = cfg["audio_model_path"]

        blacklist = _parse_csv_list(args.asr_blacklist)
        cache_path = asr_cache_path

        cache_cfg = {
            "asr_model_dir": asr_model_dir,
            "asr_language": args.asr_language,
            "asr_task": args.asr_task,
            "asr_max_chars": args.asr_max_chars,
            "asr_blacklist": blacklist,
            "asr_min_avg_logprob": args.asr_min_avg_logprob,
        }
        cache_cfg_hash = hashlib.sha1(json.dumps(cache_cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]

        gate = None
        if rank == 0:
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
            gate = WhisperASRGate(
                model_dir=asr_model_dir,
                device=device,
                torch_dtype=torch_dtype,
                language=args.asr_language,
                task=args.asr_task,
            )

            train_kept, train_stats = _asr_gate_filter_samples(
                train_ds.samples,
                gate=gate,
                max_chars=args.asr_max_chars,
                blacklist=blacklist,
                min_avg_logprob=args.asr_min_avg_logprob,
                cache_path=cache_path,
                cache_cfg_hash=cache_cfg_hash,
                hard_filter=bool(args.asr_hard_filter),
                rank=rank,
                log_print=log_print,
            )
            val_kept = val_ds.samples
            val_stats = None
            if bool(args.asr_gate_all_splits):
                val_kept, val_stats = _asr_gate_filter_samples(
                    val_ds.samples,
                    gate=gate,
                    max_chars=args.asr_max_chars,
                    blacklist=blacklist,
                    min_avg_logprob=args.asr_min_avg_logprob,
                    cache_path=cache_path,
                    cache_cfg_hash=cache_cfg_hash,
                    hard_filter=bool(args.asr_hard_filter),
                    rank=rank,
                    log_print=log_print,
                )
            obj = {
                "train": train_kept,
                "val": val_kept,
                "train_stats": train_stats,
                "val_stats": val_stats,
                "train_before": train_before,
                "val_before": val_before,
            }
        else:
            obj = None

        if _dist_is_initialized():
            obj_list = [obj]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            obj = obj_list[0]

        if isinstance(obj, dict) and "train" in obj and "val" in obj:
            train_ds.samples = obj["train"]
            val_ds.samples = obj["val"]
            train_ds.reload_reliability_cache()
            val_ds.reload_reliability_cache()
            if rank == 0:
                log_print(f"[ASR-Gate][Train] before={obj.get('train_before')} after={len(train_ds.samples)}")
                log_print(f"[ASR-Gate][Train] {obj.get('train_stats')}")
                if bool(args.asr_gate_all_splits):
                    log_print(f"[ASR-Gate][Val]   before={obj.get('val_before')} after={len(val_ds.samples)}")
                    log_print(f"[ASR-Gate][Val]   {obj.get('val_stats')}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 加载模型
    if args.audio_type == "whisper":
        audio_model = WhisperModel.from_pretrained(cfg["audio_model_path"]).to(device)
        audio_processor = WhisperProcessor.from_pretrained(cfg["audio_model_path"])
        audio_dim = audio_model.config.d_model
    else:
        audio_model = Wav2Vec2Model.from_pretrained(cfg["audio_model_path"]).to(device)
        audio_processor = Wav2Vec2Processor.from_pretrained(cfg["audio_model_path"])
        audio_dim = audio_model.config.hidden_size

    extractor = CLIPVideoFeatureExtractor(model_path=cfg["clip_model_path"], device=str(device))
    clip_model = extractor.model
    video_dim = getattr(clip_model.config, "hidden_size", 768)
    if rank == 0:
        print(f"[Config] Using Video: {args.video_type} (dim={video_dim}), Audio: {args.audio_type} (dim={audio_dim})")

    # 解冻逻辑
    def unfreeze_backbone(model, k, modality):
        if k == 0:
            for p in model.parameters(): p.requires_grad = False
            return
        
        # 默认全部冻结
        for p in model.parameters(): p.requires_grad = False
        
        try:
            if modality == "audio":
                layers = model.encoder.layers
            else:
                layers = model.vision_model.encoder.layers if hasattr(model, "vision_model") else model.encoder.layers
            
            for blk in layers[-k:]:
                for p in blk.parameters(): p.requires_grad = True
            print(f"[SSL] Unfroze last {k} layers of {modality} model.")
        except Exception as e:
            print(f"[SSL] Warning: Failed to unfreeze {modality} layers: {e}")

    unfreeze_backbone(audio_model, args.unfreeze_last_k, "audio")
    unfreeze_backbone(clip_model, args.unfreeze_last_k, "video")

    # 对齐模型
    audio_head = QuestionAttentionBinary(
        d_model=audio_dim,
        pool_mode=args.pool_mode,
        q_pool_mode=args.q_pool_mode,
        audio_layer=args.audio_layer,
        use_projector=bool(args.use_projector),
        patch_pool_mode=args.patch_pool_mode,
        patch_attn_type=args.patch_attn_type,
        patch_mha_heads=int(args.patch_mha_heads),
        patch_selfatt_layers=int(args.patch_selfatt_layers),
    ).to(device)
    video_head = QuestionAttentionBinary(
        d_model=video_dim,
        pool_mode=args.pool_mode,
        q_pool_mode=args.q_pool_mode,
        audio_layer="last",
        use_projector=bool(args.use_projector),
        patch_pool_mode=args.patch_pool_mode,
        patch_attn_type=args.patch_attn_type,
        patch_mha_heads=int(args.patch_mha_heads),
        patch_selfatt_layers=int(args.patch_selfatt_layers),
    ).to(device)

    mm_model = MultimodalSSLModel(audio_dim, video_dim, pooling_type=args.pooling_type).to(device)
    patch_size = _infer_patch_size_from_vision_model(clip_model) if clip_model is not None else 16
    
    # 计算参数量
    if rank == 0:
        print("\n" + "="*50)
        print(f"[Config] PoolMode={args.pool_mode}, QPool={args.q_pool_mode}, PatchAttn={args.patch_attn_type}, Frames={args.num_frames}")
        count_parameters(audio_model, "Audio Backbone")
        count_parameters(clip_model, "Video Backbone")
        count_parameters(audio_head, "Audio Aggregator (QAB)")
        count_parameters(video_head, "Video Aggregator (QAB)")
        count_parameters(mm_model, "SSL Projection Head")
        total_trainable = sum(
            p.numel()
            for p in list(mm_model.parameters())
            + list(audio_head.parameters())
            + list(video_head.parameters())
            + list(audio_model.parameters())
            + list(clip_model.parameters())
            if p.requires_grad
        )
        print(f"Total Trainable Parameters in MM-SSL: {total_trainable:,}")
        print("="*50 + "\n")

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            list(mm_model.parameters())
            + list(audio_head.parameters())
            + list(video_head.parameters())
            + list(audio_model.parameters())
            + list(clip_model.parameters()),
        ),
        lr=args.lr
    )

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_multimodal_ssl(
            mm_model, audio_head, video_head, clip_model, audio_model, audio_processor, train_loader, optimizer, device,
            audio_type=args.audio_type,
            audio_layer=args.audio_layer,
            grad_accum=args.grad_accum,
            mask_background=bool(args.mask_background),
            patch_mask_background=bool(args.patch_mask_background),
            patch_mask_var_thr=float(args.patch_mask_var_thr),
            patch_size=int(patch_size),
            rel_w_min=args.rel_w_min,
            use_reliability=bool(args.asr_gate and args.rel_mode != "none"),
        )
        val_loss = evaluate_multimodal_ssl(
            mm_model, audio_head, video_head, clip_model, audio_model, audio_processor, val_loader, device,
            audio_type=args.audio_type,
            audio_layer=args.audio_layer,
            mask_background=bool(args.mask_background),
            patch_mask_background=bool(args.patch_mask_background),
            patch_mask_var_thr=float(args.patch_mask_var_thr),
            patch_size=int(patch_size),
            rel_w_min=args.rel_w_min,
            use_reliability=bool(args.asr_gate_all_splits and args.asr_gate and args.rel_mode != "none"),
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:
                save_path = os.path.join(run_dir, "best_mm_ssl.pt")
                torch.save({
                    'epoch': epoch,
                    'mm_model_state_dict': mm_model.state_dict(),
                    'audio_head_state_dict': audio_head.state_dict(),
                    'video_head_state_dict': video_head.state_dict(),
                    'audio_model_state_dict': audio_model.state_dict(),
                    'clip_model_state_dict': clip_model.state_dict(),
                    'patch_size': int(patch_size),
                    'val_loss': val_loss,
                    'args': vars(args),
                }, save_path)
                log_print(f"  >>> Best Loss Updated! Model saved to {save_path}")

        if rank == 0:
            log_print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")

    if rank == 0:
        log_file.close()

if __name__ == "__main__":
    main()
