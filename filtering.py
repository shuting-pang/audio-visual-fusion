"""Whisper transcript filtering, independent of model downloads."""

from dataclasses import dataclass
import math

DEFAULT_BLACKLIST = ("大家好", "我是", "欢迎", "打赏", "订阅", "明镜", "栏目")


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    missing: bool
    reason: str


def filter_transcript(text, avg_logprob, *, max_chars=50,
                      min_avg_logprob=-1.0, blacklist=DEFAULT_BLACKLIST):
    """Return accepted text or an explicit missing value with its reason.

    Rules mirror the supplied research script's ASR gate. The blacklist is
    heuristic; passing the gate does not establish transcription correctness.
    Pass min_avg_logprob=None to disable confidence filtering.
    """
    text = "" if text is None else str(text).strip()
    if not text:
        return TranscriptResult("", True, "empty")
    if max_chars > 0 and len(text) > max_chars:
        return TranscriptResult("", True, "too_long")
    if any(word and word in text for word in blacklist):
        return TranscriptResult("", True, "blacklist")
    if min_avg_logprob is not None:
        if avg_logprob is None or not math.isfinite(float(avg_logprob)):
            return TranscriptResult("", True, "missing_confidence")
        if float(avg_logprob) < min_avg_logprob:
            return TranscriptResult("", True, "low_confidence")
    return TranscriptResult(text, False, "accepted")
