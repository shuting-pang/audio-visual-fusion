"""Small synthetic missing-modality example; no patient data or model download."""

import numpy as np

from filtering import filter_transcript
from fusion import build_fusion_features


def main():
    transcript = filter_transcript("最近睡得不太好。", avg_logprob=-2.5)
    print(f"Text missing: {transcript.missing}; reason: {transcript.reason}")
    # Two samples with pooled A/V/T features. Sample 0 has rejected text;
    # sample 1 has no video. Available modalities are retained.
    audio = np.array([[1, 2], [3, 4]], dtype=np.float32)
    video = np.array([[5, 6], [np.nan, np.nan]], dtype=np.float32)
    text = np.array([[np.nan, np.nan], [7, 8]], dtype=np.float32)
    missing = np.array([[0, 0, int(transcript.missing)], [0, 1, 0]])
    fused = build_fusion_features(audio, video, text, missing)
    print("Fused shape:", fused.shape)
    print("[audio | video | text | missing A,V,T]:\n", fused)


if __name__ == "__main__":
    main()
