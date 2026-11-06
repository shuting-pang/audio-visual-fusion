"""Concatenate pooled features while retaining explicit missingness flags."""

import numpy as np


def build_fusion_features(audio, video, text, missing):
    """Build [audio | video | text | missing flags] for aligned samples.

    Each feature matrix is [N, D_modality]. `missing` is [N, 3], in A/V/T
    order, with 1 meaning absent/rejected and 0 meaning present. Missing
    feature blocks are zeroed, including any NaNs in those blocks. Present
    blocks must be finite. A missing video does not discard available audio.
    """
    features = [np.asarray(x, dtype=np.float32) for x in (audio, video, text)]
    if any(x.ndim != 2 or x.shape[1] == 0 for x in features):
        raise ValueError("Features must be nonempty-width matrices [N, D]")
    n = features[0].shape[0]
    if any(x.shape[0] != n for x in features):
        raise ValueError("All modalities must contain the same aligned samples")
    missing = np.asarray(missing)
    if missing.shape != (n, 3) or not np.isin(missing, [0, 1]).all():
        raise ValueError("missing must have shape [N, 3] and contain only 0/1")
    blocks = []
    for column, values in enumerate(features):
        block = np.where(missing[:, column, None] == 1, 0.0, values)
        if not np.isfinite(block).all():
            raise ValueError("Present feature values must be finite")
        blocks.append(block)
    return np.concatenate([*blocks, missing.astype(np.float32)], axis=1)
