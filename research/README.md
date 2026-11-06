# Supplied audio-video SSL research script

`multimodal_ssl_train.py` is the supplied research snapshot, retained unchanged.
It contains question-conditioned Whisper prompts, transcript length/blacklist/confidence
gating, cached ASR metadata, optional silence/ASR reliability weighting and bidirectional
audio-video contrastive learning. Its default ASR gate is disabled; `--asr_gate` enables it.
`--asr_hard_filter` additionally removes samples for which no checked response passes.

This script depends on project-local modules that were not supplied:

- `school_binary/audio_dataset.py` and `school_binary/config.py`
- `video_binary/video.py`, `video_binary/audio.py` and `video_binary/train.py`
- Or the documented fallback imports `audio_dataset`, `config`, `video` and `train`

Restore those modules and local dataset/model paths before running this script.
Its additional imports include torchvision, transformers, librosa, OpenCV, Pillow and tqdm.
The feature-level example in the repository root runs independently of those modules.

This SSL entry point optimizes audio-video alignment loss, rather than the downstream
three-modality binary classifier in the reported results table. The new root-level
feature concatenation and MLP examples illustrate that downstream design; they have not
been used to reproduce the supplied metrics.
