# Audio-Visual Fusion with Missing Modalities

### How can we combine audio, video and text when some inputs are missing or unreliable?

**Filter unreliable transcripts, keep missingness explicit, and concatenate the available modality features.** This project explores multimodal learning for depression-related interview data, with two focuses: the **filtering process** and a **multimodal-versus-unimodal comparison** under observed missingness.

## 1. Filtering and fusion workflow

![Filtering, missingness handling and concatenation](docs/assets/filtering-workflow.svg)

The diagram summarizes the workflow in the supplied experiment notes. The fusion step is called **Mask-Aware Concatenation**: pool valid video inputs, keep missingness explicit, then concatenate the modality features. Fixed validity masks exclude unusable inputs; they do not learn cross-modal gate weights.

- **Audio:** extract wav2vec features; the recorded experiment pads sequences to length 208.
- **Video:** detect/crop faces with FaceNet, extract CLIP features, and mask invalid frames/patches during aggregation. If no valid video input is available, mark the video modality as missing.
- **Text:** transcribe audio with Whisper using a question-aware prompt: retain only clearly heard answers, without guessing. Reject empty text, overly long text, blacklist matches or low-confidence output before CLIP text encoding.
- **Missingness:** retain explicit flags (`1 = missing`, `0 = present`). In the feature-level example, zero-fill absent feature blocks and append A/V/T missing flags before an MLP classifier. Available audio is retained when video or text is absent.

The supplied ASR gate uses a maximum of 50 characters, average token log probability ≥ −1.0, and a configurable blacklist. These are filtering heuristics, not guarantees that a transcript is correct. The experiment notes represent rejected text as empty, while the supplied SSL script additionally supports optional sample-level hard filtering.

## 2. Transcript filtering: text-only AUC

Whisper transcripts are filtered before CLIP text encoding. The reported text-only run has **AUC 60.16%**.

An isolated before/after-filtering AUC comparison is not available in the current experiment record; no filtering gain is claimed here.

## 3. From unimodal to multimodal fusion

![AUC gains from unimodal baselines to multimodal fusion](docs/assets/reported-results.svg)

**Mask-Aware Concatenation (A/V/T)** combines audio, video and transcript features and reaches **60.97% AUC** in the supplied experiment table.

| Unimodal baseline | Unimodal AUC (%) | Multimodal AUC (%) | Gain (percentage points) |
| --- | ---: | ---: | ---: |
| Audio | 60.10 | 60.97 | **+0.87** |
| Video | 59.18 | 60.97 | **+1.79** |
| Text | 60.16 | 60.97 | **+0.81** |

Fusion improves the recorded AUC by **0.81 percentage points over the strongest unimodal baseline (text)**. These are AUC comparisons from the existing experiment notes, rather than a new training run. See [AUC gains](results/auc_gains.csv) for the displayed values; the [original metric record](results/reported_metrics.csv) is retained separately.

## Code

```text
filtering.py                    Transcript acceptance rules and rejection reasons
fusion.py                       Zero-fill + concatenate features and missing flags
fusion_head.py                  Minimal PyTorch binary MLP
demo.py                         Synthetic example without patient data
tests/test_core.py              Filtering and missing-feature checks
research/multimodal_ssl_train.py Original ASR-gated audio-video SSL research script
results/auc_gains.csv           Unimodal-to-multimodal AUC comparisons
results/reported_metrics.csv    Original experiment record
docs/assets/                   Workflow and result figures (SVG + PNG)
```

### Run the small example

```bash
python -m pip install numpy
python demo.py
python -m unittest discover -s tests -v
```

This demonstrates missing-feature handling without downloading a model or dataset. Features must already be pooled into aligned `[N, D]` matrices; the A/V/T missingness array has shape `[N, 3]`.

For a trainable classification head, install `requirements.txt` and use:

```python
import torch
from fusion import build_fusion_features
from fusion_head import ConcatClassifier

# audio, video, text: aligned pooled feature matrices
# missing: [N, 3], in audio/video/text order; 1 means missing
features = build_fusion_features(audio, video, text, missing)
model = ConcatClassifier(input_dim=features.shape[1])
logits = model(torch.from_numpy(features))  # [N, 2]
```

The root-level code is a compact reference example. The original SSL script is retained separately and requires its project-local loaders and model helpers; see [research/README.md](research/README.md). It trains audio-video alignment and is not the complete supervised A/V/T experiment behind the table. Raw interviews, transcripts, participant labels, pretrained weights and trained checkpoints are not included.

To regenerate the figures, install Pillow and run `python docs/build_figures.py` (the renderer currently uses the macOS Arial font paths).
