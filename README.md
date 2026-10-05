# HVPF: Hybrid Video Provenance Framework

Thesis pipeline: separate **organic** sensor noise from **manipulated** (PRNU removal or injection)
and **synthetic** (AI-generated) noise in videos, using PRNU + Noiseprint + FFT + TSNCS features.

## Setup

```bash
git clone --recursive <this repo>
cd PRNU-Research
git -C third_party/prnu-python apply ../prnu-python-numpy2.patch   # NumPy 2 fix for the PRNU library
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Needs `ffmpeg`/`ffprobe` on the PATH. Datasets are not included: download
[VISION](https://lesc.dinfo.unifi.it/VISION/), [Video-ACID](https://research.coe.drexel.edu/shared/ece/misl/VideoACID/)
and [FloreView](https://lesc.dinfo.unifi.it/FloreView/), then set their folders in `config.yaml`.
All other settings are in `config.yaml` too. Run every script from the project root with the venv active.

## Pipeline

| Step | Script | Output | Status |
|---|---|---|---|
| 1 Inventory | `scripts/01_inventory.py` | `data/metadata/inventory.csv`, `inventory_devices.csv` (Chapter 5 table) | done: VISION 1,914 + ACID 1,840 videos |
| 1b ACID streaming | `scripts/01b_acid_stage.py` | `inventory_acid.csv`, `data/frames/ACID/` | done: 46 devices x 40 videos; Nokia 6.1 (M23, 2 devices) unreadable by FFmpeg 6.1 |
| 2 Splits | `scripts/02_splits.py` | `model_splits.csv`, `videos.csv` (split, fold, role per video) | done |
| 3 I-frames | `scripts/03_extract_frames.py` | `data/frames/<dataset>/<device>/<content_id>__<version>.npy` | done: VISION 1,480 videos (39k frames), ACID 1,760 videos (~6 I-frames each) |
| 4 PRNU baseline | `scripts/04_prnu_baseline.py` | `data/fingerprints/`, `data/results/prnu_baseline/<DATASET>/` | done (VISION, ACID) |
| 5 Attacks | `scripts/05_attacks.py` | `data/frames/ATTACK/`, `data/metadata/attacks.csv` | running: 400 VISION videos x 11 variants |
| 6 Synthetic class | `scripts/06_synthetic.py` | `data/frames/SYNTH/`, `data/metadata/synthetic.csv` | done: 1,492 clips (892 GenVidBench, 600 GenBuster) |
| 7 Features | `scripts/07_features.py` | `data/features/{samples.csv,frames.parquet,videos.parquet}` | written and tested; full run queued (`scripts/07b_noiseprint.py` for Noiseprint, GPU) |
| 8 Models | `scripts/08_models.py`, `scripts/run_models.sh` | `data/results/models/*.parquet` | written and tested; full run queued |
| 9 Evaluation | `scripts/09_evaluate.py` | `data/results/metrics/{summary,per_variant,ablation}.csv`, `results.md` | written and tested |

Every script can be re-run; finished work (frames, fingerprints, ACID archives) is skipped.

## Design decisions (for the methodology chapter)

* **Split unit is the camera model**, not just the device. Devices of the same model
  (e.g. D05/D14/D18 iPhone 5c), including VISION/ACID pairs such as the Galaxy S5, stay together,
  because Noiseprint is a model-level trace. All versions of a recording (native/YT/WA, later attacked)
  share a split. FloreView is the cross-dataset test set only. A 60/20/20 split leaves ~5 VISION test
  cameras, so `videos.csv` also has a 5-fold grouped `fold` column for mean +- std reporting.
* **Roles**: VISION native flat videos are `reference` (they estimate K); their YT/WA copies are unused
  so the reference never leaks into evaluation; natural videos are `sample`. ACID has no flat videos:
  the first 10 videos per device are reference, the next 30 samples.
* **Frames**: up to 30 I-frames per video, evenly spaced; luma only; 480x480 center crop (not 512,
  because D04 and D23 record at 800x480 and 640x480). Frames are kept in **sensor orientation**
  (`-noautorotate`). YouTube/WhatsApp copies bake the rotation into the pixels and are downscaled,
  so they are rotated back and rescaled to the native size before cropping.
* **Stabilized devices** (D02 D05 D06 D10 D12 D14 D15 D18 D19 D20 D25 D29 D32 D34; VISION paper +
  Mandelli et al., TIFS 2020) are kept and flagged; results are reported separately.
* **Errata / broken files**: the 6 misplaced videos from the VISION README and
  `D33_V_indoorYT_panrot_0001.mp4` (truncated download, "moov atom not found") are excluded.
* **PRNU code**: Politecnico di Milano `prnu-python` in `third_party/`, patched for NumPy 2
  (`third_party/prnu-python-numpy2.patch`). `hvpf/prnu_utils.py` adds a grayscale fingerprint
  (their multi-image function assumes RGB) that matches theirs to 6e-7, plus a PCE evaluated at zero
  shift (`pce0`) next to the usual blind peak search (`pce`).

## Step 5: manipulated class

Targets are the 425 natural native VISION videos. The first 30 s of each are decoded once
(sensor orientation, analysis crop, yuv420p; only luma is attacked) and re-encoded with
libx264 (CRF 20, one I-frame per second, so ~30 I-frames like the organic samples):

| Variant | What | Label | Claimed device |
|---|---|---|---|
| `reencode` | same encoder, no attack (control) | organic | own |
| `removal_a{0.25,0.5,1}` | Y' = Y - a*Y*K_own | manipulated | own |
| `removal_opt` | white-box: a tuned per clip until NCC with the analyst's reference is ~0 | manipulated | own |
| `injection_a{0.25,0.5,1}` | Y' = Y + a*Y*K_victim | manipulated | victim |
| `denoise_a{1,2,4}` | hqdn3d at 1/2/4 x default strength | manipulated | own |

* `alpha` is in units of a 1% PRNU (attacker fingerprints are rescaled to std 0.01).
* The control goes through the same encoder, so encoder traces cannot separate the classes.
* Attacker fingerprints never reuse the analyst's flat-field reference: K_own comes from the
  other half of the device's natural videos, K_victim from all natural videos of a
  non-stabilized victim of another model in the same fold.
* Removal over-subtracts above a ~ 0.3-0.5: PCE turns strongly negative (e.g. -4765 at a = 1),
  itself a detectable trace. `removal_opt` is the hardest case for the PCE baseline (PCE ~ 0).
  A natural-video-only attacker cannot calibrate it: VISION shoots each device's natural videos
  at the same places, so two K estimates share scene residue and alpha comes out ~5x too small.

Check on D01 / D08 (PCE0 with the analyst's reference, own device / victim):

| Variant | D01 own | D01 victim | D08 own | D08 victim |
|---|---|---|---|---|
| reencode | 195 | 0 | 1381 | 1 |
| removal 0.25 / 0.5 / 1 | 12 / -199 / -1296 | | 644 / 3 / -4765 | |
| removal_opt | 0.1 (a=0.30) | | 0.0 (a=0.52) | |
| injection 0.25 / 0.5 / 1 | 146 / 91 / 18 | 49 / 315 / 919 | 1343 / 1185 / 582 | 39 / 275 / 1357 |
| denoise 1 / 2 / 4 | 103 / 75 / 38 | | 1124 / 918 / 454 | |

## Step 6: synthetic class

| Source | Generators | Clips used | Frames per clip | Role |
|---|---|---|---|---|
| [GenVidBench](https://huggingface.co/datasets/jian-0/GenVidBench) | Sora (1080p, ~17 s), Kling (720p, 5-10 s), CogVideo (480x480, 4 s), OpenSora (512x512, 2 s) | up to 300 each | ~19 / ~10 / 5 / 2 | train/val/test |
| [GenBuster-200K-mini](https://huggingface.co/datasets/l8cv/GenBuster-200K-mini) | CogVideoX, EasyAnimate, HunyuanVideo, LTX-Video (1024x1024 HEVC, 5 s) | 150 each | 6 | unseen-source test only |

* Generator files hold almost no I-frames (Kling T2V: 1 per clip), so synthetic clips go through
  the same libx264 encoder as the attacks and the organic `reencode` control (one I-frame per second,
  same crop). Frame type, spacing and encoder are then identical across classes; denser sampling for
  short clips was rejected because closer frames inflate TSNCS (a false "synthetic" cue).
* Each clip claims a random camera (VISION or ACID) of a random fold; its split follows that camera.
  GenBuster's real clips are not used (no camera reference).
* Archives are stored on the second partition (`/media/talhaaslam/Disk/datasets/synthetic`),
  re-downloadable with its `download.sh`.

## Step 7: features

One denoising pass per frame (`hvpf/features.py`) gives, per frame: NCC and zero-shift PCE of the
residual W with I*K of the claimed camera, residual std/skew/kurtosis, FFT statistics of W (flatness,
high-frequency share, peak ratio, periodic-peak count, 8-px block energy) and TSNCS (NCC of consecutive
residuals). Per video: the clip's MLE fingerprint from all frames (`v_*`) and from the first 5 frames
(`w_*`, comparable across short and long clips), scored against the claimed camera and against every
other known camera (`*_max_other`: injection leaves the true camera's PRNU next to the victim's),
plus mean/std of the per-frame values. Check on the Step 5 test clips: injection a=0.25 keeps the true
camera at PCE 146 (`max_other`); strong removal raises TSNCS from ~0.002 to 0.165.

**Step 7b (Noiseprint++, GPU):** `third_party/get_noiseprintpp.sh` fetches GRIP-UNINA's network and
weights (TruFor, pinned commit; nonprofit licence, not committed here). On our luma-only video I-frames
Noiseprint++ does not attribute cameras (spatial or spectral similarity: ~20% top-1 on 10 VISION cameras,
chance 10%), so it contributes output statistics (map std, kurtosis, FFT statistics) plus a weak spectral
similarity to the claimed camera; the ablation decides what it adds.

**Encoder confound and the controlled protocol.** Every clip we encode with libx264 (attacks, the
`reencode` control, synthetic clips) has its macroblock grid at the crop origin; camera originals do not.
Grid-sensitive features see this (Noiseprint 8-px block energy ~4 vs 0.65 for organic WhatsApp clips).
Steps 8-9 therefore report two protocols: *realistic* (all organic samples vs manipulated/synthetic) and
*encoder-controlled* (only x264 clips: `reencode` controls vs attacks vs synthetic), where encoder and grid
are identical across classes and only genuine noise traces can separate them.

## Steps 8-9: models and evaluation

* Inputs come from the first 5 I-frames of every clip (VISION has ~30, ACID ~6, synthetic 2-20), so
  clip length cannot act as a shortcut.
* Models: the conventional PCE rule (organic if PCE0 with the claimed camera > 60), SVM, XGBoost,
  and GRU / 1D-CNN / Transformer on the per-frame sequences joined with the clip features
  (Adam, lr 0.001, L2 weight decay, class-balanced loss, early stopping on validation macro F1).
* Metrics: PVA (3-class accuracy, thesis 6.8), macro precision/recall/F1, one-vs-rest AUC,
  authentic-vs-not AUC and EER (where the PCE rule is compared), per-variant detection rates for RQ2
  (native/YouTube/WhatsApp, attack type and alpha, generator, stabilization), GenBuster as unseen
  generators, and an ablation that drops one feature group at a time (5-fold CV).
* `scripts/run_models.sh` runs every protocol x scheme plus the ablations, then Step 9.

## Findings so far: ACID PCE baseline (44 devices, 20 natural reference videos each)

Attribution 50.7% (pce0), AUC 0.80, far below VISION: no flat videos, ~6 I-frames per clip, and
many 2017-18 devices stabilize electronically. Per device it is bimodal: DSLRs, action cameras and
Samsung phones reach 95-100% (match PCE 100-3000), while 19 devices (Pixel 1/2, iPhone 8 Plus,
LG Q6, Zenfone 3, compact cameras/camcorders with digital IS) have median match PCE < 5. They are
listed as `acid.prnu_weak_devices` (analysis flag; splits unchanged).

## Findings so far: PCE baseline, all 35 VISION devices (closed set, threshold 60)

`data/results/prnu_baseline/VISION/summary.csv`, `summary_by_rotation.csv`, and `scores.parquet`.

| Version | Devices | Attribution (pce0) | AUC pce0 | AUC blind PCE | EER pce0 |
|---|---|---|---|---|---|
| Native | non-stabilized | 100.0% | 1.000 | 0.999 | 0.000 |
| Native | stabilized | 31.9% | 0.738 | 0.564 | 0.344 |
| YouTube | non-stabilized | 84.7% | 0.972 | 0.812 | 0.084 |
| YouTube | stabilized | 20.4% | 0.638 | 0.489 | 0.437 |
| WhatsApp | non-stabilized | 60.7% | 0.868 | 0.713 | 0.214 |
| WhatsApp | stabilized | 7.8% | 0.571 | 0.464 | 0.453 |

* Native non-stabilized is perfect, matching published VISION results, so the pipeline is sound.
* Stabilized devices break PRNU, as expected; report them separately.
* PCE at zero shift (`pce0`) is much stronger than the blind peak search on social copies;
  use it as the conventional baseline for RQ3.
* **Social copies of rotated recordings lose PRNU entirely** (WhatsApp AUC 0.52 vs 0.92 for
  unrotated sources). WhatsApp's rotation path also rescales by ~0.5-1% and shifts ~3-5 px;
  registering frames to the native recovers the match (e.g. PCE 44 -> 213). This is a useful RQ2
  point: geometry, not compression, breaks PRNU here. `scores.parquet` has a `source_rotation` column.

## Environment notes

* GPU: the RTX 3080 is idle because the machine booted kernel 6.8.0-53, and the
  `nvidia-driver-595` modules exist only for 6.8.0-142/146. Reboot into 6.8.0-146
  (GRUB > Advanced options) before Step 7/8; until then `torch.cuda.is_available()` is False.
* `data/frames/ATTACK` is a symlink to `/media/talhaaslam/Disk/PRNU-Research-data/frames/ATTACK`
  (second partition, ~300 GB free). That partition is not in `/etc/fstab`: after a reboot,
  open it once in Files so it is mounted before running Step 5 or later steps.
* Disk: about 50 GB free on a spinning disk (~90 MB/s). ACID (177 GB of archives) and FloreView
  (~95 GB of videos, URL list in `data/metadata/floreview_video_urls.txt`) must be streamed:
  download or unpack a device, extract frames, delete the videos.
