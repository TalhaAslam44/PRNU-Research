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
| 1 Inventory | `scripts/01_inventory.py` | `data/metadata/inventory.csv`, `inventory_devices.csv` (Chapter 5 table) | done (VISION + ACID M00) |
| 1b ACID streaming | `scripts/01b_acid_stage.py` | `inventory_acid.csv`, `data/frames/ACID/` | running (40 videos per device) |
| 2 Splits | `scripts/02_splits.py` | `model_splits.csv`, `videos.csv` (split, fold, role per video) | done |
| 3 I-frames | `scripts/03_extract_frames.py` | `data/frames/<dataset>/<device>/<content_id>__<version>.npy` | VISION done (1,480 videos, 39k frames, 8.5 GB) |
| 4 PRNU baseline | `scripts/04_prnu_baseline.py` | `data/fingerprints/`, `data/results/prnu_baseline/` | VISION done |
| 5 Attacks | - | | todo |
| 6 Synthetic class | - | | todo |
| 7 Features | - | | todo (Noiseprint needs the GPU) |
| 8-9 Models, evaluation | - | | todo |

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

## Findings so far: PCE baseline, all 35 VISION devices (closed set, threshold 60)

`data/results/prnu_baseline/summary.csv`, `summary_by_rotation.csv`, and `scores.parquet`.

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
* Disk: about 50 GB free on a spinning disk (~90 MB/s). ACID (177 GB of archives) and FloreView
  (~95 GB of videos, URL list in `data/metadata/floreview_video_urls.txt`) must be streamed:
  download or unpack a device, extract frames, delete the videos.
