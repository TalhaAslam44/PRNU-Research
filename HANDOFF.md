# HANDOFF: state of the project on 29 Oct 2026

Read this first in a new Claude session (any account). Start the new session in `/home/talhaaslam/PRNU-Research`
and say: "Read HANDOFF.md and continue the project."

## Who / what
* Talha Aslam, MS thesis (FAST-NUCES Islamabad): *Robust Video Source Provenance: Distinguishing Synthetic vs. Organic
  Sensor Noise to Counter Anti-Forensic Injection Attacks*. Supervisor Dr. Hasan Mujtaba, co-supervisor Dr. Muhammad Ishtiaq.
* HVPF = Hybrid Video Provenance Framework: classifies a video clip as **organic / manipulated (PRNU removal, injection,
  denoise) / synthetic (AI)** with PRNU + Noiseprint++ + FFT/TSNCS features, SVM/XGBoost/GRU/CNN/Transformer, a ViT-S/16 on
  SRM noise residuals, and stacking. RQ1 features, RQ2 processing effects, RQ3 hybrid vs conventional PCE rule.
* Repo: https://github.com/TalhaAslam44/PRNU-Research (public). `main` = original project (60/20/20 split, phases 1-4);
  branch `hvpf-2` = 80/20 study + trained camera identification (folder `~/hvpf-2`, separate copy, frames symlinked).
* Mid-evaluation (Phase 3 results only): `report/mid_evaluation_report.pdf`, poster `report/mid_evaluation_poster.pptx`
  (built by `report/poster/build_poster.py`). All phases compared in `report/phase_comparison.pdf`.

## User preferences (keep following)
* Always tell the user **which file to run, where and how**. The user prefers ONE bash command to paste into the VS Code
  terminal (`setsid nohup ./scripts/... > data/logs/....log 2>&1 < /dev/null &`) plus a one-line progress check.
* Long jobs run detached (`setsid nohup ... &`) and write logs to `data/logs/`; the user has a single GPU (RTX 3080), keep one
  GPU job at a time except Step 8 next to the ViT.
* Never handle passwords. The user logs in to GitHub themselves (`gh auth`); commit email is the noreply address
  `99859533+TalhaAslam44@users.noreply.github.com`. Do not push without being asked. Do not delete user data.
* Never commit Noiseprint++ weights (nonprofit licence) or `data/`. Keep the Phase 1-3 result backups (`data/phase1|2|3/`).
* Be honest about weak results (the user values honest benchmarks; three shortcuts were found and removed).

## Paths
* Data disk (must be mounted): `/media/talhaaslam/Disk/datasets/{Vision-Dataset,FloreView,synthetic}`; ACID at
  `~/Data/datasets/VideoACID/Full`. Frames: `data/frames/` (ATTACK/SYNTH symlink to `~/PRNU-Research-data/frames/`).
* Results: `data/results/{models,metrics,camera_id,prnu_baseline}`; backups `data/phase1`, `data/phase2`, `data/phase3`.
* Python: `.venv` (timm, xgboost, torch). LaTeX: `~/.local/bin/tectonic`. 7-Zip: `~/.local/bin/7zz`.

## Pipeline (scripts/)
01 inventory, 02 splits (by camera model; 61 groups; 60/20/20 + 5 folds; FloreView/GenBuster test-only), 03 I-frames,
04 PRNU baseline, 05 attacks (VISION + ACID, since Phase 4), 06 synthetic (AI clips, Khan-style PRNU injection, 8 generators),
07/07b features, 08 models, 08b ViT (`hvpf/vit.py`), 08c calibration/ensembles/stacking/`+safe` rule, 09 evaluation,
10 camera identification by training. Runners: `run_phase3.sh`, `run_phase4.sh`, `run_models.sh`. Reports:
`make_results_report.py` (currently writes the Phase 3 report), `make_hvpf2_report.py` (hvpf-2 branch).

## Phases and results (realistic protocol, 5-fold, test cameras never seen in training, 5 frames, macro F1 / AUC genuine-vs-not)
| Phase | Clips | Best | F1 | AUC | Note |
|---|---|---|---|---|---|
| 1 | ~8,000 | GRU | 0.852 | 0.930 | inflated: all WhatsApp clips were genuine (shortcut) |
| 2 | 24,599 | XGBoost | 0.792 | 0.899 | honest benchmark |
| 3 | 27,583 | stacking | 0.851 | 0.941 | + ViT, ensembles, PRNU injected into AI clips; PCE rule AUC 0.549 |
| hvpf-2 | 27,583 | stacking | 0.850 | 0.941 | 80/20: single models +0.006 F1 only; unseen sources F1 0.893 |
| 4 | 46,799 | stacking | 0.814 | 0.902 | harder, bigger test set (see below) |
| **5** | 46,799 | stacking, changes 1+2+4 | **0.826** (10 frames **0.842**) | 0.911 (0.922) | same data as Phase 4; see Phase 5 |

### Phase 4 (finished 20 Oct 22:08; data added: 9,504 ACID attacks, 3,712 genuine platform copies, 6,000 AI clips from SVD/Mora/
Text2Video-Zero/Pika; ModelScope and VideoCrafter2 downloaded but unused, < 480 px)
* Realistic 5-fold: stacking F1 0.814 / AUC 0.902 / EER 0.202; ViT 0.777; ens_all 0.780; XGBoost 0.758; PCE rule AUC 0.543.
* Encoder-controlled: stacking 0.762 (Phase 3: 0.816). Unseen sources (FloreView + GenBuster): ViT 0.843, stacking 0.839 / AUC 0.955.
* 10 frames beat 5 frames: stacking 0.827 (vs 0.814), ViT 0.797 (vs 0.777), +safe 0.803.
* Injected-PRNU AI clips accepted as genuine: PCE rule 83.7%, stacking 0.0% (n = 4,184).
* **Phase 4 scores are lower than Phase 3 mainly because the test set is harder**, not because models got worse: ACID attacks
  and platform copies of ACID clips (~7 I-frames each) are the weakest conditions (stacking 26-42% for ACID removal_opt+YT,
  re-encode+WhatsApp, injection+YT). Not yet confirmed with a same-clips comparison; do that before claiming it.
* **False accusations were NOT fixed**: stacking accepts only 72.2% of genuine clips (Phase 3: 78.9%); the `+safe` rule gets
  93.8% genuine accepted but manipulated recall falls to 48.2% (AI 98.1%).
* Camera identification (Step 10, scene-grouped, 30 frames): hybrid PRNU+ViT VISION 67.4%, Video-ACID 94.9%, FloreView 90.1%
  (trained ViT 49.7 / 90.9 / 89.0; PRNU 55.9 / 57.5 / 25.9; PDC-ViT reported 94.3 / 74.77). Learned fusion was no better than
  the simple hybrid. FloreView dropped ~4 points vs hvpf-2 after the near-duplicate scene fix (leakage was real).

### Phase 5 (26-27 Oct; experiment in NEW files only: `hvpf/phase5.py`, `scripts/p5_*.py`, `scripts/run_phase5.sh`;
reads `data/phase4/features`, writes `data/phase5/`; table `data/phase5/metrics/phase5_comparison.txt`)
* Change 1 processing-matched training weights (CRF-20 condition: attack share of training weight 0.90 -> 0.63),
  2 reliability-aware stacking (`stack_rel`), 3 three-zone decision accept / inconclusive / accuse (5% error caps on
  validation), 4 camera-calibrated PRNU evidence (clip PCE vs the claimed camera's median over 10 genuine natives,
  leave-one-video-out). Re-derived Phase 4 stacking matches Phase 4 exactly (check).
* Only change 4 really raises scores: unseen cameras 10 frames F1 0.827 -> 0.836 (4) / 0.842 (1+2+4, Phase 4 ViT), AUC 0.913 ->
  0.922; unseen datasets F1 0.902 -> 0.925, AUC 0.983 -> 0.991, FloreView genuine 82 -> 88%. On Phase 3's own test clips
  (22,867): 1+2+4 10 frames F1 0.854 vs Phase 3 0.852. Three-zone: ~7% genuine accused, ~6% attacks accepted, ~70% decided.
* Negative results: block-level compression-aware correlation predictor (Chen et al. 2008) has no signal on video
  (r = 0.15), dropped. Matched ViT (`p5_vit.py`, overnight run): CRF-20 genuine 58 -> 72% but F1 0.842 -> 0.833; keep the
  Phase 4 ViT. NOTE: that run overwrote `fusion_c4m_*` / `fusion_p5_*` with the matched-ViT version; to reproduce 0.842,
  move `data/phase5/models/vit` aside and re-run `python scripts/p5_fusion.py && python scripts/p5_compare.py`.

## Git state
`main` holds everything up to Phase 5 (push it if `git status` says "ahead"). `hvpf-2` branch is pushed.

## Moving to a new PC
Restore under the same user name `talhaaslam` and the same folders: parquet/CSV files store absolute frame paths
(`/home/talhaaslam/PRNU-Research/data/frames/...`), ATTACK/SYNTH frames are symlinks to `~/PRNU-Research-data/frames/`,
`config.yaml` points at `/media/talhaaslam/Disk/datasets` and `~/Data/datasets/VideoACID/Full` (`~/Data` -> the data drive).
Not backed up (recreate): `.venv` (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`), tectonic, 7zz, gh,
ffmpeg. Noiseprint++ weights are in `third_party/noiseprintpp/` (inside the project backup; never commit them).

## Suggested next steps
1. Significance: bootstrap CIs / paired test for Phase 4 vs Phase 5 (gains ~0.015 F1, fold std ~0.012).
2. Phase 5 report PDF (changes, tables, negative results, same-clips comparison with Phase 3).
3. For publication: same-protocol baselines (PDC-ViT-style model, one published AI-video detector), an adaptive attack, real
   WhatsApp/YouTube uploads.
