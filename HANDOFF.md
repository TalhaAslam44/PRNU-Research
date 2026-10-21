# HANDOFF: state of the project on 21 Oct 2026

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
* Mid-evaluation presentation to a panel is coming: `report/mid_evaluation_report.pdf` (16 pages, architecture diagram,
  every step explained, panel Q&A). It still says Phase 4 is "running": update it with the Phase 4 results below.

## User preferences (keep following)
* Always tell the user **which file to run, where and how** (VS Code, Terminal -> Run Task...; tasks in `.vscode/tasks.json`).
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
| **4** | **46,799** | **stacking** | **0.814** | **0.902** | harder, bigger test set (see below) |

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

## Git state
`main` is ahead of `origin/main` by 3-4 commits (Phase 4 code, generators, mid-evaluation report, this file): NOT pushed.
`hvpf-2` branch is pushed.

## Suggested next steps
1. Same-clips comparison Phase 3 vs Phase 4 (like the earlier phase comparison) to separate "harder test" from "worse models".
2. Generate the Phase 4 report: `scripts/make_results_report.py` currently targets Phase 3 (reads `data/phase2` for comparison);
   point it at `data/phase3` for comparison and write `report/results_phase4.tex/pdf`; update the mid-evaluation report's Phase 4 rows.
3. Fix the false accusations: more/better genuine re-encoded + platform-copy training data (esp. ACID, few I-frames), use 10 frames
   as the main setting, consider an "uncertain" output.
4. For publication: same-protocol baselines (PDC-ViT-style model, one published AI-video detector), an adaptive attack, real
   WhatsApp/YouTube uploads, significance tests / bootstrap CIs.
5. Push to GitHub when the user asks.
