"""Thesis ablation, part 1 - the final HVPF's feature models with a feature group left out or used alone (GPU).

The final system (changes 1 + 2 + 4, Phase 4 ViT, 10 frames) trains its five feature models (SVM, XGBoost, GRU,
CNN, Transformer) exactly as scripts/p5_c4_models.py --matched does: Phase 4 features + camera-calibrated PRNU
evidence, processing-matched training weights, Step 8's grids. Here the same training is repeated with
    --drop <group>   every feature group except this one   (leave-one-out)
    --only <group>   this feature group alone              (single cue)
    (neither)        all groups again                      (re-training noise of the reference)
Groups: prnu, noiseprint, tsncs, resid, fft. The calibration columns are PRNU evidence, so they go with prnu.
The ViT is a pixel model (SRM residuals), not a feature group; it is handled in scripts/thesis_ablation.py.
Nothing of Phases 3-5 is changed: reads data/phase4/features, writes
data/thesis/ablation/models/<tag>/realistic_<scheme>_all_w<N>.parquet (tag = no_<g> | only_<g> | all_rerun).

    .venv/bin/python scripts/thesis_ablation_models.py --drop prnu --scheme folds
    .venv/bin/python scripts/thesis_ablation_models.py --only noiseprint --scheme fixed
    .venv/bin/python scripts/thesis_ablation_models.py --drop fft --scheme fixed --models xgb --out-dir /tmp/check
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.config import load_config
from hvpf.model_data import CLASSES, FRAME_GROUPS, feature_columns, frame_columns, load, protocol_mask, split_masks

p5_models = phase5.load_step("p5_models.py")
OUT = phase5.ROOT / "data" / "thesis" / "ablation" / "models"
MODELS = ["svm", "xgb", "gru", "cnn", "transformer"]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--drop", choices=list(FRAME_GROUPS), help="feature group to leave out")
    g.add_argument("--only", choices=list(FRAME_GROUPS), help="feature group to use alone")
    ap.add_argument("--scheme", choices=["fixed", "folds"], default="folds")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out-dir", type=Path, help="default data/thesis/ablation/models/<tag>")
    args = ap.parse_args()

    tag = f"no_{args.drop}" if args.drop else f"only_{args.only}" if args.only else "all_rerun"
    groups = [args.only] if args.only else [g for g in FRAME_GROUPS if g != args.drop]
    out_dir = args.out_dir or OUT / tag
    path = out_dir / f"realistic_{args.scheme}_all_w{args.window}.parquet"
    if path.exists():
        print(f"{path} exists - skipped")
        return

    cfg = load_config()
    per_video, frames = load(phase5.PHASE4 / "features", args.window)
    per_video = per_video[protocol_mask(per_video, "realistic")].reset_index(drop=True)
    per_video = pd.concat([per_video, phase5.camera_calibration(per_video)], axis=1)
    vcols = feature_columns(per_video, groups) + (phase5.CALIBRATION if "prnu" in groups else [])
    fcols = frame_columns(frames, groups)
    y = per_video.label.map(CLASSES.index).to_numpy()
    condition = phase5.processing_condition(per_video, cfg["attacks"]["crf"])
    print(f"ablation {tag} realistic/{args.scheme}/w{args.window}: groups {groups}, {len(per_video)} samples, "
          f"{len(vcols)} clip features, {len(fcols)} frame features", flush=True)

    rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        sw = phase5.matched_weights(y[tr], condition[tr])
        rows += p5_models.fit_split(per_video, frames, vcols, fcols, y, tr, va, te | cross | va, cross, name,
                                    args.window, args.models, sw)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
