"""Phase 5, change 4 - Step 8 models with camera-calibrated PRNU evidence (GPU, ~20 min per run).

The Phase 4 clip features plus hvpf.phase5.CALIBRATION: the claimed camera's usual genuine PRNU score
(median of --calib-videos genuine native videos of that camera, never the clip's own source video)
and the clip's score relative to it. Models, splits and grids are Step 8's
(scripts/p5_models.fit_split); --matched adds change 1's processing-matched training weights.
Writes data/phase5/models_c4[m]/<protocol>_<scheme>_all_w<N>.parquet (scripts/p5_fusion.py and
scripts/p5_compare.py pick them up).

Usage:
    python scripts/p5_c4_models.py --scheme folds --window 5
    python scripts/p5_c4_models.py --scheme folds --window 5 --matched
    python scripts/p5_c4_models.py --scheme fixed --models baseline xgb --out-dir /tmp/check   # quick check
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["realistic", "controlled"], default="realistic")
    ap.add_argument("--scheme", choices=["fixed", "folds"], default="folds")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--matched", action="store_true", help="also change 1 (processing-matched training weights)")
    ap.add_argument("--calib-videos", type=int, default=10, help="genuine reference videos per camera for calibration")
    ap.add_argument("--models", nargs="*", default=["baseline", "svm", "xgb", "gru", "cnn", "transformer"])
    ap.add_argument("--out-dir", type=Path, help="default data/phase5/models_c4 (models_c4m with --matched)")
    args = ap.parse_args()

    cfg = load_config()
    per_video, frames = load(phase5.PHASE4 / "features", args.window)
    per_video = per_video[protocol_mask(per_video, args.protocol)].reset_index(drop=True)
    per_video = pd.concat([per_video, phase5.camera_calibration(per_video, args.calib_videos)], axis=1)
    groups = list(FRAME_GROUPS)
    vcols = feature_columns(per_video, groups) + phase5.CALIBRATION
    fcols = frame_columns(frames, groups)
    y = per_video.label.map(CLASSES.index).to_numpy()
    condition = phase5.processing_condition(per_video, cfg["attacks"]["crf"])
    print(f"Phase 5 change 4 {args.protocol}/{args.scheme}/w{args.window}{' + matched' if args.matched else ''}: "
          f"{len(per_video)} samples, {len(vcols)} clip features ({len(phase5.CALIBRATION)} calibration, "
          f"{args.calib_videos} videos per camera; {per_video.cal_pce0.isna().sum()} clips without calibration)",
          flush=True)
    print("median clip PRNU score relative to the claimed camera (dev_pce0), test-type clips:\n" + per_video.groupby(
        [per_video.label, per_video.variant.astype(str).str.split("+").str[0].str.split("_").str[0]])
          .dev_pce0.median().round(2).to_string(), flush=True)

    out_rows = []
    for name, tr, va, te in split_masks(per_video, args.scheme):
        cross = (per_video.split == "crossdataset").to_numpy() if args.scheme == "fixed" else np.zeros(len(y), bool)
        sw = phase5.matched_weights(y[tr], condition[tr]) if args.matched else None
        out_rows += p5_models.fit_split(per_video, frames, vcols, fcols, y, tr, va, te | cross | va, cross, name,
                                        args.window, args.models, sw)

    out_dir = args.out_dir or phase5.PHASE5 / ("models_c4m" if args.matched else "models_c4")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.protocol}_{args.scheme}_all_w{args.window}.parquet"
    out = pd.concat(out_rows, ignore_index=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
