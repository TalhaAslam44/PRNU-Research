"""Phase 5, changes 2 and 3 - reliability-aware fusion and the inconclusive threshold (CPU, ~2 min).

For the Phase 4 predictions (data/phase4/models, as trained) and, when they exist, the Phase 5
processing-matched ones (data/phase5/models; ViT from data/phase5/models/vit if trained, else the
Phase 4 ViT), per split:
  stack, stack+safe          exactly Step 8c (scripts/08c_ensemble.py derive(), imported unchanged)
  stack_rel, stack_rel+safe  reliability-aware stacking (hvpf.phase5.rel_design / fit_rel_stack)
  tau_lo, tau_hi             three-zone (accept / inconclusive / accuse) thresholds of stack and
                             stack_rel, set on validation (hvpf.phase5.three_zone_thresholds)
Writes data/phase5/models/fusion_<p4|p5>_<protocol>_<scheme>_w<N>.parquet (with the PCE baseline rows).

Usage:
    python scripts/p5_fusion.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.model_data import CLASSES, load

step8c = phase5.load_step("08c_ensemble.py")
P = [f"p_{c}" for c in CLASSES]


def inputs(source, protocol, scheme, window):
    """Base model predictions (incl. the ViT) of Phase 4 or Phase 5, and where the ViT came from."""
    name = f"{protocol}_{scheme}_all_w{window}.parquet"
    vit_name = f"{protocol}_{scheme}_w{window}.parquet"
    if source == "p4":
        pred = pd.read_parquet(phase5.PHASE4 / "models" / name)
        pred = pred[~pred.model.str.fullmatch(step8c.DERIVED)]
        vit = phase5.PHASE4 / "models" / "vit" / vit_name
    else:
        path = phase5.PHASE5 / "models" / name
        if not path.exists():
            return None, None
        pred = pd.read_parquet(path)
        vit = phase5.PHASE5 / "models" / "vit" / vit_name
        vit = vit if vit.exists() else phase5.PHASE4 / "models" / "vit" / vit_name
    if vit.exists():
        pred = pd.concat([pred, pd.read_parquet(vit)], ignore_index=True)
    return pred, vit


def rel_rows(split, rel):
    """stack_rel and stack_rel+safe rows of one split; rel = reliability indicators by sample_id."""
    split = split[split.model != "baseline"]
    models = sorted(split.model.unique())
    meta = split[split.model == models[0]].drop(columns=["model", "score_organic", "threshold", *P],
                                                errors="ignore").set_index("sample_id")
    probs = {m: split[split.model == m].set_index("sample_id").reindex(meta.index)[P].to_numpy() for m in models}
    models = [m for m in models if not np.isnan(probs[m]).any()]
    y = meta.label.map(CLASSES.index).to_numpy()
    val = (meta.part == "val").to_numpy()
    r = rel.reindex(meta.index).to_numpy(float)
    med = np.nanmedian(r[val], 0)
    r = np.where(np.isnan(r), med, r)
    r = (r - r[val].mean(0)) / (r[val].std(0) + 1e-9)
    z = phase5.rel_design(np.concatenate([step8c.logp(probs[m]) for m in models], 1), r)
    lr, c = phase5.fit_rel_stack(z[val], y[val], meta.claimed_device.to_numpy()[val])
    out = {"stack_rel": lr.predict_proba(z)}
    genuine = val & (y == 0)
    lp = step8c.logp(out["stack_rel"][genuine])
    boost = next((b for b in np.arange(0, 8.01, 0.05)
                  if ((lp + [b, 0, 0]).argmax(1) == 0).mean() >= step8c.SAFE_RECALL), 8.0)
    out["stack_rel+safe"] = step8c.shift(out["stack_rel"], np.array([boost, 0.0, 0.0]))
    print(f"    stack_rel over {models}, C = {c}")
    return [meta.reset_index().assign(model=n, **{col: p[:, i] for i, col in enumerate(P)}) for n, p in out.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="realistic")
    ap.add_argument("--windows", nargs="*", type=int, default=[5, 10])
    args = ap.parse_args()
    out_dir = phase5.PHASE5 / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for window in args.windows:
        per_video, _ = load(phase5.PHASE4 / "features", window)
        rel = per_video.set_index("sample_id")[phase5.RELIABILITY]
        for scheme in ("folds", "fixed"):
            for source in ("p4", "p5"):
                pred, vit = inputs(source, args.protocol, scheme, window)
                if pred is None:
                    continue
                print(f"{source} {args.protocol}/{scheme}/w{window} (ViT: {vit.relative_to(phase5.ROOT)})", flush=True)
                rows = [pred[pred.model == "baseline"]]
                for name, split in pred.groupby("split_name"):
                    print(f"  {name}")
                    fused = [r for r in step8c.derive(split) if r.model.iat[0] in ("stack", "stack+safe")]
                    fused += rel_rows(split, rel)
                    for r in fused:
                        if not r.model.iat[0].endswith("+safe"):
                            v = (r.part == "val").to_numpy()
                            r["tau_lo"], r["tau_hi"] = phase5.three_zone_thresholds(
                                r.loc[v, P].to_numpy(), r.label[v].map(CLASSES.index).to_numpy())
                    rows += fused
                out = pd.concat(rows, ignore_index=True).assign(vit_source=str(vit.relative_to(phase5.ROOT)))
                path = out_dir / f"fusion_{source}_{args.protocol}_{scheme}_w{window}.parquet"
                out.to_parquet(path, index=False)
                print(f"  wrote {path.relative_to(phase5.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
