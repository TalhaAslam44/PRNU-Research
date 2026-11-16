"""Thesis ablation, part 2 - ablation tables of the final HVPF with paired significance tests (CPU, ~15 min).

Final HVPF = stack_rel (reliability-aware stacking) over SVM, XGBoost, GRU, CNN, Transformer (camera-calibrated
PRNU evidence + processing-matched training, data/phase5/models_c4m) and the Phase 4 ViT, 10 frames.
Every row below is fused exactly like the final system (scripts/p5_fusion.rel_rows / Step 8c's stack, imported
unchanged) from saved predictions; only the feature runs of scripts/thesis_ablation_models.py are newly trained.

  A. Models     the final system without the ViT, without each feature model, and every model alone
  B. Components all 8 combinations of change 1 (matched training), 2 (reliability-aware fusion) and
                4 (camera calibration), always with the Phase 4 ViT
  C. Features   each feature group left out / used alone, with and without the ViT (runs that exist)
A reliability indicator is used by the fusion only if its feature group is in the run (res_std: resid,
fft_*: fft), so a left-out group cannot come back through the fusion layer.

Tests: unseen cameras (5-fold test, pooled) and unseen datasets (fixed split, FloreView + GenBuster).
Significance: camera-level paired bootstrap as in scripts/paper_stats.py (2,000 resamples of whole cameras,
both systems on the same resampled clips, two-sided p), each row against its reference row.
Writes data/thesis/ablation/{ablation.csv, ablation.txt} and report/thesis/ablation.md.

    .venv/bin/python scripts/thesis_ablation.py
"""
import contextlib
import io
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.model_data import CLASSES, FRAME_GROUPS, load

step8c = phase5.load_step("08c_ensemble.py")
p5_fusion = phase5.load_step("p5_fusion.py")
stats = phase5.load_step("paper_stats.py")

P = [f"p_{c}" for c in CLASSES]
WINDOW = 10
FEATURE_MODELS = ["svm", "xgb", "gru", "cnn", "transformer"]
NAMES = {"svm": "SVM", "xgb": "XGBoost", "gru": "GRU", "cnn": "CNN", "transformer": "Transformer", "vit": "ViT"}
GROUP_NAMES = {"prnu": "PRNU", "noiseprint": "Noiseprint++", "tsncs": "temporal consistency (TSNCS)",
               "resid": "residual statistics", "fft": "FFT"}
REL_GROUP = {"res_std_mean": "resid", "fft_block8_mean": "fft", "fft_hf_ratio_mean": "fft"}
ABL = phase5.ROOT / "data" / "thesis" / "ablation"
RUNS = {"none": phase5.PHASE4 / "models", "1": phase5.PHASE5 / "models", "4": phase5.PHASE5 / "models_c4",
        "1+4": phase5.PHASE5 / "models_c4m"}
B = 2000
CAM_METRICS = ["macro F1", "AUC genuine vs not", "genuine accepted %", "attacks caught %", "AI caught %",
               "injection attack (a=1) accepted %"]
SET_METRICS = ["macro F1", "AUC genuine vs not", "genuine accepted %", "AI caught %"]


def base_predictions(run_dir, scheme):
    """Feature-model predictions of a run plus the Phase 4 ViT (never the matched ViT), one split scheme."""
    path = Path(run_dir) / f"realistic_{scheme}_all_w{WINDOW}.parquet"
    if not path.exists():
        return None
    pred = pd.read_parquet(path)
    pred = pred[pred.model.isin(FEATURE_MODELS)]
    vit = pd.read_parquet(phase5.PHASE4 / "models" / "vit" / f"realistic_{scheme}_w{WINDOW}.parquet")
    return pd.concat([pred, vit[vit.model == "vit"]], ignore_index=True)


def fuse(pred, models, kind, rel):
    """One system's predictions: kind = 'raw' (a single model as trained), 'stack' (Step 8c) or 'stack_rel'."""
    pred = pred[pred.model.isin(models)]
    if kind == "raw":
        return pred
    rows = []
    for _, split in pred.groupby("split_name"):
        with contextlib.redirect_stdout(io.StringIO()):
            if kind == "stack":
                rows += [r for r in step8c.derive(split) if r.model.iat[0] == "stack"]
            else:
                rows.append(p5_fusion.rel_rows(split, rel)[0])
    return pd.concat(rows, ignore_index=True)


def as_system(d, part):
    d = d[d.part == part].set_index("sample_id")
    return d.assign(y=d.label.map(CLASSES.index), s=d.p_organic.to_numpy(), pred=d[P].to_numpy().argmax(1))


def paired(args):
    """Camera-level paired bootstrap of macro F1 and AUC: a (row) against b (reference), same clips."""
    a, b = args
    b = b.loc[a.index]
    y, cams = a.y.to_numpy(), a.claimed_device.to_numpy()
    pa, pb, sa, sb = a.pred.to_numpy(), b.pred.to_numpy(), a.s.to_numpy(), b.s.to_numpy()
    uniq, inv = np.unique(cams, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(0)
    d_f1, d_auc = [], []
    for _ in range(B):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        yy = y[idx]
        d_f1.append(stats.macro_f1(yy, pa[idx]) - stats.macro_f1(yy, pb[idx]))
        g = yy == 0
        d_auc.append(roc_auc_score(g, sa[idx]) - roc_auc_score(g, sb[idx]) if 0 < g.sum() < len(g) else np.nan)
    out = {}
    for name, d in (("F1", np.array(d_f1)), ("AUC", np.array(d_auc))):
        p = 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))
        out[f"d{name} lo"], out[f"d{name} hi"] = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
        out[f"p {name}"] = min(max(p, 1 / B), 1.0)
    return out


def main():
    warnings.filterwarnings("ignore")
    per_video, _ = load(phase5.PHASE4 / "features", WINDOW)
    rel_all = per_video.set_index("sample_id")[phase5.RELIABILITY]
    all_models = FEATURE_MODELS + ["vit"]

    # (section, row name, run directory, models, fusion kind, feature groups in the run, reference row)
    full = "full HVPF (reference)"
    full_nv = "feature models only (no ViT)"
    spec = [("A. Models", full, RUNS["1+4"], all_models, "stack_rel", list(FRAME_GROUPS), None),
            ("A. Models", full_nv, RUNS["1+4"], FEATURE_MODELS, "stack_rel", list(FRAME_GROUPS), full),
            ("A. Models", "ViT alone", RUNS["1+4"], ["vit"], "raw", list(FRAME_GROUPS), full)]
    spec += [("A. Models", f"without {NAMES[m]}", RUNS["1+4"], [x for x in all_models if x != m], "stack_rel",
              list(FRAME_GROUPS), full) for m in FEATURE_MODELS]
    spec += [("A. Models", f"{NAMES[m]} alone", RUNS["1+4"], [m], "raw", list(FRAME_GROUPS), full) for m in FEATURE_MODELS]
    for c1 in (0, 1):
        for c4 in (0, 1):
            for c2 in (0, 1):
                on = [n for n, f in (("matched training (1)", c1), ("reliability fusion (2)", c2),
                                     ("camera calibration (4)", c4)) if f]
                if len(on) == 3:
                    continue                                   # = the reference row
                run = RUNS["+".join(k for k, f in (("1", c1), ("4", c4)) if f) or "none"]
                spec.append(("B. Components", " + ".join(on) if on else "none (hybrid stacking, Phase 4)", run,
                             all_models, "stack_rel" if c2 else "stack", list(FRAME_GROUPS), full))
    spec.append(("C. Features", "all groups, re-trained (training noise)", ABL / "models" / "all_rerun", all_models,
                 "stack_rel", list(FRAME_GROUPS), full))
    for g in FRAME_GROUPS:
        rest = [x for x in FRAME_GROUPS if x != g]
        spec.append(("C. Features", f"without {GROUP_NAMES[g]}", ABL / "models" / f"no_{g}", all_models, "stack_rel", rest, full))
    for g in FRAME_GROUPS:
        rest = [x for x in FRAME_GROUPS if x != g]
        spec.append(("C. Features, no ViT", f"without {GROUP_NAMES[g]}", ABL / "models" / f"no_{g}", FEATURE_MODELS,
                     "stack_rel", rest, full_nv))
    for g in FRAME_GROUPS:
        spec.append(("C. Features, no ViT", f"only {GROUP_NAMES[g]}", ABL / "models" / f"only_{g}", FEATURE_MODELS,
                     "stack_rel", [g], full_nv))
    for g in FRAME_GROUPS:
        spec.append(("C. Features", f"only {GROUP_NAMES[g]} + ViT", ABL / "models" / f"only_{g}", all_models,
                     "stack_rel", [g], full))

    systems, rows = {}, []
    for section, name, run, models, kind, groups, ref in spec:
        preds = {s: base_predictions(run, s) for s in ("folds", "fixed")}
        if any(p is None for p in preds.values()):
            print(f"  (not trained yet: {section} / {name})", flush=True)
            continue
        rel = rel_all[[c for c in phase5.RELIABILITY if REL_GROUP[c] in groups]]
        cam = as_system(fuse(preds["folds"], models, kind, rel), "test")
        dset = as_system(fuse(preds["fixed"], models, kind, rel), "crossdataset")
        systems[(section, name)] = (cam, dset)
        if ref is None:
            systems["full"] = (cam, dset)
        if name == full_nv:
            systems["full_nv"] = (cam, dset)
        row = {"section": section, "configuration": name, "reference": ref or ""}
        row.update({f"cameras {m}": stats.METRICS[m](cam) for m in CAM_METRICS})
        row.update({f"datasets {m}": stats.METRICS[m](dset) for m in SET_METRICS})
        rows.append(row)
        print(f"{section:20s} {name:48s} cameras F1 {row['cameras macro F1']:.3f}  datasets F1 {row['datasets macro F1']:.3f}",
              flush=True)

    jobs, where = [], []
    for i, r in enumerate(rows):
        if not r["reference"]:
            continue
        ref = systems["full_nv" if r["reference"] == full_nv else "full"]
        cam, dset = systems[(r["section"], r["configuration"])]
        jobs += [(cam, ref[0]), (dset, ref[1])]
        where += [(i, "cameras"), (i, "datasets")]
    print(f"paired bootstrap: {len(jobs)} comparisons x {B} resamples ...", flush=True)
    with get_context("fork").Pool(12) as pool:
        for (i, test), res in zip(where, pool.map(paired, jobs, chunksize=1)):
            rows[i].update({f"{test} {k}": v for k, v in res.items()})
    t = pd.DataFrame(rows)
    ABL.mkdir(parents=True, exist_ok=True)
    t.to_csv(ABL / "ablation.csv", index=False)

    def delta(r, test):
        ref = t[(t.configuration == r["reference"]) & t.section.eq("A. Models")].iloc[0]
        d = r[f"{test} macro F1"] - ref[f"{test} macro F1"]
        p = r[f"{test} p F1"]
        return f"{d:+.3f} [{r[f'{test} dF1 lo']:+.3f}, {r[f'{test} dF1 hi']:+.3f}] p={p:.3f}{'' if p < 0.05 else ' n.s.'}"

    lines, md = [], ["# Ablation of the final HVPF (10 frames)", "",
                     "Unseen cameras: 5-fold test, 42,083 clips. Unseen datasets: FloreView + GenBuster, 4,716 clips. "
                     "Δ F1 = row minus its reference, with the 95% interval and p-value of a camera-level paired "
                     "bootstrap (2,000 resamples); n.s. = not significant at 0.05.", ""]
    for section, part in t.groupby("section", sort=False):
        show = pd.DataFrame({
            "configuration": part.configuration,
            "F1": part["cameras macro F1"].map("{:.3f}".format), "AUC": part["cameras AUC genuine vs not"].map("{:.3f}".format),
            "genuine acc. %": part["cameras genuine accepted %"].map("{:.1f}".format),
            "attacks caught %": part["cameras attacks caught %"].map("{:.1f}".format),
            "AI caught %": part["cameras AI caught %"].map("{:.1f}".format),
            "inj. a=1 accepted %": part["cameras injection attack (a=1) accepted %"].map("{:.1f}".format),
            "Δ F1 cameras": [delta(r, "cameras") if r["reference"] else "reference" for _, r in part.iterrows()],
            "F1 datasets": part["datasets macro F1"].map("{:.3f}".format),
            "AUC datasets": part["datasets AUC genuine vs not"].map("{:.3f}".format),
            "Δ F1 datasets": [delta(r, "datasets") if r["reference"] else "reference" for _, r in part.iterrows()]})
        ref = part.reference[part.reference != ""].unique()
        title = f"{section}" + (f"  (reference: {ref[0]})" if len(ref) else "")
        lines += [f"\n== {title}", show.to_string(index=False)]
        md += [f"## {title}", "", show.to_markdown(index=False), ""]
    text = "\n".join(lines)
    (ABL / "ablation.txt").write_text(text + "\n")
    out_md = phase5.ROOT / "report" / "thesis" / "ablation.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n")
    print(text)
    print(f"\nwrote {ABL / 'ablation.txt'} and {out_md}")


if __name__ == "__main__":
    main()
