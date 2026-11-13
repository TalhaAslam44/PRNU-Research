"""Paper - clips for a real-platform test: 150 short videos to send through WhatsApp and YouTube by hand.

VISION already holds real WhatsApp / YouTube copies of genuine videos; what the benchmark lacks are real
platform copies of ATTACKED and AI clips. All 150 clips come from test cameras / the test split (never seen in
training) and are rendered exactly like the benchmark clips (Steps 5 / 6: first 12 s, 480 x 480 centre crop,
libx264 CRF 20, one I-frame per second):
    50 genuine   re-encode of a native test-camera video (the control, same processing as the attacks)
    25 attacked  PRNU injection, alpha 1 (Step 5's victim camera)      25 attacked  PRNU removal, alpha 0.5
    25 AI        GenVidBench test clip                                 25 AI        + injected camera PRNU, alpha 1
Files get neutral random names (V001.mp4 ...); manifest.csv (keep it, do not upload it) maps them to classes.

Writes data/paper/real_upload/{originals/, manifest.csv, INSTRUCTIONS.md}.

    .venv/bin/python scripts/paper_upload_set.py
"""
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hvpf import phase5
from hvpf.attacks import attacker_fingerprint, encode, prnu_frames, read_clip
from hvpf.config import load_config

OUT = phase5.ROOT / "data" / "paper" / "real_upload"
SECONDS = 12
INSTRUCTIONS = """# Real WhatsApp / YouTube test - instructions

The folder `originals/` holds 150 short clips (V001.mp4 ... V150.mp4, 480 x 480, ~12 s or shorter).
Do NOT upload `manifest.csv` (it says which clip is genuine, attacked or AI).

## 1. WhatsApp (about 1 hour)
1. Copy `originals/` to your phone (USB cable, or Google Drive).
2. In WhatsApp, send the clips to **another phone or account** (a friend, or your second number). Send them as
   normal videos, NOT as "document" (documents are not re-compressed). Up to 30 clips per message is fine.
3. On the receiving phone, save the received videos (or open WhatsApp Web on this PC and download them).
4. Rename each received file back to its original name (V001.mp4 ...) - WhatsApp renames files, so send and save
   them in the same order, or check them one by one.
5. Put them in `data/paper/real_upload/whatsapp/` on this PC.

## 2. YouTube (about 1-2 hours, mostly waiting)
1. In YouTube Studio, upload all 150 clips as **Unlisted** (bulk upload works; keep the file names as titles).
2. Wait until processing has finished (HD processing can take a while).
3. In YouTube Studio > Content, use each video's menu > **Download** (this gives YouTube's processed file).
4. Rename to the original names (the title is the name) and put them in `data/paper/real_upload/youtube/`.

Missing or failed clips are fine - the analysis uses whatever is there. Tell Claude when the folders are filled.
"""


def targets(cfg):
    a = pd.read_csv(phase5.PHASE4 / "metadata" / "attacks.csv")
    t = a[(a.variant == "injection_a1") & (a.dataset == "VISION") & (a.split == "test")]
    s = phase5.load_step("05_attacks.py").load_samples(cfg["paths"])
    s["half"] = s.groupby("device").cumcount() % 2
    t = t[["content_id", "device", "claimed_device"]].rename(columns={"claimed_device": "victim"})
    return t.merge(s[["content_id", "path", "fps", "half"]], on="content_id").sort_values("content_id").reset_index(drop=True)


def ai_clips(cfg):
    s = pd.read_csv(phase5.PHASE4 / "metadata" / "synthetic.csv")
    s = s[(s.dataset == "GenVidBench") & (s.split == "test") & (s.status == "ok")]
    plain = s[s.social.eq("none") & s.inj_alpha.isna() & (s.content_id == s.base_content_id)]
    inj = s[s.content_id.str.endswith("+inj1")]
    def pick(d, n, seed):                       # spread over generators, then n clips
        per = max(1, n // d.generator.nunique() + 1)
        return pd.concat([g.sample(min(len(g), per), random_state=seed) for _, g in d.groupby("generator")]) \
            .sample(n, random_state=seed)
    p = pick(plain, 25, 1)
    i = pick(inj[~inj.base_content_id.isin(p.content_id)], 25, 2)
    return p, i


def main():
    cfg = load_config()
    crop, crf, preset = cfg["frames"]["crop"], cfg["attacks"]["crf"], cfg["attacks"]["preset"]
    kdir = cfg["paths"]["fingerprints_dir"] / "attacker"
    k = lambda name: attacker_fingerprint(np.load(kdir / name))
    t = targets(cfg)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(t))
    jobs = [("genuine", "re-encode", r, None) for r in t.iloc[order[:50]].itertuples()]
    att = t.iloc[rng.permutation(len(t))[:50]]
    jobs += [("attacked", "injection a=1", r, (f"{r.victim}_all.npy", 1.0, +1)) for r in att.iloc[:25].itertuples()]
    jobs += [("attacked", "removal a=0.5", r, (f"{r.device}_h{1 - int(r.half)}.npy", 0.5, -1)) for r in att.iloc[25:].itertuples()]
    plain, inj = ai_clips(cfg)
    jobs += [("AI", f"{r.generator}", r, None) for r in plain.itertuples()]
    jobs += [("AI", f"{r.generator} + injected PRNU a=1", r, (f"{r.claimed_device}_all.npy", 1.0, +1)) for r in inj.itertuples()]
    ids = [f"V{i:03d}" for i in rng.permutation(len(jobs)) + 1]
    (OUT / "originals").mkdir(parents=True, exist_ok=True)
    for sub in ("whatsapp", "youtube"):
        (OUT / sub).mkdir(exist_ok=True)

    def render(job, vid):
        cls, kind, r, mod = job
        path = getattr(r, "path", None) or r.source_path
        y, uv = read_clip(path, crop, SECONDS)
        if mod:
            y = prnu_frames(y, k(mod[0]), mod[1], mod[2])
        fps = float(r.fps) if getattr(r, "fps", None) and r.fps > 0 else 30.0
        out = OUT / "originals" / f"{vid}.mp4"
        if not out.exists():
            encode(y, uv, out, crop, fps, crf, preset, max(1, round(fps)))
        return {"id": vid, "class": cls, "kind": kind, "content_id": r.content_id,
                "claimed_device": getattr(r, "victim", None) if "injection" in kind and cls == "attacked"
                else (r.claimed_device if cls == "AI" else r.device), "fps": fps}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(lambda a: render(*a), zip(jobs, ids)))
    m = pd.DataFrame(rows).sort_values("id")
    m.to_csv(OUT / "manifest.csv", index=False)
    (OUT / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    size = sum(f.stat().st_size for f in (OUT / "originals").glob("*.mp4")) / 1e6
    print(m.groupby(["class", "kind"]).size().to_string())
    print(f"\n{len(m)} clips, {size:.0f} MB -> {OUT / 'originals'}")


if __name__ == "__main__":
    main()
