# Moving the project to a new system (SSD)

The code is on GitHub; everything else below lives only on this machine. Wait until the running
jobs (Steps 7-9) and the FloreView download have finished, then back up.

## What to back up

| Item | Location | Size | Back up? | Why |
|---|---|---|---|---|
| Generated data (Steps 1-7: metadata, VISION/ACID frames, fingerprints, features, results) | `~/PRNU-Research/data/` | ~15 GB | **yes** | rebuilding takes ~12 h |
| Attack + synthetic frames | `/media/talhaaslam/Disk/PRNU-Research-data/` | ~31 GB | **yes** | rebuilding takes ~5 h |
| VISION videos | `~/Downloads/Vision-Dataset/` | 103 GB | **yes** | the download took ~41 h |
| FloreView videos | `/media/talhaaslam/Disk/datasets/FloreView/` | ~95 GB | **yes** | slow download (~0.5 MB/s) |
| Synthetic archives | `/media/talhaaslam/Disk/datasets/synthetic/{genvidbench,genbuster}/` | 13.5 GB | **yes** | small; `extracted/` (14 GB) can be skipped and re-extracted |
| Video-ACID archives | `~/datasets/VideoACID/` | 174 GB | optional | frames are already in `data/frames/ACID`; only needed to re-extract or use more videos |
| Progress report | `~/PRNU-Research/report/` | 1 MB | **yes** | not in git |
| Thesis files | `~/Downloads/Final_Thesis_Report_*`, other PDFs | small | **yes** | |
| Claude Code memory | `~/.claude/` | 14 MB | optional | keeps Claude's notes about this project |
| Tools (`gh`, `7zz`, `tectonic`) | `~/.local/bin/` | 69 MB | optional | easy to re-download |
| Python environment | `~/PRNU-Research/.venv/` | 6.6 GB | **no** | recreate from `requirements.txt` |

If the old hard disk stays in the new machine as a second drive, the datasets can simply stay on it:
mount it and point `config.yaml` at it.

## Restoring

1. System packages: `ffmpeg`, `git`, `python3-venv`, the NVIDIA driver (check `nvidia-smi`).
2. Code and environment:
   ```bash
   git clone --recursive https://github.com/TalhaAslam44/PRNU-Research.git ~/PRNU-Research
   cd ~/PRNU-Research
   git -C third_party/prnu-python apply ../prnu-python-numpy2.patch
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ./third_party/get_noiseprintpp.sh
   ```
3. Copy the backed-up `data/` folder back to `~/PRNU-Research/data/`.
4. `data/frames/ATTACK` and `data/frames/SYNTH` are symlinks into the second partition. Either copy
   those folders straight into `data/frames/` (if the SSD has room) or recreate the links:
   `ln -sfn <new location>/frames/ATTACK data/frames/ATTACK` (same for `SYNTH`).
5. Update the dataset folders in `config.yaml` (`vision_root`, `acid_root`, `floreview_root`,
   `synthetic.root`).
6. The tables store absolute paths. With the same user name (`talhaaslam`) and the same mount
   points nothing changes; otherwise rewrite each old prefix (dry run first):
   ```bash
   python scripts/relocate_paths.py /media/talhaaslam/Disk /new/mount/point
   python scripts/relocate_paths.py /media/talhaaslam/Disk /new/mount/point --apply
   ```
7. Check: `python scripts/04_prnu_baseline.py --datasets VISION` reuses the cached fingerprints and
   should print the same table as in the README within a few minutes.
