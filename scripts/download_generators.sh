#!/bin/bash
# Phase 4: six more AI-video generators from GenVidBench (Hugging Face jian-0/GenVidBench, ~35 GB)
# into <synthetic root>/../genvidbench/. Resumable: re-run to continue. Extraction (a few hundred
# clips per generator) is done by scripts/extract_generators.sh.
set -u
DEST=/media/talhaaslam/Disk/datasets/synthetic/genvidbench
HF=https://huggingface.co/datasets/jian-0/GenVidBench/resolve/main/GenVidBench
mkdir -p "$DEST"
get() { curl -sSL --retry 5 --retry-delay 10 -C - -o "$DEST/$2" "$HF/$1" && echo "ok   $2 $(date +%H:%M)" || echo "FAIL $2"; }
get Pair1_labels.txt Pair1_labels.txt
for f in Pair1/ms.rar Pair2/svd.rar Pair2/mora.rar Pair1/t2vz.rar Pair1/pika.rar Pair1/vc2.rar; do
  get "$f" "$(basename "$f")"
done
touch "$DEST/.phase4_download_done"
