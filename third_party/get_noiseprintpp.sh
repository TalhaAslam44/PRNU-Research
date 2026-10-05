#!/bin/bash
# Fetch the Noiseprint++ network definition and weights from GRIP-UNINA's TruFor
# (pinned commit). Licensed by GRIP-UNINA for nonprofit research only; their notices
# and LICENSE.txt are kept, and TruFor (Guillaro et al., CVPR 2023) must be cited.
set -euo pipefail
cd "$(dirname "$0")"
REV=ae54475df6f41a491d7615100feb19263dec13f7
RAW=https://raw.githubusercontent.com/grip-unina/TruFor/$REV/TruFor_train_test
mkdir -p noiseprintpp
curl -sSL -o noiseprintpp/DnCNN.py        $RAW/lib/models/DnCNN.py
curl -sSL -o noiseprintpp/LICENSE.txt     $RAW/LICENSE.txt
curl -sSL -o noiseprintpp/noiseprint++.th "$RAW/pretrained_models/noiseprint++/noiseprint++.th"
touch noiseprintpp/__init__.py
ls -la noiseprintpp
