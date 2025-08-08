#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/simulator.py
echo "Done. See figures/ and results/."
