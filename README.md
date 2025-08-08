# Quantum-Cognitive Radar (QCR) — Full Simulator

This repository contains the *full* simulator code you requested, including:
- Classical, QCB, and QNN detectors
- Comprehensive plotting (ROC, mode/noise effects, heatmaps, complexity, convergence, jamming, etc.)
- CSV outputs and high-DPI figures (PNG + PDF)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/simulator.py
```
Outputs go to `figures/` and `results/`.
