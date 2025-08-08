![IEEE-TRS](https://img.shields.io/badge/IEEE--TRS-Targeted-blue)
![Submitted](https://img.shields.io/badge/status-Submitted-orange)
![Qiskit](https://img.shields.io/badge/Qiskit-Supported-6f42c1)
![Simulated](https://img.shields.io/badge/Mode-Simulated-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Open-Access](https://img.shields.io/badge/Open--Access-Available-brightgreen)

# QCR-Full: Quantum-Cognitive Radar - Full Simulator

## Project Overview
This repository accompanies the research paper:
"Quantum-Cognitive Radar: A Next-Generation Framework for Entanglement-Driven Adaptive Sensing in Complex Environments".

Submitted to IEEE Transactions on Radar Systems.

The framework integrates:
- Entangled Two-Mode Squeezed Vacuum (TMSV) source
- Thermal-loss channel with reflectivity (kappa) and background noise (N_B)
- Detection schemes: Threshold, Quantum Chernoff Bound (QCB), adaptive Quantum Neural Network (QNN)
- Performance metrics: P_D, P_FA, ROC curves, quantum-advantage heatmaps, complexity charts
- Reproducible simulation scripts (CLI and notebooks), optional Qiskit utilities

---

## Repository Structure
qcr_full/
├─ src/                  # main source code  
├─ scripts/              # helper scripts  
├─ figures/              # generated figures (PNG, PDF)  
├─ results/              # simulation results (CSV, JSON)  
├─ tests/                # unit/integration tests  
├─ assets/               # static resources  
└─ .github/              # workflows and templates  

---

## Quick Start
```bash
git clone https://github.com/YassirALKarawi/qcr_full.git
cd qcr_full
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.run --scenario baseline --rounds 50 --seed 42 --out results/ --plots figures/

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
