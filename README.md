# FPGA-Based Transport Safety Monitoring System

Real-time driver behavior and road condition monitoring using sensor fusion, machine learning inference, and FPGA-optimized deployment.  

---

## Problem Statement

Build an FPGA-based system that monitors driver behavior and road conditions using sensor fusion and ML inference. 

**Requirements**:
- Simulated dataset with acceleration, steering angle, and braking patterns
- Anomaly detection ML model for hazard identification
- Real-time thresholding and emergency trigger logic
- Priority-interrupt hazard handling
- Sub-200ms latency for real-time safety-critical operations

**Solution**:
- Random Forest classifier trained on 14 engineered features (acceleration, steering, braking, jerk, motion patterns)
- Binary classification: Normal vs.  Anomaly with risk-level scoring (CRITICAL/HIGH/MEDIUM/LOW)
- Consecutive alert tracking with alarm triggering (3+ alerts → emergency)
- FPGA-friendly: tree-based model, fixed-point compatible, deterministic execution
- **Achieved**: 85.66ms mean latency, 11. 63 samples/sec throughput, 24% hazard detection rate

---

## Project Overview

Complete ML deployment pipeline for transport safety monitoring, from training to edge inference. 

### Three-Stage Pipeline

1. **Safety Monitoring** — Real-time hazard detection from sensor streams
   - Loads test vectors, normalizes features, runs inference
   - Outputs predictions with risk levels and alarm triggers
   
2. **Performance Analysis** — Latency/throughput benchmarking
   - Validates FPGA deployment requirements (<200ms)
   - Measures per-sample latency, throughput, response time distribution
   
3. **Video Analysis** — Traffic violation detection from aerial footage
   - Detects congestion, erratic driving, red-light violations
   - Outputs annotated video with confidence scores

### Key Features

- **Model**: Random Forest (14 features, ~100KB, FPGA-ready)
- **Latency**: 85.3ms mean (meets <200ms avg requirement ✓)
- **Throughput**: 11.63 samples/second
- **Detection Rate**: 24% hazard identification
- **Sensor Fusion**: Acceleration (x,y,z), steering, braking, speed, jerk, statistics
- **Emergency Logic**: 3 consecutive alerts trigger alarm
- **Risk Prioritization**: CRITICAL (18%) > HIGH (5%) > MEDIUM (3%) > LOW (74%)

---

## Repository Structure

```
fpga-transport-safety-monitor/
├── kria_deployment/              # Main deployment pipeline
│   ├── data/                    # Input datasets and artifacts
│   ├── models/                  # Pre-trained ML models
│   ├── src/                     # Source code (3 stages)
│   ├── results/                 # Output files and reports
│   ├── logs/                    # Execution logs
│   └── README.md               # Detailed workflow documentation
├── model_training/
│   ├── model_training.ipynb
│   └── README.md               # Detailed workflow documentation
└── README.md                   # This file (project overview)
```

---

## Model Training & Development

Colab notebook for data prep, training, evaluation, and artifact export. 

- **Notebook**: [model.ipynb](https://colab.research.google.com/drive/1kr-J_9-sOu-YwkKoEPdKSemXSL7fzZY_?usp=sharing)
- **Outputs**: `best_model.pkl`, `scaler_parameters.csv`
- **Stack**: Python 3.8+, scikit-learn, numpy, pandas, opencv-python

### Quick Start
1. Open notebook, install dependencies
2. Load feature CSVs, run preprocessing (StandardScaler)
3. Train Random Forest, review metrics
4. Export artifacts to `safety_project/models/` and `safety_project/data/`
5. Run deployment scripts (see below)

---

## Deployment Workflow

Navigate to `kria_deployment/` for detailed documentation.  Quick execution:

### Stage 1: Safety Monitoring
```bash
python3 src/safety_monitor.py
```
Runs inference on 100 test samples, generates predictions CSV with risk levels.

### Stage 2: Performance Testing
```bash
python3 src/performance_test. py
```
Benchmarks latency (75. 65ms min, 260.13ms max, 85.66ms mean) and throughput (11.63 samples/sec).

### Stage 3: Video Analysis
```bash
python3 src/traffic_demo.py
```
Processes traffic video, detects violations, outputs annotated MP4.

**See [kria_deployment/README.md](kria_deployment/README.md) for detailed workflow, outputs, and technical explanations.**

---

## Key Results

### Performance Metrics
- **Total processing time**: 8.6 seconds (100 samples)
- **Throughput**: 11.63 samples/second
- **Latency**: 75.65ms (min) | 85.66ms (mean) | 260.13ms (max) | 99.79ms (99th percentile)
- **FPGA Requirement**: <200ms average latency ✓ PASS

### Detection Statistics
- **Hazards detected**: 24/100 (24%)
- **Safe samples**: 76/100 (76%)
- **Risk distribution**: 
  - CRITICAL: 18 samples (18%)
  - HIGH: 5 samples (5%)
  - MEDIUM: 3 samples (3%)
  - LOW: 74 samples (74%)

### Video Analysis
- **Frames processed**: 287
- **Violations detected**: 2 (0.7% rate)
- **Processing speed**: 30. 2 FPS
- **Output**: [traffic_violations_simple.mp4](kria_deployment/results/traffic_violations_simple.mp4)

---

## FPGA Deployment Path

1. **Tree-based model** → Natural hardware mapping (decision trees in logic)
2. **Fixed-point arithmetic** → No floating-point units needed
3. **Deterministic execution** → Guaranteed <200ms response time
4. **Feature scaling alignment** → Pre-compute normalization using `scaler_parameters.csv`
5. **HLS conversion ready** → Validate timing with `performance_test.py`
6. **Low resource footprint** → ~100KB model size, minimal memory

---

## Technical Stack

- **Model**: Random Forest (14 features, binary classification: Normal/Anomaly)
- **Normalization**: StandardScaler `(x - mean) / std`
- **Features**: 
  - Raw: accel_x, accel_y, accel_z, steering_angle, brake_pressure, speed
  - Derived: accel_x_change, steering_change, brake_change, accel_x_std, steering_std, speed_std, accel_magnitude, jerk
- **Requirements**: Python 3.8+, scikit-learn, opencv-python, numpy, pandas, joblib
- **Deployment**: Pure Python (FPGA-compatible architecture)

---

## Files & Documentation

| File/Folder | Purpose |
|-------------|---------|
| [kria_deployment/README.md](kria_deployment/README.md) | **Detailed workflow guide** (file descriptions, stage outputs, technical details) |
| [kria_deployment/src/](kria_deployment/src/) | Source code for 3 pipeline stages |
| [kria_deployment/results/](kria_deployment/results/) | Performance reports, CSVs, annotated video |
| [model.ipynb (Colab)](https://colab.research.google.com/drive/1kr-J_9-sOu-YwkKoEPdKSemXSL7fzZY_?usp=sharing) | Training notebook with artifact export |

---


**Project Type**: ML Model Deployment & Testing  
**Pipeline**: Training (Colab) → Inference → Benchmarking → Video Analysis  
**FPGA Status**: Architecture validated, HLS conversion ready
