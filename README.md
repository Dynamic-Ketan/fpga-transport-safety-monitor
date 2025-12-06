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
- Achieved: 85.3ms mean latency, 11.63 samples/sec throughput, 90.6% accuracy

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
- **Latency**: 85.3ms mean, 48ms max (meets <200ms requirement ✓)
- **Throughput**: 52 samples/second on CPU
- **Accuracy**: 90. 6% hazard detection
- **Sensor Fusion**: Acceleration (x,y,z), steering, braking, speed, jerk, statistics
- **Emergency Logic**: 3 consecutive alerts trigger alarm
- **Risk Prioritization**: CRITICAL > HIGH > MEDIUM > LOW

---

## Model Training & Development

Colab notebook for data prep, training, evaluation, and artifact export. 

- **Notebook**: [model.ipynb](https://colab.research.google.com/drive/1kr-J_9-sOu-YwkKoEPdKSemXSL7fzZY_?usp=sharing)
- **Outputs**: `best_model.pkl`, `scaler_parameters.csv`
- **Stack**: Python 3.8+, scikit-learn, numpy, pandas, opencv-python

### Quick Start
1. Open notebook, install dependencies
2. Load feature CSVs, run preprocessing
3. Train Random Forest, review metrics
4. Export artifacts to `models/` and `data/`
5. Run deployment scripts:
   ```bash
   python3 src/safety_monitor.py      # Inference
   python3 src/performance_test.py    # Benchmarking
   python3 src/traffic_demo.py        # Video analysis
   ```

---

## Project Structure

```
safety_project/
├── data/                              # Inputs & artifacts
│   ├── scaler_parameters.csv          # Normalization params (14 features)
│   ├── fpga_test_vectors. csv          # 100 test samples
│   └── traffic_sample.mp4             # Video input
│
├── models/
│   └── best_model. pkl                 # Trained Random Forest
│
├── src/
│   ├── safety_monitor.py              # Real-time inference
│   ├── performance_test.py            # Latency benchmarking
│   └── traffic_demo.py                # Video violation detection
│
└── results/                           # Outputs
    ├── test_results_*.csv
    ├── performance_report. txt
    ├── traffic_violations_simple.mp4 
    └── traffic_violations_simple_report.txt
```

---

## Results

- **Output Video**: [traffic_violations_simple. mp4](safety_project/results/traffic_violations_simple.mp4) | [Download](safety_project/results/traffic_violations_simple.mp4? raw=1)
- **Performance**: 85.3ms latency, 52 samples/sec, <200ms max ✓
- **Detection**: 24% hazard rate, 90.6% accuracy
- **Video**: 287 frames, 2 violations (0.7%), 30. 2 FPS processing

---

## FPGA Deployment Path

1. **Tree-based model** → Natural hardware mapping
2. **Fixed-point arithmetic** → No floating-point units needed
3. **Deterministic execution** → Guaranteed <200ms response
4. **Feature scaling alignment** → Use `scaler_parameters.csv` on FPGA preprocessing
5. **HLS conversion ready** → Validate with `performance_test.py`

---

## Technical Details

- **Model**: Random Forest (14 features, binary classification)
- **Normalization**: StandardScaler `(x - mean) / std`
- **Features**: accel_x/y/z, steering, brake, speed, jerk, change rates, std deviations
- **Requirements**: Python 3.8+, scikit-learn, opencv, numpy, pandas
- **Deployment**: Pure Python (FPGA-compatible architecture)

---

## Execution

```bash
python3 src/safety_monitor. py      # Real-time hazard detection
python3 src/performance_test.py    # Latency/throughput analysis
python3 src/traffic_demo.py        # Video violation detection
```

---

**Project Type**: ML Model Deployment & Testing  
**Complete**: Training → Inference → Benchmarking → Video Analysis
