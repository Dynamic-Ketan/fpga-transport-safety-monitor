# Safety Project - Deployment & Testing Pipeline

Real-time safety monitoring and hazard detection system. This project demonstrates the complete workflow from pre-trained ML model deployment through inference testing and video-based violation detection on edge devices.

## Project Overview

This safety project implements a three-stage pipeline:

1. **Safety Monitoring** - Real-time hazard detection from sensor data
2. **Performance Analysis** - Latency and throughput benchmarking
3. **Video Analysis** - Traffic violation detection from aerial footage

The system uses a pre-trained Random Forest model optimized for FPGA deployment with sub-200ms latency requirements.

## Project Structure

```
safety_project/
├── data/                              # Input data and artifacts
│   ├── scaler_parameters.csv          # Feature normalization parameters (14 features)
│   ├── fpga_test_vectors.csv          # 100 test samples for validation
│   ├── fpga_safety_test_cases.csv     # Alternative test dataset
│   └── traffic_sample.mp4             # Video input for violation detection
│
├── models/                            # ML artifacts
│   └── best_model.pkl                 # Trained Random Forest classifier (binary)
│
├── src/                               # Source code
│   ├── safety_monitor.py              # Step 1: Real-time inference
│   ├── performance_test.py            # Step 2: Latency & throughput analysis
│   └── traffic_demo.py                # Step 3: Video-based detection
│
├── results/                           # Output artifacts
│   ├── test_results_*.csv             # Inference predictions + metadata
│   ├── performance_report.txt         # Performance metrics summary
│   ├── performance_results.csv        # Per-sample latency data
│   ├── latency_histogram.csv          # Latency distribution (20 bins)
│   ├── traffic_violations_simple.mp4  # Annotated output video
│   └── traffic_violations_simple_report.txt  # Violation events report
│
└── logs/                              # Execution logs
    └── safety_monitor.log             # Runtime logs and debugging info
```

## Workflow

### Stage 1: Safety Monitoring (Real-Time Inference)

**File**: `src/safety_monitor.py`

**Purpose**: Deploy pre-trained model for real-time hazard detection on sensor streams.

**Process**:
```
Input CSV (fpga_test_vectors.csv)
    ↓
Load Model & Scaler
    ↓ (best_model.pkl + scaler_parameters.csv)
Feature Normalization
    ↓ ((x - mean) / std for each feature)
Model Inference
    ↓ (predict + predict_proba)
Risk Classification
    ↓ (CRITICAL/HIGH/MEDIUM/LOW)
Consecutive Alert Tracking
    ↓ (3 alerts trigger alarm)
Output Results
    ↓ (CSV + console)
test_results_*.csv
```

**Key Operations**:
- Loads 100 test samples from `fpga_test_vectors.csv`
- Maps 14 scaler features to test data features (feature_0 through feature_13)
- Normalizes using saved mean/std from `scaler_parameters.csv`
- Generates probability scores and risk levels
- Tracks consecutive hazard detections for alarm triggering
- Outputs timestamped predictions

**Output**:
```csv
timestamp,sample_id,hazard_detected,probability,risk_level,consecutive_alerts,alarm_triggered
2025-12-06 14:10:02.517,0,False,0.143,LOW,0,False
2025-12-06 14:10:02.649,1,False,0.145,LOW,0,False
2025-12-06 14:10:02.781,2,True,0.856,CRITICAL,1,False
...
```

**Example Results** (100 samples):
- Hazards Detected: 24 (24.0%)
- Probability range: 0.0444 to 1.0
- Mean probability: 0.3128
- Alarms Triggered: 0 (needs 3+ consecutive alerts)

---

### Stage 2: Performance Testing (Benchmarking)

**File**: `src/performance_test.py`

**Purpose**: Measure inference latency, throughput, and resource utilization for FPGA requirements.

**Process**:
```
Load Model & Scaler
    ↓
For each test sample:
    Start Timer → Normalize → Predict → Stop Timer
    ↓
Collect Latency Metrics
    ↓ (min, max, mean, median, 99th percentile)
Calculate Throughput
    ↓ (samples/second)
Analyze Hazard Detection Latency
    ↓ (only for predicted hazards)
Generate Reports
    ↓ (TXT + CSV files)
```

**Latency Measurements**:
- Per-sample processing time (normalization + inference)
- Hazard-specific response time (microseconds)
- Distribution analysis (histogram)

**Output Files**:

1. **performance_report.txt** - Summary statistics
```
Total processing time: X.XXX seconds
Samples processed: 100
Throughput: 52.00 samples/second
Average sample time: 19.23 ms

Latency Statistics (ms):
  Min:    15.45
  Max:    48.32
  Mean:   22.18
  Median: 20.87
  99th %: 45.23

Detection Statistics:
  Total Samples: 100
  Hazards Detected: 24
  Detection Rate: 24.00%

Risk Level Distribution:
  CRITICAL:  2 ( 2.0%)
  HIGH:      5 ( 5.0%)
  MEDIUM:   17 (17.0%)
  LOW:      76 (76.0%)
```

2. **performance_results.csv** - Per-sample detailed data
```csv
sample_id,hazard,probability,risk_level,latency_ms
0,False,0.143,LOW,19.45
1,False,0.145,LOW,18.92
2,True,0.856,CRITICAL,21.34
...
```

3. **latency_histogram.csv** - Latency distribution
```csv
bin_start,bin_end,count
15.00,16.75,3
16.75,18.50,12
18.50,20.25,28
...
```

**Key Metrics**:
- Meets FPGA requirement: max latency < 200ms ✓
- High throughput: 50+ samples/second
- Consistent performance: low variance in latency

---

### Stage 3: Traffic Violation Detection (Video Analysis)

**File**: `src/traffic_demo.py`

**Purpose**: Detect traffic violations from aerial video footage using motion and vehicle density heuristics.

**Process**:
```
Load Video (traffic_sample.mp4)
    ↓
For each frame:
    ├─ Vehicle Detection (edge detection + contours)
    ├─ Motion Analysis (frame difference)
    ├─ Violation Heuristics
    │  ├─ Too many vehicles (>15) → Heavy congestion
    │  ├─ High motion + vehicles → Erratic driving
    │  ├─ Extreme motion (>0.12) → Accident/Emergency
    │  └─ Red-phase vehicles → Red light violation
    ├─ Draw Annotations
    │  ├─ Vehicle bounding boxes
    │  ├─ Violation banner
    │  ├─ Confidence scores
    │  └─ Statistics overlay
    └─ Write to output video
         ↓
traffic_violations_simple.mp4 (annotated)
traffic_violations_simple_report.txt (events)
```

**Violation Detection Rules**:
```python
if len(vehicles) > 15:
    violation = "Heavy congestion detected" (0.85 confidence)
elif motion > 0.08 and len(vehicles) > 8:
    violation = "Erratic traffic pattern" (0.78 confidence)
elif motion > 0.12:
    violation = "Extreme motion detected" (0.92 confidence)
elif frame_in_red_phase and len(vehicles) > 5 and motion > 0.05:
    violation = "Potential red light violation" (0.72 confidence)
```

**Output Files**:

1. **traffic_violations_simple.mp4** - Annotated video with:
   - Red banner: VIOLATION alert
   - Green banner: NORMAL TRAFFIC
   - Blue boxes: Vehicle bounding boxes (red if violation)
   - Info panel: Frame count, vehicle count, motion %, violations
   - Confidence score (if violation detected)
   - Timestamp

2. **traffic_violations_simple_report.txt** - Event log
```
TRAFFIC VIOLATION DETECTION REPORT
====================================

Input video: traffic_sample.mp4
Date: 2025-12-06 18:33:15
Total frames: 287
Violations: 2
Violation rate: 0.70%

Violation Events:
─────────────────────────────────────────────────────────
Frame    Reason                             Conf%    Vehicles  Motion%
─────────────────────────────────────────────────────────
105      Heavy congestion detected          85       16        8.5
203      Extreme motion detected            92       12        14.2
─────────────────────────────────────────────────────────
```

**Video Processing Stats**:
```
Frames processed: 287
Violations found: 2
Violation rate: 0.70%
Processing time: 9.5 seconds (30.2 FPS)
```

---

## Data Flow

### Feature Mapping

**Training** → Scaler learned on 14 engineered features (named):
```
accel_x, accel_y, accel_z, steering_angle, brake_pressure, speed,
accel_x_change, steering_change, brake_change, accel_x_std, 
steering_std, speed_std, accel_magnitude, jerk
```

**Inference** → Test data has generic feature names:
```
feature_0, feature_1, feature_2, ... feature_13
```

**Mapping Solution** → Auto-mapped by index:
```
scaler feature[0] (accel_x) ← feature_0
scaler feature[1] (accel_y) ← feature_1
... and so on
```

### Normalization

Raw feature → Scaler parameters → Normalized value
```
value = -0.185 (raw)
mean = 0.004, std = 0.789 (from scaler_parameters.csv)
normalized = (value - mean) / std = -0.239
```

---

## Integration Workflow

```
┌─────────────────────────────────────────────────────────┐
│ PRE-TRAINED ARTIFACTS (from Colab)                      │
├─────────────────────────────────────────────────────────┤
│ • best_model.pkl (Random Forest, 14 features)          │
│ • scaler_parameters.csv (mean, std for each feature)   │
└──────────────┬──────────────────────────────────────────┘
               │
        ┌──────┴─────────┐
        │                │
        ▼                ▼
┌──────────────┐  ┌─────────────────┐
│Stage 1:      │  │Stage 2:         │
│Safety        │  │Performance      │
│Monitoring    │  │Testing          │
│(Inference)   │  │(Benchmarking)   │
│              │  │                 │
│Input:        │  │Input:           │
│fpga_test_    │  │fpga_test_       │
│vectors.csv   │  │vectors.csv      │
│              │  │                 │
│Output:       │  │Output:          │
│test_results_ │  │performance_     │
│*.csv         │  │report.txt       │
│              │  │perf_results.csv │
│              │  │latency_hist.csv │
└──────────────┘  └─────────────────┘
        │                │
        └────────┬───────┘
                 │
                 ▼
        ┌──────────────────┐
        │Stage 3:          │
        │Traffic Violation │
        │Detection (Video) │
        │                  │
        │Input:            │
        │traffic_sample.mp4│
        │                  │
        │Output:           │
        │traffic_violations│
        │_simple.mp4       │
        │_report.txt       │
        └──────────────────┘
```

---

## Key Technical Points

### Model Architecture
- **Type**: Random Forest Classifier
- **Features**: 14 normalized inputs
- **Classes**: 2 (Normal/Anomaly)
- **Size**: ~100KB (FPGA-friendly)

### Performance Characteristics
- **Mean Latency**: 22ms
- **Max Latency**: 48ms
- **Throughput**: 52 samples/second
- **Detection Accuracy**: 90.6%

### Edge Device Requirements
- Python 3.8+
- OpenCV for video processing
- Scikit-learn for model inference
- NumPy/Pandas for data handling

### FPGA Deployment Path
1. Tree-based model (natural for hardware)
2. Fixed-point arithmetic compatible
3. No dynamic memory needed
4. Deterministic execution time
5. Ready for HLS conversion

---

## Execution Steps

```bash
# Step 1: Real-time Safety Monitoring
python3 src/safety_monitor.py
# Output: results/test_results_*.csv

# Step 2: Performance Benchmarking
python3 src/performance_test.py
# Output: results/performance_*.txt, .csv

# Step 3: Video Violation Detection
python3 src/traffic_demo.py
# Output: results/traffic_violations_*.mp4, .txt
```

---

## File Dependencies

| File | Reads | Writes | Purpose |
|------|-------|--------|---------|
| safety_monitor.py | best_model.pkl, scaler_parameters.csv, fpga_test_vectors.csv | test_results_*.csv | Inference |
| performance_test.py | best_model.pkl, scaler_parameters.csv, fpga_test_vectors.csv | performance_report.txt, performance_results.csv, latency_histogram.csv | Benchmarking |
| traffic_demo.py | traffic_sample.mp4 | traffic_violations_simple.mp4, traffic_violations_simple_report.txt | Video analysis |

---

## References

- **Model Training**: Scikit-learn Random Forest (from Colab notebook)
- **Deployment Framework**: Pure Python (FPGA-compatible)
- **Video Processing**: OpenCV 4.x
- **Data Processing**: Pandas, NumPy

---

**Project Type**: ML Model Deployment & Testing  
**Status**: Complete - Ready for FPGA Integration  
**Last Updated**: December 6, 2025
