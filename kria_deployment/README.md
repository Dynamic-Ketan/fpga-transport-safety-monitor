# Safety Project - Model Deployment & Testing

Real-time safety monitoring system using pre-trained ML model for hazard detection and traffic violation analysis.

## Project Structure

```
safety_project/
├── data/                              
│   ├── scaler_parameters.csv          # Feature normalization (mean, std for 14 features)
│   ├── fpga_test_vectors.csv          # 100 test samples
│   ├── fpga_safety_test_cases.csv     # Additional test data
│   └── traffic_sample.mp4             # Video input
│
├── models/                            
│   └── best_model.pkl                 # Pre-trained Random Forest model
│
├── src/                               
│   ├── safety_monitor.py              # Real-time inference
│   ├── performance_test.py            # Latency/throughput benchmarking
│   └── traffic_demo.py                # Video violation detection
│
├── results/                           # Generated outputs
│   ├── test_results_*.csv
│   ├── performance_report.txt
│   ├── performance_results.csv
│   ├── latency_histogram.csv
│   ├── traffic_violations_simple.mp4
│   └── traffic_violations_simple_report.txt
│
└── logs/                              
    └── safety_monitor.log
```

## Files Explained

### Data Files
- **scaler_parameters.csv** - Contains mean and std for normalizing 14 features before inference
- **fpga_test_vectors.csv** - 100 test samples with features (feature_0 to feature_13)
- **traffic_sample.mp4** - Video file for violation detection demo

### Model Files
- **best_model.pkl** - Trained Random Forest classifier (binary: Normal/Anomaly)

### Source Files

**safety_monitor.py**
- Loads model and scaler parameters
- Reads test vectors from CSV
- Normalizes features and runs inference
- Outputs predictions with timestamps and risk levels

**performance_test.py**
- Measures per-sample latency (normalization + inference)
- Calculates throughput and response times
- Generates performance reports and histograms

**traffic_demo.py**
- Processes video frame-by-frame
- Detects vehicles using edge detection
- Analyzes motion patterns
- Flags violations (congestion, erratic driving, red-light)
- Outputs annotated video

## Workflow

### Stage 1: Safety Monitoring
```
python3 src/safety_monitor.py
```
Runs inference on test vectors and generates predictions CSV.

### Output

<img width="1600" height="865" alt="image" src="https://github.com/user-attachments/assets/6aa55fc6-e98e-4959-8b73-836103249cb0" />

### Stage 2: Performance Testing
```
python3 src/performance_test.py
```
Benchmarks system latency and throughput, generates performance metrics.

### Output

<img width="1600" height="848" alt="image" src="https://github.com/user-attachments/assets/ec650f32-6ff0-4736-8d7c-34497737deba" />

### Stage 3: Video Analysis
```
python3 src/traffic_demo.py
```
Processes traffic video and generates annotated output with violations flagged.

### Output 

Check [Output Video](safety_project/results/traffic_violations_simple.mp4)

## Data Flow

```
Pre-trained Artifacts (from Colab)
    ↓
best_model.pkl + scaler_parameters.csv
    ↓
├─→ safety_monitor.py    → test_results_*.csv
├─→ performance_test.py  → performance reports
└─→ traffic_demo.py      → annotated video
```

## Technical Details

- **Model**: Random Forest (14 features, binary classification)
- **Normalization**: StandardScaler (x - mean) / std
- **Feature Mapping**: Auto-maps scaler features to test data columns
- **Requirements**: Python 3.8+, scikit-learn, opencv-python, numpy, pandas

## Execution

All scripts use absolute paths to `/home/ubuntu/safety_project/` directories. Run from any location.

---

**Status**: Deployment Ready  
**Last Updated**: December 6, 2025
