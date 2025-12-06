## Model Training & Development

This repository pairs with a Colab notebook for data preparation, training, evaluation, and exporting artifacts used by the deployment scripts.

### Notebook
- **model.ipynb** — End-to-end pipeline:
  - Data preprocessing and configuration
  - Random Forest training (baseline)
  - Metrics, evaluation, and artifact export
  - Notes on FPGA acceleration integration
- Open in Colab: [Google Colab Notebook](https://colab.research.google.com/drive/1kr-J_9-sOu-YwkKoEPdKSemXSL7fzZY_?usp=sharing)


### Notebook Outputs
- `best_model.pkl` (trained model)
- `scaler_parameters.csv` (feature-wise mean/std)
- Optional: metrics plots and experiment notes

### Reproducing Locally
- Python 3.8+; install:
  ```
  pip install numpy pandas scikit-learn opencv-python matplotlib seaborn joblib
  ```
- Place exported artifacts into `models/` and `data/`.

### Integrating with Deployment
- After placing `best_model.pkl` and `scaler_parameters.csv`, run:
  - Safety Monitoring:
    ```
    python3 src/safety_monitor.py
    ```
  - Performance Testing:
    ```
    python3 src/performance_test.py
    ```
  - Video Analysis:
    ```
    python3 src/traffic_demo.py
    ```

### Notes on FPGA Acceleration
- Align feature scaling on FPGA with `scaler_parameters.csv`.
- Consider quantization or FPGA-friendly formats.
- Validate throughput/latency with `performance_test.py`.

---

## Technical Details
- Model: Random Forest (14 features, binary classification)
- Normalization: StandardScaler `(x - mean) / std`
- Feature Mapping: Auto-maps scaler features to test data columns
- Requirements: Python 3.8+, scikit-learn, opencv-python, numpy, pandas
