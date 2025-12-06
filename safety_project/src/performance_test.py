#!/usr/bin/env python3
"""
Performance Testing & Analysis
Tests system with larger dataset and measures performance metrics
"""

import pickle
import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime

class SafetyMonitor:
    def __init__(self, model_path, scaler_path, test_file=None):
        """Initialize safety monitoring system"""
        print("Initializing Safety Monitor...")
        
        # Load trained model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded: {type(self.model).__name__}")
        
        # Load scaler parameters
        scaler_df = pd.read_csv(scaler_path)
        print(f"Scaler parameters loaded: {len(scaler_df)} features")
        
        # Map scaler features to feature_X format
        if test_file:
            test_data = pd.read_csv(test_file)
            feature_cols = [col for col in test_data.columns if col.startswith('feature_')]
            print(f"Test data has features: {feature_cols[:3]}... (showing first 3)")
        else:
            feature_cols = [f'feature_{i}' for i in range(len(scaler_df))]
        
        # Create mapping: scaler feature name → test data feature name
        self.feature_mapping = {}
        for i, scaler_feature in enumerate(scaler_df['feature'].tolist()):
            if i < len(feature_cols):
                self.feature_mapping[scaler_feature] = feature_cols[i]
        
        # Store stats using test data feature names
        self.feature_stats = {}
        for idx, row in scaler_df.iterrows():
            test_feature = self.feature_mapping.get(row['feature'], f'feature_{idx}')
            self.feature_stats[test_feature] = {
                'mean': row['mean'],
                'std': row['std']
            }
        
        self.feature_names = list(self.feature_stats.keys())
        print(f"Features mapped: {len(self.feature_names)} features")
        
        # Safety thresholds
        self.hazard_threshold = 0.7
        self.consecutive_alerts = 0
        self.alert_limit = 3
        
    def normalize_features(self, raw_data):
        """Normalize input features"""
        normalized_array = []
        
        for feature_name in self.feature_names:
            if feature_name in raw_data:
                value = raw_data[feature_name]
            else:
                value = 0.0
            
            if feature_name in self.feature_stats:
                mean = self.feature_stats[feature_name]['mean']
                std = self.feature_stats[feature_name]['std']
                
                if std > 0:
                    normalized = (value - mean) / std
                else:
                    normalized = 0.0
            else:
                normalized = 0.0
            
            normalized_array.append(normalized)
        
        return np.array([normalized_array])
    
    def predict_hazard(self, sensor_data):
        """Predict hazard from sensor data"""
        X = self.normalize_features(sensor_data)
        
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        hazard_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        if hazard_prob > 0.8:
            risk_level = "CRITICAL"
        elif hazard_prob > 0.6:
            risk_level = "HIGH"
        elif hazard_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        hazard_detected = prediction == 1 or hazard_prob > self.hazard_threshold
        
        return hazard_detected, hazard_prob, risk_level
    
    def process_sample(self, sensor_data, sample_id=None):
        """Process single sensor sample"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        hazard, probability, risk_level = self.predict_hazard(sensor_data)
        
        if hazard:
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = 0
        
        alarm_triggered = self.consecutive_alerts >= self.alert_limit
        
        result = {
            'timestamp': timestamp,
            'sample_id': sample_id,
            'hazard_detected': hazard,
            'probability': probability,
            'risk_level': risk_level,
            'consecutive_alerts': self.consecutive_alerts,
            'alarm_triggered': alarm_triggered,
            'sensor_data': sensor_data
        }
        
        return result

def run_performance_test():
    print("\n" + "="*70)
    print("PERFORMANCE TEST - TRANSPORT SAFETY MONITORING SYSTEM")
    print("="*70 + "\n")
    
    # Get script directory and build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    MODEL_PATH = os.path.join(base_dir, "models/best_model.pkl")
    SCALER_PATH = os.path.join(base_dir, "data/scaler_parameters.csv")
    TEST_FILE = os.path.join(base_dir, "data/fpga_test_vectors.csv")
    RESULTS_DIR = os.path.join(base_dir, "results")
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Initializing system...")
    monitor = SafetyMonitor(MODEL_PATH, SCALER_PATH, TEST_FILE)
    
    # Load test data
    print("\nLoading test dataset...")
    df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(df)} test samples")
    
    # Performance metrics
    processing_times = []
    hazard_latencies = []
    
    print("\n" + "-"*70)
    print("Running real-time simulation...")
    print("-"*70)
    
    start_total = time.time()
    results_list = []
    
    for idx, row in df.iterrows():
        # Measure per-sample processing time
        start_sample = time.time()
        
        sensor_data = {k: v for k, v in row.to_dict().items() 
                      if isinstance(v, (int, float))}
        result = monitor.process_sample(sensor_data, sample_id=idx)
        
        end_sample = time.time()
        sample_time = (end_sample - start_sample) * 1000  # Convert to ms
        processing_times.append(sample_time)
        
        # Track hazard detection latency
        if result['hazard_detected']:
            latency_us = sample_time * 1000  # Convert to microseconds
            hazard_latencies.append(latency_us)
        
        # Store result
        results_list.append({
            'sample_id': idx,
            'hazard': result['hazard_detected'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'latency_ms': sample_time
        })
        
        # Progress indicator
        if (idx + 1) % 20 == 0:
            avg_time = np.mean(processing_times[-20:])
            print(f"[{idx+1:3d}/{len(df)}] Avg processing time (last 20): {avg_time:.2f} ms")
    
    end_total = time.time()
    total_time = end_total - start_total
    
    # Calculate statistics
    print("\n" + "="*70)
    print("PERFORMANCE RESULTS")
    print("="*70)
    
    # Throughput
    throughput = len(df) / total_time
    print(f"\nTiming Metrics:")
    print(f"   Total processing time:     {total_time:.3f} seconds")
    print(f"   Samples processed:         {len(df)}")
    print(f"   Throughput:                {throughput:.2f} samples/second")
    print(f"   Average sample time:       {np.mean(processing_times):.2f} ms")
    
    # Latency breakdown
    print(f"\nLatency Analysis:")
    print(f"   Minimum latency:           {np.min(processing_times):.2f} ms")
    print(f"   Maximum latency:           {np.max(processing_times):.2f} ms")
    print(f"   Mean latency:              {np.mean(processing_times):.2f} ms")
    print(f"   Median latency:            {np.median(processing_times):.2f} ms")
    print(f"   99th percentile:           {np.percentile(processing_times, 99):.2f} ms")
    
    # Hazard-specific latency
    if hazard_latencies:
        print(f"\nHazard Detection Response Times:")
        print(f"   Hazard samples detected:   {len(hazard_latencies)}")
        print(f"   Avg hazard latency:        {np.mean(hazard_latencies):.2f} us")
        print(f"   Max hazard latency:        {np.max(hazard_latencies):.2f} us")
        
        # Check if meets requirements
        max_latency_ms = np.max(hazard_latencies) / 1000
        requirement_met = "PASS" if max_latency_ms < 200 else "FAIL"
        print(f"   Requirement (<200ms):      {requirement_met}")
    else:
        print(f"\nHazard Detection Response Times:")
        print(f"   No hazards detected in test set")
    
    # Detection statistics
    hazard_count = sum(1 for r in results_list if r['hazard'])
    detection_rate = (hazard_count / len(results_list)) * 100
    
    print(f"\nDetection Statistics:")
    print(f"   Total samples:             {len(results_list)}")
    print(f"   Hazards detected:          {hazard_count}")
    print(f"   Detection rate:            {detection_rate:.2f}%")
    print(f"   Safe samples:              {len(results_list) - hazard_count}")
    
    # Risk level distribution
    risk_counts = {}
    for r in results_list:
        risk = r['risk_level']
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    print(f"\nRisk Level Distribution:")
    for risk in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = risk_counts.get(risk, 0)
        percentage = (count / len(results_list)) * 100
        print(f"   {risk:12s}: {count:3d} ({percentage:5.1f}%)")
    
    # Save performance report
    print("\nSaving performance report...")
    report_path = os.path.join(RESULTS_DIR, 'performance_report.txt')
    with open(report_path, 'w') as f:
        f.write("FPGA Transport Safety Monitoring - Performance Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples Processed: {len(df)}\n")
        f.write(f"Total Time: {total_time:.3f} seconds\n")
        f.write(f"Throughput: {throughput:.2f} samples/sec\n\n")
        f.write("Latency Statistics (ms):\n")
        f.write(f"  Min:    {np.min(processing_times):.2f}\n")
        f.write(f"  Max:    {np.max(processing_times):.2f}\n")
        f.write(f"  Mean:   {np.mean(processing_times):.2f}\n")
        f.write(f"  Median: {np.median(processing_times):.2f}\n")
        f.write(f"  99th %: {np.percentile(processing_times, 99):.2f}\n\n")
        f.write("Detection Statistics:\n")
        f.write(f"  Total Samples:    {len(results_list)}\n")
        f.write(f"  Hazards Detected: {hazard_count}\n")
        f.write(f"  Detection Rate:   {detection_rate:.2f}%\n\n")
        f.write("Risk Level Distribution:\n")
        for risk in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = risk_counts.get(risk, 0)
            percentage = (count / len(results_list)) * 100
            f.write(f"  {risk:12s}: {count:3d} ({percentage:5.1f}%)\n")
    
    print(f"Performance report saved to: {report_path}")
    
    # Save detailed results
    results_df = pd.DataFrame(results_list)
    results_csv = os.path.join(RESULTS_DIR, 'performance_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Detailed results saved to: {results_csv}")
    
    # Create latency histogram data
    hist, bins = np.histogram(processing_times, bins=20)
    hist_path = os.path.join(RESULTS_DIR, 'latency_histogram.csv')
    with open(hist_path, 'w') as f:
        f.write("bin_start,bin_end,count\n")
        for i in range(len(hist)):
            f.write(f"{bins[i]:.2f},{bins[i+1]:.2f},{hist[i]}\n")
    
    print(f"Latency histogram saved to: {hist_path}")
    
    print("\n" + "="*70)
    print("PERFORMANCE TEST COMPLETE!")
    print("="*70 + "\n")
    
    return {
        'throughput': throughput,
        'mean_latency': np.mean(processing_times),
        'max_latency': np.max(processing_times),
        'detection_rate': detection_rate
    }

if __name__ == "__main__":
    results = run_performance_test()

