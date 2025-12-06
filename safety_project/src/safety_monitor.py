#!/usr/bin/env python3
"""
Real-time Safety Monitoring System for KV260
Fixed to handle feature name mismatch
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys

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
        # Get test data to determine correct mapping
        if test_file:
            test_data = pd.read_csv(test_file)
            feature_cols = [col for col in test_data.columns if col.startswith('feature_')]
            print(f"Test data has features: {feature_cols}")
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
        print(f"Features mapped: {self.feature_names[:3]}... (showing first 3)")
        
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
    
    def print_result(self, result):
        """Print monitoring result"""
        status_symbol = "[HAZARD]" if result['hazard_detected'] else "[SAFE]"
        alarm_symbol = "ALARM TRIGGERED" if result['alarm_triggered'] else ""
        
        print(f"\n{status_symbol} [{result['timestamp']}] Sample {result['sample_id']}")
        print(f"   Hazard: {result['hazard_detected']} | "
              f"Probability: {result['probability']:.3f} | "
              f"Risk: {result['risk_level']}")
        if alarm_symbol:
            print(f"   Consecutive Alerts: {result['consecutive_alerts']} {alarm_symbol}")

def test_with_file(monitor, test_file):
    """Test monitor with CSV file"""
    print(f"\nLoading test data from: {test_file}")
    test_data = pd.read_csv(test_file)
    print(f"Loaded {len(test_data)} test samples\n")
    
    print("="*60)
    print("Starting Safety Monitoring Test")
    print("="*60)
    
    results = []
    for idx, row in test_data.iterrows():
        sensor_data = {k: v for k, v in row.to_dict().items() 
                      if isinstance(v, (int, float))}
        
        result = monitor.process_sample(sensor_data, sample_id=idx)
        results.append(result)
        
        monitor.print_result(result)
        time.sleep(0.05)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    hazard_count = sum(1 for r in results if r['hazard_detected'])
    alarm_count = sum(1 for r in results if r['alarm_triggered'])
    prob_values = [r['probability'] for r in results]
    
    print(f"Total Samples: {len(results)}")
    print(f"Hazards Detected: {hazard_count} ({hazard_count/len(results)*100:.1f}%)")
    print(f"Alarms Triggered: {alarm_count}")
    print(f"\nProbability Statistics:")
    print(f"  Min: {np.min(prob_values):.4f}")
    print(f"  Max: {np.max(prob_values):.4f}")
    print(f"  Mean: {np.mean(prob_values):.4f}")
    print(f"  Std: {np.std(prob_values):.4f}")
    
    return results

if __name__ == "__main__":
    MODEL_PATH = "../models/best_model.pkl"
    SCALER_PATH = "../data/scaler_parameters.csv"
    TEST_FILE = "../data/fpga_test_vectors.csv"
    
    print("="*60)
    print("KV260 Safety Monitoring System")
    print("="*60)
    
    # Initialize with test file for proper feature mapping
    monitor = SafetyMonitor(MODEL_PATH, SCALER_PATH, TEST_FILE)
    
    print("\nSystem initialized successfully!")
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = TEST_FILE
    
    results = test_with_file(monitor, test_file)
    
    results_file = f"../results/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    

