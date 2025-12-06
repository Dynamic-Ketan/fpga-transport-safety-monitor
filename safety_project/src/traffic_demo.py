#!/usr/bin/env python3
"""
Simple Traffic Violation Demo
Uses motion + vehicle density to flag anomalous traffic
"""

import cv2
import numpy as np
from datetime import datetime
import time
import os


class SimpleTrafficDemo:
    def __init__(self, input_video='../data/traffic_sample.mp4', 
                 output_video='../results/traffic_violations_simple.mp4'):
        self.input_video = input_video
        self.output_video = output_video
        self.violations = 0
        self.violation_frames = []
        
    def detect_vehicles(self, frame):
        """Detect vehicles using edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        vehicles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 15000:
                x, y, w, h = cv2.boundingRect(cnt)
                vehicles.append((x, y, w, h, area))
        
        return vehicles
    
    def calculate_motion(self, frame, prev_frame):
        """Calculate frame motion intensity"""
        if prev_frame is None:
            return 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray, prev_gray)
        motion_score = np.mean(diff) / 255.0
        
        return motion_score
    
    def is_violation(self, vehicles, motion, frame_idx):
        """Determine if frame shows violation"""
        
        if len(vehicles) > 15:
            return True, "Heavy congestion detected", 0.85
        
        if motion > 0.08 and len(vehicles) > 8:
            return True, "Erratic traffic pattern", 0.78
        
        if motion > 0.12:
            return True, "Extreme motion detected", 0.92
        
        red_phases = [(100, 130), (180, 210)]
        for start, end in red_phases:
            if start <= frame_idx <= end and len(vehicles) > 5 and motion > 0.05:
                return True, "Potential red light violation", 0.72
        
        return False, "", 0.0
    
    def process(self):
        print("\n" + "="*70)
        print("SIMPLE TRAFFIC VIOLATION DEMO")
        print("="*70 + "\n")
        
        os.makedirs('../results', exist_ok=True)
        
        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            print(f"Cannot open: {self.input_video}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Input  : {self.input_video}")
        print(f"Output : {self.output_video}")
        print(f"Video  : {width}x{height} @ {fps} FPS")
        print(f"Frames : {total_frames}\n")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))
        
        frame_idx = 0
        prev_frame = None
        t0 = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            vehicles = self.detect_vehicles(frame)
            motion = self.calculate_motion(frame, prev_frame)
            is_viol, reason, confidence = self.is_violation(vehicles, motion, frame_idx)
            
            if is_viol:
                self.violations += 1
                self.violation_frames.append({
                    'frame': frame_idx,
                    'reason': reason,
                    'confidence': confidence,
                    'vehicles': len(vehicles),
                    'motion': motion
                })
            
            disp = frame.copy()
            
            if is_viol:
                banner_color = (0, 0, 255)
                banner_text = f"WARNING: {reason}"
            else:
                banner_color = (0, 200, 0)
                banner_text = "NORMAL TRAFFIC"
            
            cv2.rectangle(disp, (0, 0), (width, 80), banner_color, -1)
            cv2.putText(disp, banner_text, (30, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            for (x, y, w, h, area) in vehicles:
                color = (0, 0, 255) if is_viol else (0, 255, 0)
                cv2.rectangle(disp, (x, y), (x+w, y+h), color, 2)
            
            info_y = 110
            cv2.putText(disp, f"Frame: {frame_idx}/{total_frames}",
                       (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(disp, f"Vehicles: {len(vehicles)}",
                       (20, info_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(disp, f"Motion: {motion*100:.1f}%",
                       (20, info_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(disp, f"Total Violations: {self.violations}",
                       (20, info_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            if is_viol:
                cv2.putText(disp, f"Confidence: {confidence*100:.0f}%",
                           (20, info_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(disp, ts, (width-150, height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(disp)
            
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                pct = frame_idx / total_frames * 100
                print(f"\rProcessing: [{frame_idx:3d}/{total_frames}] {pct:5.1f}% | "
                      f"Violations: {self.violations}", end='')
            
            prev_frame = frame
        
        print()
        cap.release()
        out.release()
        
        dt = time.time() - t0
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Frames processed : {frame_idx}")
        print(f"Violations found : {self.violations}")
        print(f"Violation rate   : {self.violations/frame_idx*100:.2f}%")
        print(f"Processing time  : {dt:.1f} seconds ({frame_idx/dt:.1f} FPS)")
        print(f"\nVideo saved      : {self.output_video}")
        
        if self.violation_frames:
            report = self.output_video.replace('.mp4', '_report.txt')
            with open(report, 'w') as f:
                f.write("TRAFFIC VIOLATION DETECTION REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Input video: {self.input_video}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total frames: {frame_idx}\n")
                f.write(f"Violations: {self.violations}\n")
                f.write(f"Violation rate: {self.violations/frame_idx*100:.2f}%\n\n")
                f.write("Violation Events:\n")
                f.write("-"*70 + "\n")
                f.write(f"{'Frame':<8} {'Reason':<35} {'Conf%':<8} {'Vehicles':<10} {'Motion%':<10}\n")
                f.write("-"*70 + "\n")
                
                for v in self.violation_frames:
                    f.write(f"{v['frame']:<8} {v['reason']:<35} "
                           f"{v['confidence']*100:<8.0f} {v['vehicles']:<10} "
                           f"{v['motion']*100:<10.1f}\n")
            
            print(f"Report saved     : {report}")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    demo = SimpleTrafficDemo(
        input_video='../data/traffic_sample.mp4',
        output_video='../results/traffic_violations_simple.mp4'
    )
    demo.process()
