import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
import cv2
import numpy as np
from PIL import Image
import time
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional

class VideoMAEProcessor:
    def __init__(self, model_name='MCG-NJU/videomae-base-ssv2'):
        """Initialize VideoMAE model and configurations"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize VideoMAE
        self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Frame processing settings
        self.num_frames = 16  # VideoMAE default
        self.image_size = 224  # Default size

class VideoClipDataset(Dataset):
    def __init__(self, frames: List[np.ndarray], feature_extractor):
        self.frames = frames
        self.feature_extractor = feature_extractor

    def __len__(self):
        return 1  # Process one clip at a time

    def __getitem__(self, idx):
        # Process frames using VideoMAE feature extractor
        inputs = self.feature_extractor(
            self.frames,
            return_tensors="pt",
            padding=True
        )
        return inputs.pixel_values

class EnergyDetector:
    def __init__(self):
        self.processor = VideoMAEProcessor()
        self.energy_threshold = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """Process video and detect energy levels"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path, 
                fourcc, 
                fps,
                (int(cap.get(3)), int(cap.get(4)))
            )
        
        frames_buffer = []
        energy_scores = []
        timestamps = []
        frame_count = 0
        
        print("Starting video analysis...")
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames_buffer.append(frame)
                frame_count += 1
                
                # Process when buffer reaches required size
                if len(frames_buffer) == self.processor.num_frames:
                    # Create dataset from frame buffer
                    dataset = VideoClipDataset(
                        frames_buffer,
                        self.processor.feature_extractor
                    )
                    
                    # Process frames
                    with torch.no_grad():
                        inputs = dataset[0].to(self.processor.device)
                        outputs = self.processor.model(inputs)
                        
                        # Calculate energy score
                        energy_score = self.calculate_energy_score(outputs)
                        energy_scores.append(energy_score)
                        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                        
                        # Visualize if output path provided
                        if output_path:
                            self.visualize_frame(
                                frames_buffer[-1],
                                energy_score,
                                frame_count / fps,
                                out
                            )
                    
                    # Slide window
                    frames_buffer.pop(0)
                
                # Progress update
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {frame_count} frames. "
                          f"Time elapsed: {elapsed:.2f}s")
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        return self.generate_analysis_report(energy_scores, timestamps)
    
    def calculate_energy_score(self, model_outputs) -> float:
        """Calculate energy score from model outputs"""
        # Extract features and calculate energy
        features = model_outputs.logits
        energy = torch.mean(torch.abs(features))
        return torch.sigmoid(energy).item()
    
    def visualize_frame(self, frame, energy_score, timestamp, writer):
        """Add visualizations to frame"""
        height, width = frame.shape[:2]
        
        # Add timestamp
        time_str = f"Time: {timestamp:.1f}s"
        cv2.putText(
            frame, time_str,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add energy level
        energy_type = self.get_energy_type(energy_score)
        color = self.get_energy_color(energy_score)
        
        cv2.putText(
            frame,
            f"Energy: {energy_type} ({energy_score:.2f})",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Add energy bar
        bar_width = int(width * energy_score)
        cv2.rectangle(
            frame,
            (0, height - 5),
            (bar_width, height),
            color,
            -1
        )
        
        writer.write(frame)
    
    def get_energy_type(self, score: float) -> str:
        """Determine energy type based on score"""
        if score < self.energy_threshold['low']:
            return "Low"
        elif score < self.energy_threshold['medium']:
            return "Medium"
        else:
            return "High"
    
    def get_energy_color(self, score: float) -> tuple:
        """Get color based on energy score"""
        if score < self.energy_threshold['low']:
            return (0, 255, 0)  # Green
        elif score < self.energy_threshold['medium']:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red
    
    def generate_analysis_report(self, energy_scores: List[float], 
                               timestamps: List[float]) -> Dict:
        """Generate detailed analysis report"""
        return {
            "Analysis Summary": {
                "Duration": timestamps[-1] - timestamps[0],
                "Average Energy": np.mean(energy_scores),
                "Peak Energy": np.max(energy_scores),
                "Energy Distribution": {
                    "Low": self.calculate_percentage(energy_scores, 'low'),
                    "Medium": self.calculate_percentage(energy_scores, 'medium'),
                    "High": self.calculate_percentage(energy_scores, 'high')
                },
                "Key Moments": self.find_key_moments(energy_scores, timestamps)
            }
        }
    
    def calculate_percentage(self, scores: List[float], level: str) -> float:
        """Calculate percentage of time spent at each energy level"""
        threshold = self.energy_threshold[level]
        if level == 'low':
            count = sum(1 for s in scores if s < threshold)
        elif level == 'medium':
            count = sum(1 for s in scores if threshold <= s < self.energy_threshold['high'])
        else:
            count = sum(1 for s in scores if s >= self.energy_threshold['high'])
        return (count / len(scores)) * 100
    
    def find_key_moments(self, scores: List[float], 
                        timestamps: List[float]) -> List[Dict]:
        """Identify key moments in the video"""
        key_moments = []
        window_size = 5  # Number of frames to consider
        
        for i in range(len(scores) - window_size):
            window_avg = np.mean(scores[i:i+window_size])
            if window_avg > self.energy_threshold['high']:
                key_moments.append({
                    "timestamp": timestamps[i],
                    "duration": window_size / 30,  # Assuming 30 fps
                    "energy_level": window_avg
                })
        
        return key_moments

def main():
    # Example usage
    video_path = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/video_ed_for_multimodal.mp4"
    output_path = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/video_action_energy_recognition/outputs/output_analyzed_video.mp4"
    
    detector = EnergyDetector()
    
    print("Starting video analysis...")
    start_time = time.time()
    
    try:
        analysis = detector.process_video(video_path, output_path)
        
        # Print analysis results
        print("\nAnalysis Complete!")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        print("\nResults:")
        print(f"Average Energy: {analysis['Analysis Summary']['Average Energy']:.2f}")
        print(f"Peak Energy: {analysis['Analysis Summary']['Peak Energy']:.2f}")
        print("\nEnergy Distribution:")
        for level, percentage in analysis['Analysis Summary']['Energy Distribution'].items():
            print(f"{level}: {percentage:.1f}%")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
