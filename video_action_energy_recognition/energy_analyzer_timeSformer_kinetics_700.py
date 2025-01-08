import torch
import torch.nn as nn
import torchvision.transforms as transforms
from timesformer.models.vit import TimeSformer
import cv2
import numpy as np
from PIL import Image
import time
from torch.utils.data import Dataset, DataLoader

class VideoProcessor:
    def __init__(self, model_path='pretrained_timesformer_k700.pth'):
        """Initialize TimeSformer model and configurations"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize TimeSformer
        self.model = TimeSformer(
            img_size=224,
            num_classes=700,
            num_frames=8,
            attention_type='divided_space_time',
            pretrained=True
        )
        
        # Load pretrained weights
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], 
                              std=[0.225, 0.225, 0.225])
        ])

class VideoFrameDataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(Image.fromarray(frame))
        return frame

def process_video(video_path, output_path=None):
    """Process video and detect energy levels"""
    processor = VideoProcessor()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (int(cap.get(3)), int(cap.get(4))))
    
    frames_buffer = []
    energy_scores = []
    frame_count = 0
    
    print("Processing video...")
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_buffer.append(frame)
            frame_count += 1
            
            # Process when buffer reaches window size
            if len(frames_buffer) == 8:  # TimeSformer's temporal window
                # Prepare batch
                batch = VideoFrameDataset(frames_buffer, processor.transform)
                loader = DataLoader(batch, batch_size=8, shuffle=False)
                
                # Process batch
                with torch.no_grad():
                    for frames in loader:
                        frames = frames.to(processor.device)
                        features = processor.model.forward_features(frames)
                        energy_score = calculate_energy_score(features)
                        energy_scores.append(energy_score)
                        
                        # Visualize and save if required
                        if output_path:
                            visualize_frame(frames_buffer[-1], energy_score, out)
                
                frames_buffer.pop(0)
            
            # Print progress
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {frame_count} frames. "
                      f"Time elapsed: {elapsed_time:.2f}s")
                
    finally:
        cap.release()
        if output_path:
            out.release()
            
    return energy_scores

def calculate_energy_score(features):
    """Calculate energy score from features"""
    # Convert features to energy score (0-1)
    energy = torch.mean(torch.abs(features))
    return torch.sigmoid(energy).item()

def visualize_frame(frame, energy_score, writer):
    """Visualize energy score on frame"""
    # Add energy score visualization
    height, width = frame.shape[:2]
    energy_color = get_energy_color(energy_score)
    cv2.putText(frame, f"Energy: {energy_score:.2f}", 
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, energy_color, 2)
    
    # Add energy bar
    bar_width = int(width * energy_score)
    cv2.rectangle(frame, (0, height - 5), 
                 (bar_width, height), energy_color, -1)
    
    writer.write(frame)

def get_energy_color(score):
    """Get color based on energy score"""
    if score < 0.3:
        return (0, 255, 0)  # Green for low energy
    elif score < 0.7:
        return (0, 255, 255)  # Yellow for medium energy
    else:
        return (0, 0, 255)  # Red for high energy

def main():
    # Example usage
    video_path = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/video_ed_for_multimodal.mp4"
    output_path = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/video_action_energy_recognition/outputs/output_video.mp4"
    
    print("Starting video processing...")
    start_time = time.time()
    
    try:
        energy_scores = process_video(video_path, output_path)
        
        # Save energy scores
        np.save('energy_scores.npy', energy_scores)
        
        # Print statistics
        print("\nProcessing complete!")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        print(f"Average energy score: {np.mean(energy_scores):.2f}")
        print(f"Peak energy score: {np.max(energy_scores):.2f}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
