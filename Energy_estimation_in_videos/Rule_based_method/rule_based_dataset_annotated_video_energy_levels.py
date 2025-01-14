import cv2
import numpy as np
from typing import Dict, List
import json
from pathlib import Path

class SimpleEnergyEstimator:
    def __init__(self, threshold: int = 30):
        """
        Initialize the simple energy estimator
        Args:
            threshold (int): Pixel difference threshold (0-255)
        """
        self.threshold = threshold
        
        # Define energy levels
        self.energy_levels = {
            0: (0.0, 0.1),    # Static
            1: (0.1, 0.3),    # Very Low
            2: (0.3, 0.5),    # Low
            3: (0.5, 0.7),    # Medium
            4: (0.7, 0.9),    # High
            5: (0.9, 1.0)     # Very High
        }

    def estimate_energy(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict:
        """
        Estimate energy between two frames
        Args:
            frame1: First frame
            frame2: Second frame
        Returns:
            Dictionary containing energy value and level
        """
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Count changed pixels
        changed_pixels = np.sum(diff > self.threshold)
        
        # Calculate energy as percentage of changed pixels
        energy_value = changed_pixels / diff.size
        
        # Determine energy level
        energy_level = self.get_energy_level(energy_value)
        
        return {
            'energy_value': float(energy_value),
            'energy_level': energy_level
        }

    def get_energy_level(self, energy_value: float) -> int:
        """
        Map energy value to discrete level
        Args:
            energy_value: Normalized energy value (0-1)
        Returns:
            Energy level (0-5)
        """
        for level, (min_val, max_val) in self.energy_levels.items():
            if min_val <= energy_value <= max_val:
                return level
        return 5  # Return maximum level if above all ranges

def process_video(video_path: str, output_path: str = None, 
                 save_annotations: bool = True) -> List[Dict]:
    """
    Process video and estimate energy
    Args:
        video_path: Path to input video
        output_path: Path for output video (optional)
        save_annotations: Whether to save annotations to JSON
    Returns:
        List of frame energy measurements
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize energy estimator
    estimator = SimpleEnergyEstimator()
    
    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (frame_width, frame_height))
    
    # Process video frames
    frame_energies = []
    prev_frame = None
    frame_count = 0
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Show progress every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.1f}% complete", end='\r')
            
        if prev_frame is not None:
            # Estimate energy
            energy_info = estimator.estimate_energy(prev_frame, frame)
            energy_info['frame_number'] = frame_count
            frame_energies.append(energy_info)
            
            # Draw energy information on frame
            if output_path:
                energy_text = f"Energy: {energy_info['energy_value']:.3f}"
                level_text = f"Level: {energy_info['energy_level']}"
                
                cv2.putText(frame, energy_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, level_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw energy bar
                bar_height = int(300 * energy_info['energy_value'])
                cv2.rectangle(frame, (20, 300), (50, 300 - bar_height), 
                            (0, 255, 0), -1)
                
                out.write(frame)
        
        prev_frame = frame.copy()
        frame_count += 1
        
    # Release resources
    cap.release()
    if output_path:
        out.release()
    
    # Save annotations if requested
    if save_annotations:
        annotation_path = Path(video_path).with_suffix('.json')
        save_energy_annotations(annotation_path, frame_energies)
    
    return frame_energies

def save_energy_annotations(output_path: str, frame_energies: List[Dict]):
    """
    Save energy annotations to JSON file
    Args:
        output_path: Path to save annotations
        frame_energies: List of frame energy measurements
    """
    # Calculate summary statistics
    energy_values = [f['energy_value'] for f in frame_energies]
    energy_levels = [f['energy_level'] for f in frame_energies]
    
    annotations = {
        'frame_annotations': frame_energies,
        'summary': {
            'mean_energy': float(np.mean(energy_values)),
            'max_energy': float(np.max(energy_values)),
            'min_energy': float(np.min(energy_values)),
            'std_energy': float(np.std(energy_values)),
            'level_distribution': {
                str(level): energy_levels.count(level) / len(energy_levels)
                for level in range(6)
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)

def main():
    """Main function demonstrating usage"""
    # Process video with full paths
    video_path = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/pingPong.mp4"
    output_path = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos_output/pingPong_rule_base_energy_estimation.mp4"
    
    try:
        print(f"Processing video: {video_path}")
        frame_energies = process_video(video_path, output_path)
        
        # Print summary statistics
        energy_values = [f['energy_value'] for f in frame_energies]
        print("\nVideo Energy Statistics:")
        print(f"Mean Energy: {np.mean(energy_values):.3f}")
        print(f"Max Energy: {np.max(energy_values):.3f}")
        print(f"Min Energy: {np.min(energy_values):.3f}")
        print(f"Std Energy: {np.std(energy_values):.3f}")
        
        print("\nProcessing complete!")
        print(f"Output video saved to: {output_path}")
        print(f"Annotations saved to: {Path(video_path).with_suffix('.json')}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()
