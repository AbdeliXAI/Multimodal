#!/usr/bin/env python3
"""
===================================================================================================
Enhanced Multi-Feature Energy Estimator running 68 short videos of 10-to-30sec from movies trailer.
Please the Accuracy_score_and_confusion_matrix.py for Results of the Analysis.
===================================================================================================

This module implements a comprehensive approach combining multiple features
for estimating motion energy levels in video sequences.

Energy Level Classification:
- LOW: energy < 0.3 (minimal motion/changes between frames)
- MEDIUM: 0.3 <= energy < 0.7 (moderate motion/changes)
- HIGH: energy >= 0.7 (significant motion/changes)

The output will look like this:
-------------------------------
Processing video1...
video1: HIGH (Average Energy: 0.823)

Processing video2...
video2: MEDIUM (Average Energy: 0.456)

Processing video3...
video3: LOW (Average Energy: 0.234)


The energy estimation combines:
-------------------------------
1. Pixel differences
2. Optical flow analysis
3. Block motion detection

Final energy level is determined by averaging the energy across all frames
and classifying into one of three levels: LOW, MEDIUM, or HIGH.
"""

"""calculate_pixel_difference
Calculate frame-to-frame difference based on pixel values.
Process:
    1. Convert frames to grayscale
    2. Calculate absolute difference
    3. Normalize and return mean energy
"""

"""calculate_optical_flow
Calculate motion energy using optical flow analysis.
Process:
    1. Convert frames to grayscale
    2. Calculate Farneback optical flow
    3. Compute flow magnitude
"""

"""calculate_block_motion
Calculate motion energy using block-based analysis.
Process:
    1. Determine optimal block size
    2. Divide frames into blocks
    3. Calculate block-wise differences
    4. Normalize and return energy
"""

"""enhanced_multi_feature_energy
Calculate overall energy using multiple features with adaptive weighting.
Process:
    1. Calculate individual energies
    2. Compute adaptive weights
    3. Combine weighted energies
"""

# System imports for environment information and error handling
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    print("Script starting...")
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    import os
    from typing import Tuple, List, Dict
    print("All imports completed")
except Exception as e:
    print(f"Error during imports: {str(e)}")
    exit(1)

class EnhancedEnergyEstimator:
    """
    A class for estimating visual energy levels in video frames using multiple features.
    Combines pixel differences, optical flow, and block motion analysis.
    """
    
    def __init__(self):
        """
        Initialize the energy estimator with default parameters.
        - Sets block size ranges for motion analysis
        - Configures optical flow parameters
        """
        print("Initializing EnhancedEnergyEstimator...")
        self.min_block_size = 8      # Minimum size for block analysis
        self.max_block_size = 32     # Maximum size for block analysis
        self.flow_params = dict(     # Optical flow parameters
            pyr_scale=0.5,           # Image scale (<1) to build pyramids for each image
            levels=3,                # Number of pyramid layers
            winsize=15,              # Averaging window size
            iterations=3,            # Number of iterations at each pyramid level
            poly_n=5,                # Size of pixel neighborhood for polynomial expansion
            poly_sigma=1.2,          # Standard deviation for Gaussian used to smooth derivatives
            flags=0                  # Optional flags
        )
        
    def calculate_pixel_difference(self, frame1: np.ndarray, 
                                 frame2: np.ndarray) -> float:
        """
        Calculate frame-to-frame difference based on pixel values.
        Args:
            frame1: First video frame
            frame2: Second video frame
        Returns:
            float: Normalized mean difference energy (0-1)
        Process:
            1. Convert frames to grayscale
            2. Calculate absolute difference
            3. Normalize and return mean energy
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff) / 255.0

    def calculate_optical_flow(self, frame1: np.ndarray, 
                             frame2: np.ndarray) -> float:
        """
        Calculate motion energy using optical flow analysis.
        Args:
            frame1: First video frame
            frame2: Second video frame
        Returns:
            float: Mean motion magnitude
        Process:
            1. Convert frames to grayscale
            2. Calculate Farneback optical flow
            3. Compute flow magnitude
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            self.flow_params['pyr_scale'],
            self.flow_params['levels'],
            self.flow_params['winsize'],
            self.flow_params['iterations'],
            self.flow_params['poly_n'],
            self.flow_params['poly_sigma'],
            self.flow_params['flags']
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)

    def calculate_block_motion(self, frame1: np.ndarray, 
                             frame2: np.ndarray) -> float:
        """
        Calculate motion energy using block-based analysis.
        Args:
            frame1: First video frame
            frame2: Second video frame
        Returns:
            float: Normalized block motion energy (0-1)
        Process:
            1. Determine optimal block size
            2. Divide frames into blocks
            3. Calculate block-wise differences
            4. Normalize and return energy
        """
        height, width = frame1.shape[:2]
        block_size = self.determine_optimal_block_size(frame1)
        energy = 0.0
        
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block1 = frame1[y:y+block_size, x:x+block_size]
                block2 = frame2[y:y+block_size, x:x+block_size]
                diff = np.mean(np.abs(block1.astype(float) - block2.astype(float)))
                energy += diff
                
        total_blocks = ((height // block_size) * (width // block_size))
        return energy / (total_blocks * 255.0)

    def determine_optimal_block_size(self, frame: np.ndarray) -> int:
        """
        Determine the optimal block size based on frame content.
        Args:
            frame: Input video frame
        Returns:
            int: Optimal block size
        Process:
            1. Convert to grayscale
            2. Detect edges
            3. Calculate edge density
            4. Choose block size based on complexity
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        if edge_density > 0.1:           # High complexity
            return self.min_block_size
        elif edge_density > 0.05:        # Medium complexity
            return (self.min_block_size + self.max_block_size) // 2
        else:                           # Low complexity
            return self.max_block_size

    def calculate_adaptive_weights(self, pixel_energy: float, 
                                 flow_energy: float, 
                                 block_energy: float) -> List[float]:
        """
        Calculate adaptive weights for combining different energy measures.
        Args:
            pixel_energy: Pixel-based energy
            flow_energy: Optical flow energy
            block_energy: Block motion energy
        Returns:
            List[float]: Normalized weights for each energy type
        Process:
            1. Calculate total energy
            2. Compute relative contributions
            3. Normalize weights
        """
        total_energy = pixel_energy + flow_energy + block_energy
        if total_energy == 0:
            return [0.33, 0.33, 0.34]  # Equal weights if no energy
            
        weights = [
            pixel_energy / total_energy,
            flow_energy / total_energy,
            block_energy / total_energy
        ]
        
        sum_weights = sum(weights)
        return [w / sum_weights for w in weights]

    def enhanced_multi_feature_energy(self, frame1: np.ndarray, 
                                    frame2: np.ndarray) -> Dict:
        """
        Calculate overall energy using multiple features with adaptive weighting.
        Args:
            frame1: First video frame
            frame2: Second video frame
        Returns:
            Dict: Complete energy analysis including:
                - total_energy: Combined energy score
                - individual energies (pixel, flow, block)
                - weights used
        Process:
            1. Calculate individual energies
            2. Compute adaptive weights
            3. Combine weighted energies
        """
        pixel_energy = self.calculate_pixel_difference(frame1, frame2)
        flow_energy = self.calculate_optical_flow(frame1, frame2)
        block_energy = self.calculate_block_motion(frame1, frame2)
        
        weights = self.calculate_adaptive_weights(
            pixel_energy, flow_energy, block_energy
        )
        
        total_energy = (weights[0] * pixel_energy + 
                       weights[1] * flow_energy + 
                       weights[2] * block_energy)
        
        return {
            'total_energy': total_energy,
            'pixel_energy': pixel_energy,
            'flow_energy': flow_energy,
            'block_energy': block_energy,
            'weights': weights
        }

    def classify_energy_level(self, energy: float) -> str:
        """
        Classify energy value into three discrete levels (LOW, MEDIUM, HIGH).
        Args:
            energy: Combined energy value
        Returns:
            str: Energy level classification (LOW, MEDIUM, HIGH)
        Process:
            1. Compare energy with thresholds
            2. Return appropriate level
        """
        if energy < 0.3:
            return "LOW"
        elif energy < 0.7:
            return "MEDIUM"
        return "HIGH"  # For energy >= 0.7

if __name__ == "__main__":
    """
    Main execution block for processing video files.
    Process:
        1. Initialize estimator
        2. Process each video file
        3. Calculate and display statistics
        4. Save results
    """
    print("Main block starting...")
    print("Starting video processing...")
    estimator = EnhancedEnergyEstimator()
    
    # Define base directory for videos
    video_dir = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/Evaluate_Rule_Based_Methods/short_movies_trailer"
    
    # Automatically create video_paths dictionary from all videos in directory
    video_paths = {}
    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_name = os.path.splitext(video_file)[0]
            video_paths[video_name] = os.path.join(video_dir, video_file)
    
    print("\nFound videos:")
    for name, path in video_paths.items():
        print(f"- {name}: {path}")
    
    print("\nProcessing videos...")
    # Process each video
    for video_name, video_path in video_paths.items():
        print(f"\nProcessing {video_name}...")
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            continue
        
        # Process video and collect results
        results = []
        cap = cv2.VideoCapture(video_path)
        
        ret, prev_frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame of {video_name}")
            continue
            
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
                
            # Calculate energy
            energy_data = estimator.enhanced_multi_feature_energy(prev_frame, curr_frame)
            results.append(energy_data['total_energy'])
            prev_frame = curr_frame
        
        cap.release()
        
        # Calculate average energy and final classification for the entire video
        if results:
            avg_energy = sum(results) / len(results)
            final_level = estimator.classify_energy_level(avg_energy)
            print(f"Results for {video_name}:")
            print(f"- Energy Level: {final_level}")
            print(f"- Average Energy: {avg_energy:.3f}")
            print("-" * 50)
        else:
            print(f"{video_name}: Could not process video") 
