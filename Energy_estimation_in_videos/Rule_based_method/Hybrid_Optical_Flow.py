#!/usr/bin/env python3
"""
Hybrid Optical Flow Energy Estimator
-----------------------------------
This module implements a hybrid approach combining sparse and dense optical flow
for estimating motion energy levels in video sequences.
"""

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

class HybridOpticalFlowEstimator:
    """
    A class that combines sparse and dense optical flow techniques for motion energy estimation.
    Uses both feature tracking and pixel-wise flow calculation for robust energy detection.
    """
    
    def __init__(self):
        """
        Initialize parameters for both sparse and dense optical flow calculations.
        - feature_params: Parameters for detecting good features to track
        - flow_params: Parameters for Lucas-Kanade optical flow
        - dense_flow_params: Parameters for Farneback dense optical flow
        """
        print("Initializing HybridOpticalFlowEstimator...")
        # Parameters for feature detection in sparse flow
        self.feature_params = dict(
            maxCorners=100,    # Maximum number of corners to detect
            qualityLevel=0.3,  # Minimum quality level for corner detection
            minDistance=7,     # Minimum distance between detected corners
            blockSize=7        # Size of block for corner detection
        )
        
        # Parameters for Lucas-Kanade optical flow
        self.flow_params = dict(
            winSize=(15, 15),  # Size of search window
            maxLevel=2,        # Number of pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria
        )
        
        # Parameters for Farneback dense optical flow
        self.dense_flow_params = dict(
            pyr_scale=0.5,     # Image scale (<1) for building pyramids
            levels=3,          # Number of pyramid levels
            winsize=15,        # Averaging window size
            iterations=3,      # Number of iterations at each pyramid level
            poly_n=5,          # Size of pixel neighborhood
            poly_sigma=1.2,    # Standard deviation for Gaussian
            flags=0            # Optional flags
        )
        
    def detect_sparse_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect good features (corners) in the frame for sparse optical flow tracking.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Array of detected feature points
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            **self.feature_params
        )
        return features
        
    def calculate_sparse_flow(self, frame1: np.ndarray, 
                            frame2: np.ndarray, 
                            features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate sparse optical flow using Lucas-Kanade method.
        
        Args:
            frame1: First frame
            frame2: Second frame
            features: Points to track from first frame
            
        Returns:
            Tuple of (new feature positions, old feature positions)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate flow using Lucas-Kanade
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            features,
            None,
            **self.flow_params
        )
        
        # Select good points (where status == 1)
        good_new = new_features[status == 1]
        good_old = features[status == 1]
        
        return good_new, good_old
        
    def calculate_dense_flow(self, frame1: np.ndarray, 
                           frame2: np.ndarray, 
                           mask: np.ndarray = None) -> np.ndarray:
        """
        Calculate dense optical flow using Farneback method in regions of interest.
        
        Args:
            frame1: First frame
            frame2: Second frame
            mask: Optional mask for regions of interest
            
        Returns:
            Dense optical flow field
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            **self.dense_flow_params
        )
        
        # Apply mask if provided
        if mask is not None:
            flow[~mask] = 0
            
        return flow
        
    def create_roi_mask(self, frame_shape: Tuple[int, int], 
                       features: np.ndarray, 
                       radius: int = 50) -> np.ndarray:
        """
        Create a mask for regions of interest around detected features.
        
        Args:
            frame_shape: Shape of the frame (height, width)
            features: Detected feature points
            radius: Radius around each feature for ROI
            
        Returns:
            Boolean mask indicating ROIs
        """
        mask = np.zeros(frame_shape[:2], dtype=bool)
        
        # Create circular ROIs around each feature
        for feature in features:
            x, y = feature.ravel()
            y, x = int(y), int(x)
            y1 = max(0, y - radius)
            y2 = min(frame_shape[0], y + radius)
            x1 = max(0, x - radius)
            x2 = min(frame_shape[1], x + radius)
            mask[y1:y2, x1:x2] = True
            
        return mask
        
    def calculate_hybrid_energy(self, sparse_flow: Tuple[np.ndarray, np.ndarray], 
                              dense_flow: np.ndarray) -> Dict:
        """
        Combine sparse and dense flow information to calculate motion energy.
        
        Args:
            sparse_flow: Tuple of (new points, old points) from sparse flow
            dense_flow: Dense optical flow field
            
        Returns:
            Dictionary containing energy metrics
        """
        # Calculate sparse flow energy (average motion magnitude)
        if len(sparse_flow[0]) > 0:
            flow_vectors = sparse_flow[0] - sparse_flow[1]
            sparse_energy = np.mean(np.sqrt(
                flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2
            ))
        else:
            sparse_energy = 0
            
        # Calculate dense flow energy (average motion magnitude)
        dense_magnitude = np.sqrt(dense_flow[..., 0]**2 + dense_flow[..., 1]**2)
        dense_energy = np.mean(dense_magnitude)
        
        # Combine energies with weights (40% sparse, 60% dense)
        total_energy = 0.4 * sparse_energy + 0.6 * dense_energy
        
        return {
            'total_energy': total_energy,
            'sparse_energy': sparse_energy,
            'dense_energy': dense_energy
        }
        
    def classify_energy_level(self, energy: float) -> int:
        """
        Classify energy value into discrete levels (0-5).
        
        Args:
            energy: Calculated energy value
            
        Returns:
            Energy level classification (0-5)
        """
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for level, threshold in enumerate(thresholds):
            if energy < threshold:
                return level
        return 5
        
    def hybrid_optical_flow(self, frame1: np.ndarray, 
                          frame2: np.ndarray) -> Dict:
        """
        Calculate energy using hybrid optical flow approach.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Dictionary containing energy calculations and metadata
        """
        # Detect features for sparse flow
        features = self.detect_sparse_features(frame1)
        
        if features is not None:
            # Calculate sparse flow
            sparse_flow = self.calculate_sparse_flow(frame1, frame2, features)
            
            # Create ROI mask around features
            roi_mask = self.create_roi_mask(frame1.shape, features)
            
            # Calculate dense flow in ROIs
            dense_flow = self.calculate_dense_flow(frame1, frame2, roi_mask)
            
            # Calculate combined energy
            energy_data = self.calculate_hybrid_energy(sparse_flow, dense_flow)
            
            # Add feature count to results
            energy_data['feature_count'] = len(features)
            
            return energy_data
        else:
            return {
                'total_energy': 0,
                'sparse_energy': 0,
                'dense_energy': 0,
                'feature_count': 0
            }
            
    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process entire video and return frame-by-frame energy analysis.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of dictionaries containing energy data for each frame
        """
        results = []
        cap = cv2.VideoCapture(video_path)
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return results
            
        frame_count = 0
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
                
            # Calculate energy between consecutive frames
            energy_data = self.hybrid_optical_flow(prev_frame, curr_frame)
            
            # Add frame information
            energy_data.update({
                'frame_number': frame_count,
                'energy_level': self.classify_energy_level(
                    energy_data['total_energy']
                )
            })
            
            results.append(energy_data)
            prev_frame = curr_frame
            frame_count += 1
            
        cap.release()
        return results

# Example usage and testing
if __name__ == "__main__":
    print("\nStarting video processing...")
    estimator = HybridOpticalFlowEstimator()
    
    # Define test video paths
    video_paths = {
        'crowd': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/crowd.mp4",
        'ocean': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/ocean.mp4",
        'pingPong': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/pingPong.mp4",
        'market': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/market.mp4",
        'surf': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/surf.mp4"
    }
    
    # Process each test video
    for video_name, video_path in video_paths.items():
        print(f"\nProcessing {video_name} video...")
        
        # Verify file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            continue
        
        # Process video and collect results
        results = estimator.process_video(video_path)
        print(f"\nProcessing complete for {video_name}.")
        print(f"Total frames processed: {len(results)}")
        
        # Initialize statistics containers
        level_stats = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        
        # Collect statistics per energy level
        for frame_data in results:
            level = frame_data['energy_level']
            level_stats[level].append({
                'energy': frame_data['total_energy'],
                'sparse_energy': frame_data['sparse_energy'],
                'dense_energy': frame_data['dense_energy'],
                'feature_count': frame_data['feature_count']
            })
        
        # Print detailed statistics
        print(f"\nEnergy Level Statistics for {video_name}:")
        print("Level | Frames (%) | Avg Energy | Feature Count | Individual Energies [Sparse, Dense]")
        print("-" * 100)
        
        # Calculate and display statistics for each level
        for level in range(6):
            frames = level_stats[level]
            if frames:
                num_frames = len(frames)
                percentage = (num_frames / len(results)) * 100
                avg_energy = sum(f['energy'] for f in frames) / num_frames
                avg_features = sum(f['feature_count'] for f in frames) / num_frames
                avg_sparse = sum(f['sparse_energy'] for f in frames) / num_frames
                avg_dense = sum(f['dense_energy'] for f in frames) / num_frames
                
                print(f"  {level}   | {num_frames:3d} ({percentage:5.1f}%) | {avg_energy:9.3f} | {avg_features:12.1f} | [{avg_sparse:.3f}, {avg_dense:.3f}]")
        
        print("\n" + "="*100)  # Visual separator between videos
