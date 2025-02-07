#!/usr/bin/env python3
"""
Optimized Block Motion Analysis
------------------------------
This module implements an adaptive block-based motion estimation algorithm
for video energy level detection. It uses variable block sizes based on
frame content complexity and edge density.
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

class OptimizedBlockMotionEstimator:
    """
    A class that implements adaptive block-based motion estimation.
    Uses variable block sizes based on frame content complexity.
    Optimizes processing by adapting to local image characteristics.
    """
    
    def __init__(self):
        """
        Initialize parameters for block motion estimation.
        - min_block_size: Minimum block size for detailed areas
        - max_block_size: Maximum block size for uniform areas
        - search_range: Range for motion search
        - edge_thresholds: Thresholds for edge density classification
        """
        print("Initializing OptimizedBlockMotionEstimator...")
        self.min_block_size = 8      # Minimum block size for high-detail regions
        self.max_block_size = 32     # Maximum block size for uniform regions
        self.search_range = 16       # Search range for motion estimation
        self.edge_threshold = 100    # Edge detection threshold
        self.edge_thresholds = {
            'high': 0.1,    # High edge density threshold
            'medium': 0.05  # Medium edge density threshold
        }
        
    def determine_optimal_block_size(self, frame: np.ndarray, 
                                   min_block: int, 
                                   max_block: int) -> int:
        """
        Determine optimal block size based on frame content complexity.
        Uses edge density to adapt block size: smaller blocks for complex regions,
        larger blocks for uniform areas.
        
        Args:
            frame: Input frame in BGR format
            min_block: Minimum allowed block size
            max_block: Maximum allowed block size
            
        Returns:
            Optimal block size for the frame
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)  # Calculate edge density
        
        # Rule-based block size selection based on edge density
        if edge_density > self.edge_thresholds['high']:
            return min_block  # Small blocks for high detail
        elif edge_density > self.edge_thresholds['medium']:
            return (min_block + max_block) // 2  # Medium blocks
        else:
            return max_block  # Large blocks for uniform areas
            
    def calculate_block_difference(self, block1: np.ndarray, 
                                 block2: np.ndarray) -> float:
        """
        Calculate the mean absolute difference between two blocks.
        
        Args:
            block1: First block
            block2: Second block
            
        Returns:
            Mean difference between blocks
        """
        # Convert blocks to float for accurate difference calculation
        b1 = block1.astype(float)
        b2 = block2.astype(float)
        
        # Calculate absolute difference
        diff = np.abs(b1 - b2)
        
        # Calculate mean difference
        mean_diff = np.mean(diff)
        
        return mean_diff
        
    def normalize_energy(self, energy: float, 
                        height: int, 
                        width: int, 
                        block_size: int) -> float:
        """
        Normalize energy value based on frame and block dimensions.
        
        Args:
            energy: Raw energy value
            height: Frame height
            width: Frame width
            block_size: Current block size
            
        Returns:
            Normalized energy value
        """
        # Calculate total number of blocks
        total_blocks = (height // block_size) * (width // block_size)
        return energy / (total_blocks * 255.0)  # Normalize by max possible value
        
    def optimized_block_motion(self, frame1: np.ndarray, 
                             frame2: np.ndarray) -> Dict:
        """
        Calculate motion energy using optimized block motion analysis.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Dictionary containing energy calculations and block information
        """
        height, width = frame1.shape[:2]
        
        # Get optimal block size based on frame content
        block_size = self.determine_optimal_block_size(
            frame1, 
            self.min_block_size, 
            self.max_block_size
        )
        
        total_energy = 0
        block_energies = []
        
        # Process each block in the frame
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                # Extract corresponding blocks from both frames
                block1 = frame1[y:y+block_size, x:x+block_size]
                block2 = frame2[y:y+block_size, x:x+block_size]
                
                # Calculate block energy
                block_energy = self.calculate_block_difference(block1, block2)
                total_energy += block_energy
                
                # Store block energy and position
                block_energies.append({
                    'position': (x, y),
                    'energy': block_energy
                })
        
        # Normalize total energy
        normalized_energy = self.normalize_energy(
            total_energy, height, width, block_size
        )
        
        return {
            'total_energy': normalized_energy,
            'block_size': block_size,
            'block_energies': block_energies
        }
        
    def classify_energy_level(self, energy: float) -> int:
        """
        Classify energy value into discrete levels (0-5).
        
        Args:
            energy: Normalized energy value
            
        Returns:
            Energy level classification (0-5)
        """
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for level, threshold in enumerate(thresholds):
            if energy < threshold:
                return level
        return 5
        
    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process entire video and return frame-by-frame analysis.
        
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
            energy_data = self.optimized_block_motion(prev_frame, curr_frame)
            
            # Add frame information and energy level
            energy_data.update({
                'frame_number': frame_count,
                'energy_level': self.classify_energy_level(
                    energy_data['total_energy']
                ),
                'block_energy': energy_data['total_energy'],
                'edge_energy': 0.0,  # Placeholder for edge energy
                'motion_energy': 0.0  # Placeholder for motion energy
            })
            
            results.append(energy_data)
            prev_frame = curr_frame
            frame_count += 1
            
        cap.release()
        return results
        
    def visualize_block_energies(self, frame: np.ndarray, 
                               block_energies: List[Dict]) -> np.ndarray:
        """
        Visualize block energies on frame using color-coded rectangles.
        
        Args:
            frame: Input frame
            block_energies: List of block energy data
            
        Returns:
            Frame with visualized block energies
        """
        vis_frame = frame.copy()
        
        for block in block_energies:
            x, y = block['position']
            energy = block['energy']
            
            # Color based on energy (blue to red)
            color = (
                int(255 * energy),  # B
                0,                  # G
                int(255 * (1 - energy))  # R
            )
            
            cv2.rectangle(
                vis_frame,
                (x, y),
                (x + self.block_size, y + self.block_size),
                color,
                2
            )
            
        return vis_frame

# Example usage and testing
if __name__ == "__main__":
    print("Main block starting...")
    estimator = OptimizedBlockMotionEstimator()
    
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
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_name}")
            continue
            
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in {video_name}: {total_frames}")
        cap.release()
        
        # Process video and collect results
        results = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        ret, prev_frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame of {video_name}")
            continue
            
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
                
            # Calculate energy
            energy_data = estimator.optimized_block_motion(prev_frame, curr_frame)
            
            # Add frame information
            energy_data.update({
                'frame_number': frame_count,
                'energy_level': estimator.classify_energy_level(
                    energy_data['total_energy']
                ),
                'block_energy': energy_data['total_energy'],
                'edge_energy': 0.0,
                'motion_energy': 0.0
            })
            
            results.append(energy_data)
            prev_frame = curr_frame
            frame_count += 1
            
        cap.release()
        print(f"\nProcessing complete for {video_name}. Found {len(results)} frames.")
        
        # Initialize statistics containers
        level_stats = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        
        # Collect statistics per energy level
        for frame_data in results:
            level = frame_data['energy_level']
            level_stats[level].append({
                'energy': frame_data['total_energy'],
                'block_energy': frame_data['block_energy'],
                'edge_energy': frame_data['edge_energy'],
                'motion_energy': frame_data['motion_energy'],
                'block_size': frame_data['block_size']
            })
        
        # Print detailed statistics
        print(f"\nEnergy Level Statistics for {video_name}:")
        print("Level | Frames (%) | Avg Energy | Avg Block Size | Individual Energies [Block, Edge, Motion]")
        print("-" * 100)
        
        for level in range(6):
            frames = level_stats[level]
            if frames:
                num_frames = len(frames)
                percentage = (num_frames / len(results)) * 100
                avg_energy = sum(f['energy'] for f in frames) / num_frames
                avg_block_size = sum(f['block_size'] for f in frames) / num_frames
                
                avg_block = sum(f['block_energy'] for f in frames) / num_frames
                avg_edge = sum(f['edge_energy'] for f in frames) / num_frames
                avg_motion = sum(f['motion_energy'] for f in frames) / num_frames
                
                print(f"  {level}   | {num_frames:3d} ({percentage:5.1f}%) | {avg_energy:9.3f} | {avg_block_size:13.1f} | [{avg_block:.3f}, {avg_edge:.3f}, {avg_motion:.3f}]")
        
        print("\n" + "="*100)  # Visual separator between videos
