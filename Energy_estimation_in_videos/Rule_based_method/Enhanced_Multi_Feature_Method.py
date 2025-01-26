
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
    def __init__(self):
        print("Initializing EnhancedEnergyEstimator...")
        self.min_block_size = 8
        self.max_block_size = 32
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
    def calculate_pixel_difference(self, frame1: np.ndarray, 
                                 frame2: np.ndarray) -> float:
        """Calculate pixel-based difference energy"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Normalize and return mean energy
        return np.mean(diff) / 255.0

    def calculate_optical_flow(self, frame1: np.ndarray, 
                             frame2: np.ndarray) -> float:
        """Calculate optical flow based energy"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
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
        
        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude)

    def calculate_block_motion(self, frame1: np.ndarray, 
                             frame2: np.ndarray) -> float:
        """Calculate block-based motion energy"""
        height, width = frame1.shape[:2]
        block_size = self.determine_optimal_block_size(frame1)
        energy = 0.0
        
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                # Extract blocks
                block1 = frame1[y:y+block_size, x:x+block_size]
                block2 = frame2[y:y+block_size, x:x+block_size]
                
                # Calculate block difference
                diff = np.mean(np.abs(block1.astype(float) - block2.astype(float)))
                energy += diff
                
        # Normalize
        total_blocks = ((height // block_size) * (width // block_size))
        return energy / (total_blocks * 255.0)

    def determine_optimal_block_size(self, frame: np.ndarray) -> int:
        """Determine optimal block size based on frame content"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        # Adjust block size based on edge density
        if edge_density > 0.1:
            return self.min_block_size
        elif edge_density > 0.05:
            return (self.min_block_size + self.max_block_size) // 2
        else:
            return self.max_block_size

    def calculate_adaptive_weights(self, pixel_energy: float, 
                                 flow_energy: float, 
                                 block_energy: float) -> List[float]:
        """Calculate adaptive weights based on energy values"""
        total_energy = pixel_energy + flow_energy + block_energy
        if total_energy == 0:
            return [0.33, 0.33, 0.34]
            
        # Adjust weights based on relative contributions
        weights = [
            pixel_energy / total_energy,
            flow_energy / total_energy,
            block_energy / total_energy
        ]
        
        # Normalize weights
        sum_weights = sum(weights)
        return [w / sum_weights for w in weights]

    def enhanced_multi_feature_energy(self, frame1: np.ndarray, 
                                    frame2: np.ndarray) -> Dict:
        """Calculate energy using multiple features with adaptive weighting"""
        # Calculate individual energies
        pixel_energy = self.calculate_pixel_difference(frame1, frame2)
        flow_energy = self.calculate_optical_flow(frame1, frame2)
        block_energy = self.calculate_block_motion(frame1, frame2)
        
        # Calculate adaptive weights
        weights = self.calculate_adaptive_weights(
            pixel_energy, flow_energy, block_energy
        )
        
        # Combined energy
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

    def classify_energy_level(self, energy: float) -> int:
        """Classify energy into levels 0-5"""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for level, threshold in enumerate(thresholds):
            if energy < threshold:
                return level
        return 5

if __name__ == "__main__":
    print("Main block starting...")
    print("Starting video processing...")
    estimator = EnhancedEnergyEstimator()
    
    # Define all video paths
    video_paths = {
        'crowd': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/crowd.mp4",
        'ocean': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/ocean.mp4",
        'pingPong': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/pingPong.mp4",
        'market': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/market.mp4",
        'surf': "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/surf.mp4"
    }
    
    # Process each video
    for video_name, video_path in video_paths.items():
        print(f"\nProcessing {video_name} video...")
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_name}")
            continue
            
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in {video_name}: {total_frames}")
        cap.release()
        
        print(f"Processing {video_name}...")
        frame_count = 0
        
        # Process with progress updates
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
            
            # Add frame information
            energy_data.update({
                'frame_number': frame_count,
                'energy_level': estimator.classify_energy_level(energy_data['total_energy'])
            })
            
            results.append(energy_data)
            prev_frame = curr_frame
            frame_count += 1
            
            # Show progress every 10 frames
            #if frame_count % 10 == 0:
            #    print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        print(f"\nProcessing complete for {video_name}. Found {len(results)} frames.")
        
        # Initialize statistics per level
        level_stats = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        
        # Collect data for each level
        for frame_data in results:
            level = frame_data['energy_level']
            level_stats[level].append({
                'energy': frame_data['total_energy'],
                'weights': frame_data['weights'],
                'pixel_energy': frame_data['pixel_energy'],
                'flow_energy': frame_data['flow_energy'],
                'block_energy': frame_data['block_energy']
            })
        
        # Print statistics for current video
        print(f"\nEnergy Level Statistics for {video_name}:")
        print("Level | Frames (%) | Avg Energy | Avg Weights [Pixel, Flow, Block] | Individual Energies [Pixel, Flow, Block]")
        print("-" * 100)
        
        # Calculate and print statistics for each level
        for level in range(6):
            frames = level_stats[level]
            if frames:
                num_frames = len(frames)
                percentage = (num_frames / len(results)) * 100
                avg_energy = sum(f['energy'] for f in frames) / num_frames
                
                # Calculate average individual energies
                avg_pixel = sum(f['pixel_energy'] for f in frames) / num_frames
                avg_flow = sum(f['flow_energy'] for f in frames) / num_frames
                avg_block = sum(f['block_energy'] for f in frames) / num_frames
                
                # Calculate average weights
                avg_weights = [
                    sum(f['weights'][i] for f in frames) / num_frames 
                    for i in range(3)
                ]
                
                print(f"  {level}   | {num_frames:3d} ({percentage:5.1f}%) | {avg_energy:9.3f} | [{', '.join(f'{w:.3f}' for w in avg_weights)}] | [{avg_pixel:.3f}, {avg_flow:.3f}, {avg_block:.3f}]")
        
        print("\n" + "="*100)  # Separator between videos 
