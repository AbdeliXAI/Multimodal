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
    def __init__(self):
        print("Initializing HybridOpticalFlowEstimator...")
        # Parameters for feature detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Parameters for optical flow
        self.flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Dense flow parameters
        self.dense_flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
    def detect_sparse_features(self, frame: np.ndarray) -> np.ndarray:
        """Detect good features to track"""
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
        """Calculate sparse optical flow"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate flow
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            features,
            None,
            **self.flow_params
        )
        
        # Select good points
        good_new = new_features[status == 1]
        good_old = features[status == 1]
        
        return good_new, good_old
        
    def calculate_dense_flow(self, frame1: np.ndarray, 
                           frame2: np.ndarray, 
                           mask: np.ndarray = None) -> np.ndarray:
        """Calculate dense optical flow in regions of interest"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            **self.dense_flow_params
        )
        
        if mask is not None:
            flow[~mask] = 0
            
        return flow
        
    def create_roi_mask(self, frame_shape: Tuple[int, int], 
                       features: np.ndarray, 
                       radius: int = 50) -> np.ndarray:
        """Create mask for regions of interest around features"""
        mask = np.zeros(frame_shape[:2], dtype=bool)
        
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
        """Combine sparse and dense flow information"""
        # Calculate sparse flow energy
        if len(sparse_flow[0]) > 0:
            flow_vectors = sparse_flow[0] - sparse_flow[1]
            sparse_energy = np.mean(np.sqrt(
                flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2
            ))
        else:
            sparse_energy = 0
            
        # Calculate dense flow energy
        dense_magnitude = np.sqrt(dense_flow[..., 0]**2 + dense_flow[..., 1]**2)
        dense_energy = np.mean(dense_magnitude)
        
        # Combine energies (weighted average)
        total_energy = 0.4 * sparse_energy + 0.6 * dense_energy
        
        return {
            'total_energy': total_energy,
            'sparse_energy': sparse_energy,
            'dense_energy': dense_energy
        }
        
    def classify_energy_level(self, energy: float) -> int:
        """Classify energy into levels 0-5"""
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for level, threshold in enumerate(thresholds):
            if energy < threshold:
                return level
        return 5
        
    def hybrid_optical_flow(self, frame1: np.ndarray, 
                          frame2: np.ndarray) -> Dict:
        """Calculate energy using hybrid optical flow"""
        # Detect features
        features = self.detect_sparse_features(frame1)
        
        if features is not None:
            # Calculate sparse flow
            sparse_flow = self.calculate_sparse_flow(frame1, frame2, features)
            
            # Create ROI mask
            roi_mask = self.create_roi_mask(frame1.shape, features)
            
            # Calculate dense flow in ROIs
            dense_flow = self.calculate_dense_flow(frame1, frame2, roi_mask)
            
            # Calculate combined energy
            energy_data = self.calculate_hybrid_energy(sparse_flow, dense_flow)
            
            # Add feature information
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
        """Process entire video and return frame-by-frame analysis"""
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
                
            # Calculate energy
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

# Usage example:
if __name__ == "__main__":
    print("\nStarting video processing...")
    estimator = HybridOpticalFlowEstimator()
    
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
        
        # Process video
        results = estimator.process_video(video_path)
        print(f"\nProcessing complete for {video_name}.")
        print(f"Total frames processed: {len(results)}")
        
        # Initialize statistics per level
        level_stats = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        
        # Collect data for each level
        for frame_data in results:
            level = frame_data['energy_level']
            level_stats[level].append({
                'energy': frame_data['total_energy'],
                'sparse_energy': frame_data['sparse_energy'],
                'dense_energy': frame_data['dense_energy'],
                'feature_count': frame_data['feature_count']
            })
        
        # Print statistics for current video
        print(f"\nEnergy Level Statistics for {video_name}:")
        print("Level | Frames (%) | Avg Energy | Feature Count | Individual Energies [Sparse, Dense]")
        print("-" * 100)
        
        # Calculate and print statistics for each level
        for level in range(6):
            frames = level_stats[level]
            if frames:
                num_frames = len(frames)
                percentage = (num_frames / len(results)) * 100
                avg_energy = sum(f['energy'] for f in frames) / num_frames
                avg_features = sum(f['feature_count'] for f in frames) / num_frames
                
                # Calculate average individual energies
                avg_sparse = sum(f['sparse_energy'] for f in frames) / num_frames
                avg_dense = sum(f['dense_energy'] for f in frames) / num_frames
                
                print(f"  {level}   | {num_frames:3d} ({percentage:5.1f}%) | {avg_energy:9.3f} | {avg_features:12.1f} | [{avg_sparse:.3f}, {avg_dense:.3f}]")
        
        print("\n" + "="*100)  # Separator between videos
