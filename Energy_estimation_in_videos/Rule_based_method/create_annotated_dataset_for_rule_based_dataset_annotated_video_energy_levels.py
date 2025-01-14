import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from rule_based_dataset_annotated_video_energy_levels import SimpleEnergyEstimator, process_video
from datetime import datetime

class EnergyDatasetCreator:
    def __init__(self, base_path: str):
        """Initialize dataset creator"""
        self.base_path = Path(base_path)
        self.setup_directories()
        
    def setup_directories(self):
        """Create dataset directory structure"""
        directories = [
            'raw_videos',          # Original videos
            'processed_videos',     # Videos with visualizations
            'annotations',         # JSON annotations
            'metadata',           # Dataset information
            'statistics'          # Analysis results
        ]
        
        for dir_path in directories:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)
            
    def process_dataset(self, input_videos_path: str):
        """Process all videos and create dataset"""
        # Get all video files
        video_files = list(Path(input_videos_path).glob('*.mp4'))
        dataset_info = {
            'creation_date': datetime.now().isoformat(),
            'total_videos': len(video_files),
            'videos': []
        }
        
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            
            try:
                # Copy original video to raw_videos
                raw_path = self.base_path / 'raw_videos' / video_file.name
                shutil.copy2(video_file, raw_path)
                
                # Process video and generate annotations
                output_path = self.base_path / 'processed_videos' / f"{video_file.stem}_annotated.mp4"
                annotations = process_video(
                    str(video_file), 
                    str(output_path), 
                    save_annotations=False
                )
                
                # Save annotations
                annotation_path = self.base_path / 'annotations' / f"{video_file.stem}.json"
                self.save_annotations(annotation_path, annotations, str(video_file))
                
                # Add to dataset info
                video_info = {
                    'file_name': video_file.name,
                    'raw_path': str(raw_path),
                    'processed_path': str(output_path),
                    'annotation_path': str(annotation_path),
                    'duration': len(annotations),
                    'energy_summary': self.calculate_summary(annotations)
                }
                dataset_info['videos'].append(video_info)
                
            except Exception as e:
                print(f"Error processing {video_file.name}: {str(e)}")
        
        # Save dataset metadata
        self.save_dataset_info(dataset_info)
        
        return dataset_info
    
    def save_annotations(self, path: Path, annotations: list, video_path: str):
        """Save video annotations with metadata"""
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        metadata = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'resolution': (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        }
        cap.release()
        
        # Prepare annotation data
        data = {
            'video_metadata': metadata,
            'frame_annotations': annotations,
            'summary': self.calculate_summary(annotations)
        }
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def calculate_summary(self, annotations: list) -> dict:
        """Calculate summary statistics for video"""
        energy_values = [a['energy_value'] for a in annotations]
        energy_levels = [a['energy_level'] for a in annotations]
        
        return {
            'mean_energy': float(np.mean(energy_values)),
            'max_energy': float(np.max(energy_values)),
            'min_energy': float(np.min(energy_values)),
            'std_energy': float(np.std(energy_values)),
            'level_distribution': {
                str(level): energy_levels.count(level) / len(energy_levels)
                for level in range(6)
            }
        }
    
    def save_dataset_info(self, info: dict):
        """Save dataset metadata"""
        metadata_path = self.base_path / 'metadata' / 'dataset_info.json'
        with open(metadata_path, 'w') as f:
            json.dump(info, f, indent=4)

def main():
    # Initialize dataset creator
    dataset_path = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/energy_estimation_dataset/"
    creator = EnergyDatasetCreator(dataset_path)
    
    # Process videos
    input_videos = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/energyDetection/dataset_annotated_for_energy_level/rule_based_dataset_annotated/short_videos/"
    dataset_info = creator.process_dataset(input_videos)
    
    # Print summary
    print("\nDataset Creation Complete!")
    print(f"Total Videos Processed: {dataset_info['total_videos']}")
    print(f"Dataset Location: {dataset_path}")
    print("\nDirectory Structure:")
    print(f"- Raw Videos: {dataset_path}/raw_videos/")
    print(f"- Processed Videos: {dataset_path}/processed_videos/")
    print(f"- Annotations: {dataset_path}/annotations/")
    print(f"- Metadata: {dataset_path}/metadata/")
    print(f"- Statistics: {dataset_path}/statistics/")

if __name__ == "__main__":
    main()
