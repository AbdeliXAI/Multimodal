import cv2
import numpy as np
import os
from scipy.signal import find_peaks


def detect_shot_boundaries(video_path, shot_threshold=30):
    cap = cv2.VideoCapture(video_path)
    frame_diffs = []
    frames = []
    shot_boundaries = []

    ret, prev_frame = cap.read()
    prev_hist = cv2.calcHist([prev_frame], [0], None, [256], [0,256])
    frames.append(prev_frame)

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_hist = cv2.calcHist([curr_frame], [0], None, [256], [0,256])
        shot_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)
        frame_diffs.append(shot_diff)
        frames.append(curr_frame)
        
        if shot_diff > shot_threshold:
            shot_boundaries.append(len(frames)-1)
            
        prev_hist = curr_hist

    cap.release()
    return shot_boundaries, frames

def detect_scene_boundaries(video_path, scene_threshold=0.8):
    cap = cv2.VideoCapture(video_path)
    scene_diffs = []
    frames = []
    
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(prev_frame)

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        scene_diff = np.mean(np.abs(curr_gray - prev_gray))
        scene_diffs.append(scene_diff)
        frames.append(curr_frame)
        prev_gray = curr_gray

    scene_peaks, _ = find_peaks(scene_diffs, height=np.mean(scene_diffs)*scene_threshold)
    
    cap.release()
    return list(scene_peaks), frames

def extract_sequences(boundaries, frames):
    sequences = []
    prev_boundary = 0
    
    for boundary in boundaries:
        sequence = frames[prev_boundary:boundary]
        sequences.append(sequence)
        prev_boundary = boundary
        
    return sequences

def save_sequences(sequences, output_dir, prefix="sequence", fps=30):
    """Save sequences as individual video files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, sequence in enumerate(sequences):
        if not sequence:  # Skip empty sequences
            continue
            
        height, width = sequence[0].shape[:2]
        output_path = f"{output_dir}/{prefix}_{i}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in sequence:
            out.write(frame)
            
        out.release()
        print(f"Saved {output_path}")

# Usage
#video_path = "/Users/mynewowner/Desktop/Resume2022/Mulimodality_vision_video/video_ed_for_multimodal.mp4"
#output_dir = "/Users/mynewowner/Desktop/Resume2022/Mulimodality_vision_video/output_dir_shots"
# Use the exact path where we know the file exists
video_path = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/video_ed_for_multimodal.mp4"
output_dir = "/home/abdeli/yobi_gitLab/batch-call-transcription/ai_external_services/multimodal/output_dir_shots"

# Add debug information before processing
print(f"Video path: {video_path}")
print(f"File exists: {os.path.exists(video_path)}")
print(f"File size: {os.path.getsize(video_path)} bytes")

# Test video reading
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video file")
else:
    print("Successfully opened video file")
    print(f"Video properties:")
    print(f"Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
cap.release()


# Detect and save shots
shot_boundaries, frames = detect_shot_boundaries(video_path)
shots = extract_sequences(shot_boundaries, frames)
save_sequences(shots, f"{output_dir}/shots", "shot")

# Detect and save scenes
scene_boundaries, frames = detect_scene_boundaries(video_path)
scenes = extract_sequences(scene_boundaries, frames)
save_sequences(scenes, f"{output_dir}/scenes", "scene")
