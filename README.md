# Football Player Tracking System

## Overview
This system tracks football players and the ball across tactical and broadcast camera views using:
- YOLOv8 for object detection
- ByteTrack for object tracking
- ResNet50 for player re-identification

## Requirements
- Python 3.8+
- NVIDIA GPU with CUDA 11.x (recommended)

## Installation
```bash
conda create -n football_tracking python=3.8
conda activate football_tracking
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install ultralytics opencv-python scipy numpy lap
```
## Data Preparation
Place these files in project root:
- `tacticam.mp4` - Tactical camera video
- `broadcast.mp4` - Broadcast camera video
- `best.pt` - Custom-trained YOLO model

## Usage

1. Process tactical camera video:
   ```
   python track_tacticam.py
   ```
   Output: `tacticam_tracked_bytetrack.mp4`

2. Process Broadcast camera video:
   ```
   python track_broadcast.py
   ```
   Output: `broadcast_tracked.mp4`


## Output Videos
* Players marked with colored feet ellipses and tracking IDs
* Green ellipse = player near ball
* Ball marked with green triangle
* Consistent player IDs across views

