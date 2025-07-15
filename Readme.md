# ⚽ Soccer Player Re-Identification Using YOLOv8 + Deep SORT + ResNet50

This project performs player detection, re-identification, and multi-object tracking using two videos captured from different camera perspectives (Tacticam and Broadcast). The combination of YOLOv8, ResNet50 embeddings, and Deep SORT tracking ensures reliable and continuous player tracking.
---

## 📁 Project Structure
📂 Soccer Player Re-Identification Assignment
├── best.pt # Trained YOLOv8 model weights
├── tacticam.mp4 # Tacticam input video
├── broadcast.mp4 # Broadcast input video
├── main.py # Main Python script
├── Tacticam_IDs.mp4 # Output video with IDs for tacticam
├── Broadcast_IDs.mp4 # Output video with IDs for broadcast
├── README.md # Documentation (this file)
└── report.pdf / report.md # Brief report with methodology and analysis
---

## 🔧 How to Set Up and Run the Code

### 1. Clone or prepare the directory

Move all the files into a single working directory.

### 2. Install dependencies

Ensure you're using Python 3.8 or above. Install required libraries:

```bash
pip install ultralytics deep-sort-realtime opencv-python torch torchvision
``` 
### 3 Make sure the YOLOv8 model file exists
Place best.pt (your trained weights) in the same directory as main.py

### 4 Run the Script 
python main.py

This will generate:
tacticam_IDs.mp4 
broadcast_IDs.mp4

with bounding boxes and player tracking IDs overlaid.

### 5 📦 Dependencies
Python >= 3.8

Ultralytics YOLO

deep-sort-realtime

OpenCV

Torch & Torchvision

