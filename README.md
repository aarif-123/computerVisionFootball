# 🧠 Football Player Re-Identification in a Single Feed

This project performs **real-time player re-identification** in a 15-second football video. Each player is uniquely detected and tracked, even when they leave and re-enter the frame.

---

## 📌 Project Overview

- Detect players using **YOLOv11**
- Extract visual features using **ResNet18**
- Track players using **cosine similarity-based Re-ID**
- Assign and maintain consistent player IDs

---

## 🛠️ Requirements

### 🔧 Install Dependencies

First, create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

Then install the required packages:

```bash
pip install torch torchvision ultralytics opencv-python scikit-learn
```

---

## 📂 Folder Structure

```
computer_vision_football_players/
├── yolov11.pt                   # Pretrained YOLOv11 weights (required)
├── 15sec_input_720p.mp4         # Input video
├── reid_tracker.py              # Main script for detection + re-identification
├── player_utils.py              # Helper module: feature extraction & ID logic
├── output.avi                   # Output video with tracked player IDs
├── output_log.json              # (Optional) JSON log of player IDs per frame
├── ReID_Project_Report.pdf      # Final report
└── README.md                    # This file
```

---

## ▶️ How to Run the Code

### 1. Place your files

Ensure the following are in the project directory:
- `yolov11.pt` (YOLOv11 weights)
- `15sec_input_720p.mp4` (input video)

### 2. Run the main tracking script

```bash
python reid_tracker.py
```

### 3. Output files

- ✅ `output.avi` — Video with bounding boxes and player IDs
- ✅ `output_log.json` — Optional log of player identities and bounding boxes per frame

---

## 📈 What’s Inside?

- YOLOv11 for detection
- ResNet18 (Torchvision) for feature extraction
- Cosine similarity for identity matching
- Modular code (`player_utils.py`) for clarity and reuse

---

## Future Enhancement Can be made 

- Reduce video resolution for faster processing
- Upgrade ReID to OSNet or add Deep SORT for long videos
- Add jersey number OCR for visual redundancy

---

## 👨‍💻 Author

Mohammed Aarif  
Lovely Professional University  
B.Tech in AI & ML  
