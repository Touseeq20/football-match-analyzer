# 🦅 Comprehensive Football Match Analyzer

The ultimate football analytics tool that integrates goal detection, foul detection, player identification, and jersey tracking into a single unified system.

## 🎯 Features

- **All-in-One Analytics**: Monitors goals, fouls, and player movements simultaneously.
- **Intelligent Goal Detection**: Real-time ball trajectory tracking + goal net segmentation.
- **Automated Foul Logging**: Detects and highlights player contact events.
- **Identity Fusion**: Combines YOLOv8 player detection with Tesseract OCR for jersey recognition.
- **Smart Event Triggering**: Automatically extracts and saves highlights for key match events.
- **Integrated Results**: A unified visualization overlay showing all match statistics.

## 📋 Requirements

- Python 3.9+
- OpenCV
- Ultralytics YOLOv8
- Tesseract OCR engine
- PyTesseract
- NumPy & SciPy
- PyTorch

## 🚀 Installation

1. Install Tesseract OCR on your system (required for jersey recognition).
2. Clone this repository:
```bash
git clone https://github.com/Touseeq20/football-match-analyzer.git
cd football-match-analyzer
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Download necessary models:
   - `goal.pt`: Goal net segmentation
   - `yolofoul.pt`: Foul detection model
   - `yolov8n.pt`: General person/ball detector
   - `best(0-90).pt`: Jersey number localizer

## 💻 Usage

### Run Analysis
```bash
python match_analyzer.py
```

### Configuration
Edit the model paths and video source in `match_analyzer.py`:
```python
video_path = r'match_video.mp4'
goal_model_path = 'goal.pt'
foul_model_path = 'yolofoul.pt'
```

## 🔧 How It Works

This system runs multiple specialized pipelines in parallel on each frame:

1. **Goal Pipeline**: Tracks the ball and checks intersection with the goal net mask.
2. **Foul Pipeline**: Processes a sharpened version of the frame to find `Foul` or `Guilty` classes.
3. **Player Pipeline**: Detects players and uses proximity-based tracking for IDs.
4. **Jersey Pipeline**: Localizes numbers on player jerseys and runs OCR.

All detections are fused together. For example, a "Goal" event is saved to the `outputdetection/goal` folder with the context of which players were involved.

## 📊 Technical Architecture

- **Object Detection**: YOLOv8 (Multiple specialized models)
- **Segmentation**: YOLOv8-seg (Goal Net)
- **OCR**: Tesseract with multi-preprocessing
- **Tracking**: Euclidean Distance + IoU Fusion
- **Event Management**: Buffer-based clip generation

## 📦 Dependencies

- opencv-python
- ultralytics
- pytesseract
- numpy
- scipy
- torch

## 👤 Author

**Touseeq Ahmed**
- GitHub: [@Touseeq20](https://github.com/Touseeq20)
