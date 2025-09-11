# People Counter

A **People Counting Application** using **YOLOv5 (ONNX)** and **OpenCV**.  
This tool can detect, track, and count people entering and exiting an area from video files, webcams, or IP cameras.

---

## ✨ Features
- Supports:
  - **Video files** (mp4, avi, mkv, etc.)
  - **Local webcam**
  - **IP cameras (RTSP/HTTP)**
- Counts **entries and exits** based on selected direction:
  - Top → Bottom
  - Bottom → Top
  - Left → Right
  - Right → Left
- Saves reports to **Excel** or **CSV**.
- **Headless video analysis** (no preview, faster).
- **Settings panel** to adjust:
  - Crowd level  
  - Accuracy level  
  - Video playback speed (0.5x – 2x)  
  - Model confidence & NMS thresholds  

---

## 🛠 Requirements
- Python 3.9+  
- Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage
Run the main script to start the UI:
```bash
python main.py
```

---

## 📂 Project Structure
```
people_counter/
│
├── core/          # Core pipeline, detection, tracking
├── tracker/       # Centroid tracker + Trackable object
├── ui/            # User interface (Tkinter)
├── utils/         # Config + helpers
├── models/        # ONNX model (e.g. yolov5s.onnx)
├── videos/        # Sample videos
├── requirements.txt
└── main.py
```

---

## 📊 Example Output
- **Live preview mode**: shows video feed with bounding boxes and counts.  
- **Headless mode**: processes video and directly exports results.  

---

## ⚡ Notes
- Default model: `models/yolov5s.onnx` (COCO pretrained, person class only).  
- For better accuracy: set **Accuracy Level → 4 (Very Accurate)**.  
- GPU acceleration (CUDA) is supported if OpenCV was built with CUDA and your GPU is compatible.  

---

## 📜 License
MIT License. Free to use and modify.
