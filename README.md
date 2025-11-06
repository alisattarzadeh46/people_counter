# People Counter

A **People Counting Application** using **YOLOv5 (ONNX)** and **OpenCV**.  
Detects, tracks, and counts people entering/exiting an area from video files, webcams, or IP cameras.

---

## ✨ Features
- Input sources: **Video files**, **Local webcam**, **IP cameras (RTSP/HTTP)**
- Direction-aware counting (Top↔Bottom, Left↔Right)
- **Headless Analyze** mode with auto **Save as Excel/CSV**
- Live preview HUD (enter / exit / inside)
- Robust “armed-after-start” logic → avoids false counts when an object appears on the midline at t=0
- Simple **Settings** panel (sliders + advanced fields)
- CSV/Excel export to `data/exports/`

---

## 🖼 Example Screenshots

### Live Detection & Counting
![Example1](images/example_detection1.jpg)
![Example2](images/example_detection2.jpg)

### Main UI
![Main UI](images/example_ui.jpg)

### Settings Panel
![Settings](images/example_settings.jpg)

> Place these screenshots under a new folder called `images/` in your repository.

---

## 🛠 Requirements
- Python **3.9+**
- Install dependencies:
```bash
pip install -r requirements.txt
```

Model file:
- Default model included: `models/yolov5s.onnx` (COCO pretrained, person class only).

---

## 🚀 Usage
Start the UI:
```bash
python main.py
```
- **Open Video…** → Live preview with HUD  
- **Analyze Video (No Preview)** → Offline analysis, ends with **Save As** dialog (Excel/CSV)  
- **Live Webcam / Start IP Camera** → Preview from camera

**Hotkeys (preview window):**
- `q` → Quit
- `e` → Export report (same as clicking the Export button)

---

## ⚙️ Settings Guide
(unchanged, as in previous version…)

---

## 🧪 Experimental Results
(unchanged, as in previous version…)

---

## 💻 Code Architecture

The project is built using a **Modular Pipeline Architecture** combined with an **Event-Driven UI**.  
While not strictly MVC, the design achieves similar separation of concerns — isolating data input, detection logic, visualization, and user interaction.

### 🏗️ Architecture Type
> **Modular Pipeline with Event-Driven UI**

This structure ensures that each functional block (input, detection, tracking, and visualization) can be modified or replaced independently without affecting the rest of the system.

---

### 📂 Project Structure Overview

| **Layer** | **Description** | **Example Files** |
|------------|-----------------|------------------|
| **1️⃣ Input Layer** | Handles video input, frame extraction, and basic preprocessing. | `io.py`, `pipeline.py` |
| **2️⃣ Detection Layer** | Runs YOLOv5 model exported to ONNX for lightweight detection of people in each frame. | `tracking.py` |
| **3️⃣ Tracking & Logic Layer** | Performs centroid calculation, direction detection, and cross-line logic for entry/exit counting. | `direction.py`, `drawing.py` |
| **4️⃣ Visualization Layer** | Draws bounding boxes, line indicators, and live counters on the video stream. | `drawing.py` |
| **5️⃣ UI Layer** | Provides user controls and interactive interface for running or stopping the system. | `ui_main.py`, `ui_settings.py`, `widgets.py` |
| **6️⃣ Configuration Layer** | Defines adjustable parameters (paths, thresholds, and detection settings). | `config.py` |

---

### 🧠 Design Highlights

- 🧩 **Pipeline-based processing:** Frames flow through a defined sequence — Input → Detect → Track → Count → Draw.  
- 🪶 **Lightweight modules:** Each step is contained in a separate file to enhance readability and maintainability.  
- ⚙️ **Decoupled UI:** Graphical interface elements are independent from processing logic.  
- 💡 **CPU-Optimized:** Designed for real-time operation without GPU dependency.  
- 🔁 **Extensible:** New detectors (e.g., DeepSORT, ByteTrack) or trackers can be easily integrated.  

---

## 📜 License
MIT License. Free to use and modify.
