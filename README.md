# 🚦 Smart Traffic Monitoring System (YOLOv8 + ByteTrack + OCR)

<h2 align="center">🚗 Safe Road - AI Traffic Monitoring System</h2>

<p align="center">
  <img src="https://github.com/user-attachments/assets/feab6498-f189-45de-92f9-57f04e0710df" width="800"/>
</p>

An end-to-end AI-based traffic monitoring system that performs real-time vehicle detection,
tracking, speed estimation,violation detection, and Arabic license plate recognition.

---

## 🚀 Project Overview

This project consists of four main intelligent components:

🚗 Vehicle Detection using YOLOv8  
🔄 Multi-Object Tracking using ByteTrack  
⚡ Real-time Speed Estimation  
🔍 Violation Detection Logic  
🔢 Arabic License Plate Recognition (Detection + OCR)

The system automatically detects:

- 🚨 Speeding Vehicles
- 🔁 Wrong Direction Driving
- 🛑 Illegal Stopping
- ⚠️ Suspicious Vehicles
- 🔢 Extracts Arabic License Plate Numbers

---

## 📂 Project Structure

├── main.py                    
├── config.py                 
├── vehicle_tracker.py        
├── violation_detector.py      
├── utils.py                  
├── Models                    
├── video test           
├── output                   
└── README.md  

---

## 🧠 System Architecture

### 1️⃣ Vehicle Detection (YOLOv8)

- Uses YOLOv8 for detecting cars and trucks
- Filters specific class IDs
- Applies confidence threshold

### 2️⃣ Multi-Object Tracking

- Persistent tracking IDs
- Position history buffer
- Speed history tracking
- Automatic cleanup of old tracks

### 3️⃣ Speed Calculation

Speed is calculated using:

- Pixel displacement between frames
- FPS calibration
- Pixels-per-meter ratio
- Converted from m/s to km/h

### 4️⃣ Violation Detection Logic

The system checks for:

| Violation Type       | Description |
|----------------------|------------|
| Speeding             | Exceeds configured speed threshold |
| Wrong Direction      | Moves opposite vertical direction |
| Illegal Stopping     | Stops longer than allowed duration |
| Suspicious Vehicle   | Stationary for long duration |

Cooldown logic prevents duplicate violations.

### 5️⃣ License Plate Recognition

- Detects plate inside vehicle bounding box
- Crops plate region
- Applies OCR model
- Maps characters to Arabic format
- Sorts digits and letters spatially

---

## 📊 Output Generated

The system automatically creates:

📸 Violation images with bounding boxes  
📄 JSON file per violation  
🎥 Fully processed output video  
📈 Live statistics (vehicles + violations count)

Example JSON Output:

```json
[
  {
    "track_id": 16,
    "plate_number": "ط ف س ٥٨٦",
    "violation_type": "wrong_direction",
    "real_time": "2026-02-24 15:24:06",
    "description": "المركبة 16 تسير عكس الاتجاه",
    "image_filename": "violation_16_wrong_direction_20260224.jpg"
  }
]
```

### 5️⃣ License Plate Recognition

- Detects plate inside vehicle bounding box
- Crops plate region
- Applies OCR model
- Maps characters to Arabic format
- Sorts digits and letters spatially

---

## 🧪 Technologies Used

- Python
- Ultralytics YOLOv8
- OpenCV
- NumPy
- ByteTrack
- Deep Learning (Custom OCR Model)

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
pip install ultralytics opencv-python numpy


### 2️⃣ Download Required Models

Place the following models in the project root directory:
- yolov8n.pt
- plate_detector.pt
- reader_ocr.pt


### 3️⃣ Run the System

python main.py

---

## 💡 Key Learning Outcomes

- Building a complete Computer Vision pipeline
- Real-time multi-object tracking
- Speed estimation using geometry
- Violation detection logic engineering
- OCR for Arabic license plates
- Structured JSON logging system
- Production-style project architecture

---

## 👨‍💻 Author

Ibrahem Sayed  
Ai & Computer Vision Engineer
