# 🪖 Helmet & Seatbelt Detector

**Problem:** Head injuries are the top killer of two-wheeler riders in India. Traffic police can't watch every junction, and manual checking creates jams. Violations go unpoliced at night and on highways.

**Solution:** A camera pipeline that automatically finds riders, checks each one for a helmet, and generates a **violation report with cropped evidence photos** — the exact tech smart-city traffic systems use.

## 🧠 The 3-layer pipeline
```
frame → [YOLOv8 detection] → [rider-helmet pairing logic] → [evidence logging]
```
1. **YOLOv8** (ultralytics) — real-time object detection of persons, motorcycles, bicycles
2. **Geometric pairing logic** — a rider is a person vertically overlapping a bike; helmet check = detected object over the top 30% of the rider box. Custom helmet classes make this exact (`--train` mode included)
3. **Evidence crops** — every violation saved as a timestamped image for human review (a real system never auto-fines — a human confirms. Say that; it shows responsibility)

## ▶️ Run
```bash
pip install -r requirements.txt

python main.py --demo street.jpg    # analyze one photo → result.jpg + violations/
python main.py --video traffic.mp4   # frame-by-frame analysis
python main.py --train               # fine-tune on Roboflow helmet dataset (dataset.yaml)
```

## 🌍 Portfolio talking point
This project shows you can build **safety-critical AI with accountability built in** — human review before any penalty, evidence logging, and a custom-training path for real deployment. Also: it runs on a laptop GPU or CPU, no cloud costs.

## 🚀 Extension ideas
- Seatbelt detection inside cars (crop windshield region → classification CNN)
- Number-plate extraction (YOLO + OCR) for automated reporting
- Live RTSP camera feed with SQLite violation database
