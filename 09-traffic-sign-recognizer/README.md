# 🚦 Traffic Sign Recognizer (Indian Roads)

**Problem:** India's roads have 100+ sign types. New drivers, licence learners, and tourists miss critical signs (wrong-way entry, no-overtaking, give-way) — and missing a sign can be fatal.

**Solution:** Photograph a sign → CNN classifies it → the app tells the driver **what to actually do**, not just the sign's name.

## 🧠 The ML inside
- **Custom CNN, 3 conv blocks** (32→64→128 filters), BatchNorm + Dropout — a serious architecture you designed yourself, not a copy
- Trains on **GTSRB** (German Traffic Sign Benchmark): 50,000 images, 43 classes — its conventions (red circles = prohibition, triangles = warning, blue = mandatory) match Indian signage closely
- Early stopping on validation accuracy; typically reaches **~97%+ test accuracy**

## ▶️ Run
```bash
pip install -r requirements.txt

# Download GTSRB (Pickles format) from kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
# Convert to train_images.npy (48×48×3, float32 /255) and train_labels.npy, then:

python main.py --train
python main.py sign.jpg
```

## 🌍 The value-add
Every class maps to a **driver action** ("STOP" → *full stop; proceed only when clear*). Classification alone is a school project; classification → *actionable decision* is engineering.

## 🚀 Extension ideas
- Collect your own Indian sign photos and fine-tune (transfer learning)
- Real-time dashcam version with OpenCV loop
- Voice alert output for accessibility
