# ♻️ Waste Classifier — "Which Bin?"

**Problem:** Waste segregation fails at home because people genuinely don't know which bin a pizza box, milk packet, or broken toy belongs in. So everything lands in one bag and recycling dies downstream.

**Solution:** Photo of any item → material classification → **exact bin (wet/dry/recyclable/hazardous/e-waste) + the reason why**, aligned with India's Solid Waste Management Rules 2016.

## 🧠 The two-layer intelligence (this is the design that impresses)
```
photo → [Layer 1: CNN] → material → [Layer 2: expert rules] → bin + explanation
```
- **Layer 1 (learned):** MobileNetV2 transfer learning — material classes: cardboard, glass, metal, paper, plastic, trash, battery, e-waste
- **Layer 2 (curated):** rules encode *why* (plastic code 1/2/5 recyclable, 3/6/7 often rejected in India; batteries never in regular bins)

Pure ML can't know local recycling policy; pure rules can't see a photo. Combining both is real product engineering.

## ▶️ Run
```bash
pip install -r requirements.txt

# Dataset: kaggle.com/datasets/mostafaabla/garbage-classification (12,509 images)
# extract into dataset/<class-name>/ folders, then:

python main.py --train
python main.py milk_packet.jpg
```

Example output:
```
Material: plastic (94% confident)
→ ♻️ Recyclable bin (BLUE) — CHECK code
  Look for recycling code 1,2,5 (recyclable). Code 3,6,7 often rejected in India.
```

## 🚀 Extension ideas
- Gradio phone-camera web app
- Add compostable (wet waste) class with food images
- Map output to your city's bin colours
