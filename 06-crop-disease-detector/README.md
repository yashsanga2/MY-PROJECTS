# 🌾 Crop Disease Detector

**Problem:** Farmers lose **30–40% of harvest** to plant diseases identified too late. Agricultural extension officers can't reach every village in time, and by the time symptoms are obvious, treatment is expensive or useless.

**Solution:** Photo of an affected leaf → **instant disease diagnosis + treatment advice** in plain language. Built for smallholder farmers — the model (MobileNetV2) is small enough to run on budget Android phones.

## 🧠 The ML inside
- **Transfer learning** with **MobileNetV2** pretrained on ImageNet — reuses learned visual features (edges, spots, blight textures), so it trains in minutes on a laptop with a small dataset
- Frozen convolution base + new classification head (GlobalAvgPool → Dropout → Dense → Softmax)
- Data augmentation (flip/rotate/zoom) to prevent overfitting on small datasets
- Early stopping + checkpointing on best validation accuracy

## ▶️ Run
```bash
pip install -r requirements.txt

# 1. Get data: download PlantVillage dataset (kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
#    and put class folders inside  dataset/  (even 10 images/class works for a demo)

python main.py --train          # fine-tune → saves crop_model.keras
python main.py leaf.jpg         # diagnose a leaf photo
```

Output example:
```
🔬 Result for leaf.jpg
   Disease: Tomato___Late_blight  (confidence 91%)
   💊 Act fast — remove and burn infected plants; spray metalaxyl + mancozeb.
```

## 🌍 Why this matters
This is the classic "AI for social good" project — publications, interviews, and essays love it because the impact story is concrete: earlier detection = saved crops = farmer income.

## 🚀 Extension ideas
- Gradio web demo (3 lines: `gr.Interface(fn=predict, inputs="image", outputs="text").launch()`)
- Hindi/regional-language output
- Severity estimate from lesion area
