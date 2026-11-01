# ⚡ Energy Bill Predictor

**Problem:** Families get blindsided by their electricity bill every month — no way to know in advance, no idea which appliance is the money-eater, no budgeting.

**Solution:** Enter daily usage hours of your 5 main appliances → get:
- Your **predicted monthly bill** (₹)
- **Cost per hour** each appliance actually incurs (learned by the model, not hardcoded)
- **Savings advice:** what a 10% cut to your costliest appliance saves per month *and* per year

## 🧠 The ML inside
**Multivariate linear regression via gradient descent — written from scratch** (no sklearn):
- Learns `bill = w₁·fan_hrs + w₂·light_hrs + w₃·AC_hrs + w₄·fridge_hrs + w₅·TV_hrs + bias`
- Trained on 300 realistic households (Indian tariff ~₹6.5/unit, correct appliance wattages)
- Evaluated with **R² score** on held-out test data (typical: ~0.99)

The magic moment: the model **discovers** that AC costs the most — nobody told it. That's real ML.

## ▶️ Run
```bash
python main.py
```
No dependencies. Edit `my_usage` in the demo to predict your own household.

## 🚀 Extension ideas
- CSV import of your actual electricity bills
- Season-aware features (summer AC usage spike)
- Turn into a Streamlit web app for a live demo
