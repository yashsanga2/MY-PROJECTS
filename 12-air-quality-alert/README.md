# 🌫️ Air Quality Alert System

**Problem:** Every winter, Delhi's AQI crosses 300+ and asthma patients, children, and the elderly get caught completely off-guard. People react *after* the bad air arrives — masks sell out, sports days get cancelled, hospitals fill up.

**Solution:** A system that fetches **real live AQI data**, forecasts **tomorrow's air quality**, and gives **personalized health alerts** — before the spike, not after.

## 🧠 The ML inside
- **Weighted Moving Average** — recent days count more (weights 1, 2, 3...)
- **Least-squares linear trend** — slope over the last 7 days, damped for the day-after estimate
- Honest engineering: simple methods chosen deliberately for explainability; the architecture is ready to swap in ARIMA or an LSTM (that's a stretch goal, not a crutch)
- **Live data integration** — OpenAQ-compatible API with graceful fallback to a built-in demo dataset (runs offline, zero setup)

## ▶️ Run
```bash
python main.py              # instant demo with realistic Delhi winter data
python main.py --city delhi # real live data (internet required)
```
Zero dependencies — even the HTTP call is stdlib.

## 📊 Output
```
TOMORROW   : AQI 401.3  ← forecast
Trend      : rising 📈
Tomorrow's band: Severe ☠️
🚨 ALERT: Sustained severe air expected...
```

## 🌍 Portfolio talking point
Uses **Indian CPCB AQI bands** (Good → Severe) with health advice per band — grounded in local standards, not generic US EPA numbers. Judges notice that detail.

## 🚀 Extension ideas
- Daily WhatsApp/email alert bot for your family
- City comparison dashboard (Delhi vs Mumbai vs Bengaluru)
- Upgrade forecast to LSTM once you have 365+ days of data
