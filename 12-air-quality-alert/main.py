"""
AIR QUALITY ALERT SYSTEM
Problem: Air pollution spikes in Indian cities (AQI > 300 in Delhi winters)
catch people off-guard. Asthma patients, children, and elderly get hit because
nobody warns them the day BEFORE the bad air arrives.

Solution: Fetches REAL live AQI data (open data.forsociety.pl / OpenAQ style
API), builds a short-term forecast, and gives personalized health advice:
  - Tomorrow's expected AQI band
  - Who should stay indoors / wear a mask
  - Trend compared to yesterday

Model: Weighted moving average + linear trend (from scratch) — classic
time-series forecasting, honest about being simple. The architecture allows
plugging in ARIMA/LSTM later (see README).

API: works with any OpenAQ-compatible endpoint; includes a built-in demo
dataset so it runs with zero setup even offline.

  python main.py              # forecast from built-in demo data
  python main.py --city delhi # fetch live data (requires internet + API)
"""

import json
import sys
import urllib.request
from datetime import datetime, timedelta

# ---------------------------------------------------------------------
# Health advice bands (Indian CPCB AQI categories)
# ---------------------------------------------------------------------
def aqi_advice(aqi: float) -> tuple[str, str]:
    if aqi <= 50:
        return "Good ✅", "Air is clean. Enjoy outdoor activities."
    if aqi <= 100:
        return "Satisfactory 🙂", "Fine for most people; sensitive ones watch for symptoms."
    if aqi <= 200:
        return "Moderate 😷", "Asthma/heart patients: reduce prolonged outdoor exertion."
    if aqi <= 300:
        return "Poor ⚠️", "Everyone: limit outdoor activity; wear N95 mask outside."
    if aqi <= 400:
        return "Very Poor 🚨", "Children & elderly stay indoors; use air purifier if possible."
    return "Severe ☠️", "Everyone indoors. Masks mandatory outside. Avoid exertion."

# ---------------------------------------------------------------------
# Data source 1: built-in demo history (Delhi-style winter pattern)
# ---------------------------------------------------------------------
DEMO_HISTORY = [342, 355, 349, 361, 373, 380, 377, 390, 395, 388, 402, 410, 407]

# ---------------------------------------------------------------------
# Data source 2: live fetch (OpenAQ-compatible endpoint)
# ---------------------------------------------------------------------
def fetch_live(city: str) -> list[float]:
    url = f"https://api.openaq.org/v2/averages?city={city}&parameter=pm25"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aqi-school-project/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        values = [row["value"] for row in data.get("results", []) if row.get("value")]
        if len(values) < 5:
            raise ValueError("not enough data returned")
        return values[-14:]
    except Exception as e:
        print(f"⚠️ Live fetch failed ({e}). Using built-in demo history instead.")
        return DEMO_HISTORY

# ---------------------------------------------------------------------
# Forecast: weighted moving average + linear trend (from scratch)
# ---------------------------------------------------------------------
def forecast_aqi(history: list[float]) -> dict:
    if len(history) < 3:
        raise ValueError("Need at least 3 days of history")

    # Weighted MA — recent days matter more (weights 1, 2, 3, ...)
    recent = history[-7:] if len(history) >= 7 else history
    weights = list(range(1, len(recent) + 1))
    wma = sum(v * w for v, w in zip(recent, weights)) / sum(weights)

    # Linear trend via least-squares slope over last 7 points
    n = len(recent)
    xs = list(range(n))
    mean_x, mean_y = sum(xs) / n, sum(recent) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, recent)) / \
            sum((x - mean_x) ** 2 for x in xs) or 0.0

    tomorrow = wma + slope
    day_after = tomorrow + slope * 0.6  # damped second-day estimate

    return {
        "yesterday": history[-2],
        "today": history[-1],
        "tomorrow": round(tomorrow, 1),
        "day_after": round(day_after, 1),
        "trend": "rising 📈" if slope > 1 else ("falling 📉" if slope < -1 else "stable ➡️"),
    }

# ---------------------------------------------------------------------
def report(city: str, history: list[float]):
    f = forecast_aqi(history)
    band, advice = aqi_advice(f["tomorrow"])

    print(f"\n🌍 Air Quality Report — {city.title()}")
    print("=" * 48)
    print(f"  Yesterday : AQI {f['yesterday']:.0f}")
    print(f"  Today      : AQI {f['today']:.0f}")
    print(f"  TOMORROW   : AQI {f['tomorrow']:.0f}  ← forecast")
    print(f"  Trend      : {f['trend']}")
    print(f"\n  Tomorrow's band: {band}")
    print(f"  👉 {advice}")

    if f["tomorrow"] > 300 and f["trend"].startswith("rising"):
        print("\n  🚨 ALERT: Sustained severe air expected. Consider:")
        print("     • Keeping asthmatic family members indoors")
        print("     • Buying N95 masks before stock runs out")
        print("     • School sports events likely to be cancelled")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    if "--city" in sys.argv:
        city = sys.argv[sys.argv.index("--city") + 1]
        history = fetch_live(city)
    else:
        city = "delhi (demo)"
        history = DEMO_HISTORY
    report(city, history)