# 🍱 Food Donation Matcher

**Problem:** India wastes **~74 million tonnes of food a year** while ~200 million people are undernourished. Restaurants, wedding halls, and hostels dump perfectly good surplus daily — while NGOs nearby would gladly take it. The gap isn't supply; it's **information**.

**Solution:** A matching engine that pairs every food donor with the **best NGO** using distance, food-safety time windows, capacity, and quantity — and reports the impact in meals saved.

## 🧠 The algorithm inside (operations research, from scratch)
```
donor × NGO pairs → feasibility check → score → greedy best-first assignment
```
- **Haversine formula** (from scratch) — true great-circle GPS distance on Earth
- **Food-safety hard constraints:** NGO must arrive before cooked food expires (~2 hrs); this constraint *outranks* any score
- **Multi-criteria scoring:** distance, pickup time (modelled on ~25 km/h city traffic), capacity mismatch
- **Greedy matching:** best pairs assigned first, each donor/NGO used once per run — the same dispatch-algorithm class used by food-rescue orgs and delivery platforms

## ▶️ Run
```bash
python main.py
```
Zero dependencies. Ships with realistic Bengaluru donor/NGO data (real GPS coords) — see matches, reasons, and unmatched donors flagged for outreach.

## 🌍 Portfolio talking point
Most students build *prediction* projects; you built an **optimization/matching system** with real-world constraints (perishability!). The one-liner for essays: *"I built logistics software for food that would otherwise rot."* Pair it with projects 12–15 and you have a genuine 'AI for social good' portfolio arc.

## 🚀 Extension ideas
- NGO signup + donor web form (Flask) → live database
- Route optimization for multi-pickup NGO runs ( Travelling Salesman flavour)
- Time-series dashboard: meals saved per month
