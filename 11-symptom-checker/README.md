# 🩺 Symptom Checker (Educational Triage)

**Problem:** Symptom-Googling leads to two disasters: panic ("I have 15 minutes to live!") or dangerous ignoring ("it's probably nothing"). There's no fast, calm middle ground.

**Solution:** Enter symptoms → ranked **likely common conditions** with matched-symptom evidence, **red-flag emergency alerts** (chest pain → call 108 NOW), and an honest doctor disclaimer on every output.

## 🧠 The ML inside
- **Multi-label probabilistic scoring (Bayesian, from scratch)** — each disease has a weighted symptom profile; input symptoms are matched and normalized into a likelihood share
- **Safety-first architecture:** red-flag symptoms short-circuit the ranking entirely — emergency advice comes before any ML output. This ordering is a *design decision*, and it's the best interview talking point in the whole repo
- Knowledge base covers 7 common Indian conditions (cold, flu, dengue-like viral fever, gastroenteritis, migraine, anemia, typhoid)

## ▶️ Run
```bash
python main.py
> fever, body ache, rash, eye pain     # → likely viral fever, advises dengue test
```
Zero dependencies.

## ⚖️ Ethics statement (put this in your application)
> "Good AI knows its limits. This system triages; it never diagnoses. Emergency signals bypass the model entirely, and every output ends with a doctor referral."

That sentence alone demonstrates engineering maturity most applicants lack.

## 🚀 Extension ideas
- Real dataset (e.g. Disease Symptom Prediction, Kaggle) + ML classifier
- Confidence calibration + "I don't know" output class
- Multi-language symptom input (Hindi/English)
