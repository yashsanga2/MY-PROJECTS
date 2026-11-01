# 📵 Fake Forward Detector

**Problem:** WhatsApp misinformation spreads panic daily — fake cures, fake prizes, "forward to 10 people or else...". People forward without checking, and lies travel faster than truth.

**Solution:** Paste any forwarded message → get a **REAL / FAKE verdict**, a **confidence score**, and the **exact words that triggered the decision** (explainability matters — a black box helps nobody).

## 🧠 The ML inside
**Multinomial Naive Bayes, written completely from scratch** (no sklearn):
- Bayes' theorem: $P(\text{fake} \mid \text{words}) \propto P(\text{words} \mid \text{fake}) \cdot P(\text{fake})$
- Laplace smoothing for unseen words
- Softmax over log-scores → calibrated confidence
- Per-word evidence diff → "which words made it look fake?"

## ▶️ Run
```bash
python main.py
```
Runs a held-out test set (reports accuracy) then classifies 4 sample messages with explanations.

## 📊 Data
Comes with 20 patterned messages (10 fake / 10 real). **To make it stronger:** add 100+ real WhatsApp forwards you've received to `TRAINING_DATA`, or load a CSV of messages.

## 🚀 Extension ideas
- Train on a public fake-news dataset (e.g. ISOT Fake News dataset) for serious accuracy
- WhatsApp-style input loop in the terminal
- Confidence below 60% → verdict "uncertain, please fact-check at pib.gov.in"
