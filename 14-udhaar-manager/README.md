# 📒 Udhaar Manager — Shop Credit Tracker + Risk Prediction

**Problem:** Crores of Indian kirana shops run on *udhaar* — informal credit tracked in a paper notebook. Bad debts quietly pile up; by the time a shopkeeper notices, the money's gone.

**Solution:** A digital udhaar register that tracks credit/repayment history per customer, computes a **risk score with reasons** ("3 credits without repayment in a row"), and flags who's becoming dangerous *before* the next handout.

## 🧠 The ML inside
A **transparent logistic risk-scoring model, from scratch** — the same math banks use for credit scores:
$$\text{risk} = \sigma(z), \quad z = w_1 \cdot \text{(low repayment ratio)} + w_2 \cdot \text{(unpaid streak)} + w_3 \cdot \text{(high balance)} - b$$
- Features engineered from raw history: repayment ratio, consecutive unpaid credits, outstanding balance
- **Explainability by design** — every score ships with human-readable reasons, because a shopkeeper needs to *see* why their regular customer got flagged

## ▶️ Run
```bash
python main.py
```
Seeds a realistic 3-customer database into `udhaar.json` (edit it with real data or add your own entries). Zero dependencies.

Output ranks customers 🔴 HIGH → 🟢 safe with reasons for each score.

## 🌍 Portfolio talking point
Credit-scoring with explainability is a live, billion-dollar industry problem (CIBIL scores, fintech lending). Building a toy version that a shopkeeper could actually understand demonstrates both ML skill and product sense.

## 🚀 Extension ideas
- SMS/WhatsApp reminder drafts per flagged customer (polite ones!)
- Logit weights learned from labelled outcomes instead of hand-set (real logistic regression)
- Streamlit dashboard for the shop counter
