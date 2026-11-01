# 🎓 Student Dropout Risk Predictor

**Problem:** Students don't drop out suddenly — they fade. Attendance slips, grades slide, homework stops... and schools notice only when the seat is empty. By then it's too late to help.

**Solution:** A model that screens students **early** from attendance, grades, homework rate, and absence streaks — flags at-risk students **with the reason path attached**, so teachers know *how* to intervene (an attendance problem and an academic struggle need different help).

## 🧠 The ML inside
- **Decision tree implemented completely from scratch** — entropy, information gain, recursive best-split search, max-depth pruning
- Trains on a 400-student synthetic dataset built from real dropout-risk research factors
- Evaluated honestly on a **held-out 100-student test set** with accuracy reported
- **Explainability is the whole point:** the model prints human-readable rules like
  `if attendance_pct ≤ 68.5 AND avg_grade ≤ 51.2 → ⚠️ AT RISK` — plus the exact rule path per flagged student

## ▶️ Run
```bash
python main.py
```
Zero dependencies. You'll see: test accuracy → the learned rule tree → 3 example students screened, flagged ones with reasons.

## 🌍 Portfolio talking point
Why a decision tree and not a neural net? *Because the user is a teacher, and teachers intervene based on reasons, not probabilities.* Choosing an explainable model over a fashionable one is the mark of a real engineer — say exactly that in interviews.

## 🚀 Extension ideas
- Train on a real dataset (Kaggle "Dropout" / UCI education datasets)
- Add feature: guardian occupation, distance from school (real Indian dropout factors)
- Weekly auto-screening report per class
