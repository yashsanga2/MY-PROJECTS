# 🧠 Mood Journal — NLP Emotion Tracker

**Problem:** Student stress builds silently. By the time anyone notices, it's already a crisis. Journals exist, but nobody rereads them — and patterns stay invisible.

**Solution:** A **private, offline journal** that detects the emotion in each entry, tracks your mood history, and produces a weekly report with trend detection — including early-warning signs and real helplines.

## 🧠 The NLP inside (explainable, from scratch)
- **Lexicon-based emotion classifier** with 5 emotion classes (happy, sad, angry, anxious, neutral)
- **Negation handling:** "not happy" correctly flips toward negative
- **Intensifier weighting:** "very sad" weighs more than "sad"
- **Punctuation signals:** "!!!" adds anxious/angry signals
- **Explainability:** it *shows you the exact words* that drove each classification — a black box helps nobody in mental health
- **Trend detection:** compares negativity across recent entries → "worsening ⚠️" vs "stable ✅"

## ▶️ Run
```bash
python main.py           # write a journal entry
python main.py --report  # weekly mood report
```
Zero dependencies. Your entries stay in `journal.json` on **your machine only**.

## ⚖️ Ethics (worth mentioning in interviews)
This is a *self-awareness tool*, not a diagnosis. The weekly report deliberately includes the Govt. of India **Tele-MANAS helpline (14416)** when sustained distress is detected. Privacy-by-design: no cloud, no telemetry.

## 🚀 Extension ideas
- Password-protect journal.json (cryptography library)
- Mood-over-time matplotlib chart
- Upgrade classifier to a pretrained transformer (goemotions) and compare accuracy
