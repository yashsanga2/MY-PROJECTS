# 🎓 StudySense AI — Student Success Companion *(Flagship Capstone)*

> *"I built 19 tools that solve problems for others. This one solves my own."*

**The personal problem:** Students study hard yet walk into exams blind. We don't know our weak topics, we don't notice our stress building, and nothing connects **what we know + how we feel + how much time is left** into one honest picture. So I built that picture.

## 🧠 The architecture — where 4 portfolio projects become one pipeline
```
        NOTES ──→ [Stage 1] TF-IDF keyword + quiz engine  (from Project 01)
                       │            → knowledge score
        JOURNAL ─→ [Stage 2] emotion lexicon classifier   (from Project 07)
                       │            → stress score
                       ↓
                 [Stage 3] decision-tree readiness rules  (from Project 15)
                       │            → risk % + reason
                       ↓
                 [Stage 4] study-plan generator           (new)
                                    → a day-by-day plan, personalized
```

| Input | Model | Output |
|---|---|---|
| Study notes | TF-IDF (Project 01) | weak topics + quiz score |
| Weekly journal | Emotion lexicon (Project 07) | stress band |
| Quiz score + stress + days + sleep | Decision rules (Project 15) | readiness risk + reason |
| All of the above | Plan generator (new) | 7-day study plan with rest built in |

## ▶️ Run
```bash
python main.py
```
Zero dependencies. Watch it analyze a real scenario: a tired student, weak on respiration, 6 days to the exam, 5.5h sleep — and StudySense catches every bit of it.

## 💡 Why this is the flagship (essay-ready)
1. **It unites the portfolio** — admissions officers see you didn't build 19 random projects; you built a *system* and then composed them.
2. **It's honest AI** — every stage is explainable; the final line even admits the plan only works if you follow it.
3. **It's personal** — you are the user. That's the strongest origin story a student project can have.

## 🚀 Extension roadmap
- Streamlit web UI (one weekend)
- Real quiz interaction (typed answers instead of simulated)
- Track predictions vs. actual exam results over a term — closing the ML feedback loop
