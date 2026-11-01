"""
STUDYSENSE AI — Student Success Companion  (FLAGSHIP CAPSTONE)
════════════════════════════════════════════════════════════════
This is not one project. It is the union of four:

  Project 01 (TF-IDF quiz generation)  → measures WHAT you know
  Project 07 (emotion lexicon NLP)    → measures HOW you feel
  Project 15 (decision-tree risk model)→ predicts if you're exam-ready
  + New: personalized study-plan generator → tells you WHAT TO DO NEXT

Together they form a full ML pipeline — the way real products are built
from components. A student types their notes + how their week went, and
StudySense outputs: knowledge gaps, stress level, readiness risk, and a
step-by-step plan for the next 7 days.

════════════════════════════════════════════════════════════════
Problem (personal): Students like me study hard yet walk into exams
blind — we don't know our weak topics, we don't notice our own stress
building, and no tool connects feelings, facts, and time into one honest
picture. I built StudySense to be that picture.
"""

import math
import re
from collections import Counter
from datetime import date, timedelta

# ═══════════════════════════════════════════════════════════════
# STAGE 1 — Knowledge extraction (from Project 01: TF-IDF)
# ═══════════════════════════════════════════════════════════════
STOPWORDS = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
             "in", "of", "to", "for", "with", "as", "by", "that", "this",
             "it", "from", "are", "was", "were", "be", "been", "has", "have"}

def tokenize(text: str) -> list:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def split_sentences(text: str) -> list:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if len(s.split()) >= 4]

def tfidf_keywords(notes: str, n: int = 8) -> list:
    sentences = split_sentences(notes)
    if not sentences:
        return []
    docs = [tokenize(s) for s in sentences]
    df = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    scores = {}
    for word, freq in df.items():
        idf = math.log((len(docs) + 1) / (freq + 1)) + 1
        best_tf = max((Counter(d)[word] for d in docs), default=0)
        scores[word] = best_tf * idf
    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]

def make_quiz(notes: str, n_questions: int = 5) -> list:
    sentences = split_sentences(notes)
    keywords = tfidf_keywords(notes, 15)
    quiz, used = [], set()
    for word in keywords:
        for s in sentences:
            if word in used:
                break
            if re.search(rf"\b{re.escape(word)}\b", s, re.IGNORECASE):
                blanked = re.sub(rf"\b{re.escape(word)}\b", "_____", s,
                                 flags=re.IGNORECASE)
                quiz.append({"q": blanked, "a": word})
                used.add(word)
                break
        if len(quiz) >= n_questions:
            break
    return quiz

# ═══════════════════════════════════════════════════════════════
# STAGE 2 — Stress detection (from Project 07: emotion lexicon)
# ═══════════════════════════════════════════════════════════════
NEGATIVE_WORDS = {"tired", "exhausted", "stressed", "pressure", "anxious",
                  "nervous", "scared", "worried", "overwhelmed", "sad",
                  "angry", "frustrated", "cant", "sleep", "panic", "behind"}
POSITIVE_WORDS = {"good", "great", "confident", "calm", "ready", "fun",
                  "happy", "proud", "better", "improved"}

def stress_score(journal: str) -> dict:
    words = journal.lower().split()
    neg = sum(1 for w in words if w.strip(".,!?") in NEGATIVE_WORDS)
    pos = sum(1 for w in words if w.strip(".,!?") in POSITIVE_WORDS)
    total = neg + pos
    score = neg / total if total else 0.0
    if "!" in journal and journal.count("!") >= 3:
        score = min(score + 0.15, 1.0)
    band = ("high 🔴" if score > 0.6 else
            "moderate 🟡" if score > 0.35 else "low 🟢")
    return {"score": round(score, 2), "band": band,
            "neg_words": neg, "pos_words": pos}

# ═══════════════════════════════════════════════════════════════
# STAGE 3 — Readiness risk (decision-tree style rules, from Project 15)
# ═══════════════════════════════════════════════════════════════
def readiness_risk(quiz_score: float, stress: dict, days_left: int,
                   sleep_hrs: float) -> tuple:
    """Hand-grown decision tree (rules learned from Project 15's approach)."""
    node = "start"
    if quiz_score < 0.5:
        node = "low_knowledge"
    elif stress["score"] > 0.6:
        node = "high_stress"
    else:
        node = "ok_knowledge"

    if node == "low_knowledge" and days_left <= 3:
        risk, reason = 0.85, "weak topics + very little time"
    elif node == "low_knowledge":
        risk, reason = 0.6, "weak topics detected"
    elif node == "high_stress" and sleep_hrs < 6:
        risk, reason = 0.7, "high stress + low sleep — burnout pattern"
    elif node == "high_stress":
        risk, reason = 0.45, "stress is eating your study efficiency"
    elif days_left <= 3 and quiz_score < 0.8:
        risk, reason = 0.5, "exam close, knowledge not solid yet"
    else:
        risk, reason = 0.15, "on track"
    return risk, reason

# ═══════════════════════════════════════════════════════════════
# STAGE 4 — Study plan generator (new component)
# ═══════════════════════════════════════════════════════════════
def make_plan(weak_keywords: list, quiz_score: float, stress: dict,
              risk: float, days_left: int) -> list:
    plan = []
    focus_days = max(days_left - 1, 1)
    for i in range(min(focus_days, 7)):
        day = date.today() + timedelta(days=i)
        if i % 2 == 0 and weak_keywords:
            topic = weak_keywords[i % len(weak_keywords)]
            plan.append(f"Day {i+1} ({day:%a}): 45 min on '{topic}' — rewrite it "
                        f"in your own words + 5 practice questions")
        else:
            plan.append(f"Day {i+1} ({day:%a}): full mock test + review every "
                        f"mistake (mistakes are the syllabus)")
    if stress["score"] > 0.35:
        plan.append("Every evening: 20-min walk + no phone 30 min before sleep")
    if risk > 0.6:
        plan.append("Talk to your teacher about the 2 weakest topics THIS week")
    plan.append("Night before exam: summary only, sleep 8h — memory consolidates asleep")
    return plan

# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def analyze(notes: str, journal: str, quiz_answers: dict,
            days_left: int, sleep_hrs: float) -> dict:
    # Stage 1: knowledge
    keywords = tfidf_keywords(notes)
    quiz = make_quiz(notes, 5)
    correct = sum(1 for q in quiz if quiz_answers.get(q["a"]) == q["a"])
    quiz_score = correct / len(quiz) if quiz else 0.0

    # Stage 2: stress
    stress = stress_score(journal)

    # Stage 3: risk
    risk, reason = readiness_risk(quiz_score, stress, days_left, sleep_hrs)

    # Stage 4: plan
    plan = make_plan(keywords, quiz_score, stress, risk, days_left)

    return {"keywords": keywords, "quiz": quiz, "quiz_score": quiz_score,
            "stress": stress, "risk": risk, "risk_reason": reason, "plan": plan}

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    NOTES = """
    Photosynthesis converts sunlight into chemical energy inside chloroplasts.
    Chlorophyll absorbs light in the blue and red regions of the spectrum.
    Light-dependent reactions occur in thylakoid membranes and produce ATP.
    The Calvin cycle fixes carbon dioxide into glucose inside the stroma.
    Respiration releases energy from glucose in the mitochondria of cells.
    """
    JOURNAL = """
    This week was exhausting. I feel behind on biology and stressed about the
    test. I keep making mistakes in practice papers and I'm anxious. !!
    """
    # simulate a student answering: they know photosynthesis, not respiration
    answers = {"photosynthesis": "photosynthesis", "chlorophyll": "chlorophyll",
               "atp": "atp", "calvin": "calvin", "respiration": "?????"}

    result = analyze(NOTES, JOURNAL, answers, days_left=6, sleep_hrs=5.5)

    print("═" * 62)
    print("🎓 STUDYSENSE AI — your honest exam-readiness picture")
    print("═" * 62)

    print(f"\n📖 KNOWLEDGE (Stage 1 — TF-IDF, from Project 01)")
    print(f"   Key topics in your notes: {', '.join(result['keywords'][:5])}")
    print(f"   Quick quiz score: {result['quiz_score']:.0%}")
    for q in result["quiz"]:
        print(f"     • {q['q'][:70]}...")

    print(f"\n🧠 STRESS (Stage 2 — emotion lexicon, from Project 07)")
    print(f"   Journal stress level: {result['stress']['band']} "
          f"({result['stress']['neg_words']} stress signals found)")

    print(f"\n⚠️  READINESS RISK (Stage 3 — decision logic, from Project 15)")
    print(f"   Risk: {result['risk']:.0%} — {result['risk_reason']}")

    print(f"\n🗓️  YOUR 7-DAY PLAN (Stage 4 — generated for YOU)")
    for step in result["plan"]:
        print(f"   ▸ {step}")

    print("\n💬 One honest sentence: the plan only works if you follow it. Start today.")