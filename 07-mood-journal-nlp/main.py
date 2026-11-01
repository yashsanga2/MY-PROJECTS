"""
MOOD JOURNAL — NLP EMOTION TRACKER
Problem: Student stress builds silently. By the time it's visible, it's a crisis.
Nobody notices the pattern: weeks of small negative entries → burnout.

Solution: A private journal that:
  1. Detects the emotion in each entry (lexical NLP sentiment/emotion classifier)
  2. Stores a personal mood history (JSON)
  3. Weekly report: mood trend, warning signs, self-care suggestions

Design decisions (important for portfolio):
  - 100% offline & private — your feelings never leave your laptop
  - Lexicon-based emotion detection (explainable: it can SHOW you the words it
    matched, unlike a black-box neural net)
  - Emotion-aware suggestions, not preachy generic advice

Run:  python main.py          # interactive journal
      python main.py --report # weekly summary
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

DB_FILE = Path("journal.json")

# ---------------------------------------------------------------
# Emotion lexicons (seeded — extend with your own vocabulary)
# ---------------------------------------------------------------
EMOTION_WORDS = {
    "happy": ["happy", "great", "awesome", "excited", "joy", "proud", "grateful",
              "good", "amazing", "wonderful", "fun", "celebrated", "loved", "hopeful", "calm"],
    "sad": ["sad", "down", "lonely", "hurt", "cry", "cried", "depressed", "upset",
            "heartbroken", "disappointed", "empty", "tired", "exhausted", "hopeless"],
    "angry": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated",
              "hate", "unfair", "raging", "fed up"],
    "anxious": ["anxious", "nervous", "worried", "scared", "afraid", "panic",
                "stressed", "pressure", "overwhelmed", "tense", "exam", "deadline"],
    "neutral": [],
}

INTENSIFIERS = {"very": 1.5, "so": 1.4, "really": 1.4, "extremely": 2.0, "totally": 1.5}
NEGATIONS = {"not", "never", "no", "hardly"}

# ---------------------------------------------------------------
# Emotion detection — explainable lexical classifier
# ---------------------------------------------------------------
def detect_emotion(text: str) -> dict:
    words = text.lower().split()
    scores = {e: 0.0 for e in EMOTION_WORDS}
    matched = []

    for i, word in enumerate(words):
        clean = word.strip(".,!?;:'\"()").lower()
        weight = 1.0

        # negation flips meaning ("not happy" → negative signal)
        prev = words[i - 1].strip(".,!?;:'\"()").lower() if i > 0 else ""
        if prev in NEGATIONS:
            weight *= -0.7

        # intensifiers boost ("very sad")
        if prev in INTENSIFIERS:
            weight *= INTENSIFIERS[prev]

        for emotion, vocab in EMOTION_WORDS.items():
            if clean in vocab:
                scores[emotion] += weight
                matched.append((clean, emotion, weight))

    # anxious gets a small boost from punctuation like "!!!"/"???"
    if text.count("!") >= 3:
        scores["anxious"] += 1.0
        scores["angry"] += 0.5

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best, conf = "neutral", 0.5
    else:
        total = sum(max(s, 0) for s in scores.values())
        conf = scores[best] / total if total else 0.5

    return {"emotion": best, "confidence": round(conf, 2),
            "matched_words": matched,
            "negativity": round(scores["sad"] + scores["angry"] + scores["anxious"], 2)}

# ---------------------------------------------------------------
# Journal storage
# ---------------------------------------------------------------
def add_entry():
    print("📖 Write today's entry (type your feelings, then a blank line to save):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    text = " ".join(lines)

    analysis = detect_emotion(text)
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text": text,
        **analysis,
    }
    data = json.loads(DB_FILE.read_text()) if DB_FILE.exists() else []
    data.append(entry)
    DB_FILE.write_text(json.dumps(data, indent=2))

    print(f"\n💬 Detected emotion: {analysis['emotion']} (confidence {analysis['confidence']:.0%})")
    if analysis["matched_words"]:
        print("   because of words like:", ", ".join(w for w, _, _ in analysis["matched_words"][:5]))

# ---------------------------------------------------------------
# Weekly report — trend + early warning
# ---------------------------------------------------------------
def weekly_report():
    if not DB_FILE.exists():
        print("No entries yet. Write one first!")
        return
    data = json.loads(DB_FILE.read_text())
    recent = data[-7:]
    negs = [e["negativity"] for e in recent]
    avg_neg = statistics.mean(negs) if negs else 0
    trend = "worsening ⚠️" if len(negs) >= 3 and statistics.mean(negs[-3:]) > statistics.mean(negs[:3]) else "stable ✅"

    print(f"\n📊 Last {len(recent)} entries:")
    print(f"   average negativity score: {avg_neg:.2f}")
    print(f"   trend: {trend}")

    if avg_neg > 2.5:
        print("\n💛 Your entries show sustained distress. Please talk to someone you")
        print("   trust — a friend, parent, teacher, or a helpline:")
        print("   🇮🇳 Tele-MANAS: 14416 (free, Govt. of India, 24×7)")
    elif avg_neg > 1.0:
        print("\n🌿 Some stress showing up. Ideas that help: 10-min walk, calling a")
        print("   friend, 4-7-8 breathing, and cutting late-night scrolling.")
    else:
        print("\n🌟 Your journal looks balanced. Keep it up!")

# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        weekly_report()
    else:
        add_entry()
        weekly_report()