"""
AI STUDY BUDDY — turns any text notes into a quiz + summary.
Problem: Students waste hours manually making revision notes and quizzes.
Model: TF-IDF (Term Frequency-Inverse Document Frequency) keyword extraction.
"""

import math
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------
# 1. Load raw study notes (plain text; convert PDFs to .txt first)
# ---------------------------------------------------------------
def load_notes(path: str) -> str:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Notes file not found: {path}")
    return file.read_text(encoding="utf-8")


# ---------------------------------------------------------------
# 2. Split text into sentences and "documents" (for TF-IDF)
# ---------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.split()) >= 4]


# ---------------------------------------------------------------
# 3. TF-IDF scoring — the ML/NLP core of this project
# ---------------------------------------------------------------
STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in",
    "on", "of", "to", "for", "with", "as", "by", "that", "this", "it",
    "from", "are", "was", "were", "be", "been", "has", "have", "had",
}

def tokenize(sentence: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", sentence.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

def tfidf_scores(sentences: list[str]) -> dict[str, float]:
    docs = [tokenize(s) for s in sentences]
    n_docs = len(docs) or 1

    # TF: word count within each sentence
    tfs = [Counter(d) for d in docs]

    # IDF: how rare a word is across all sentences
    df = Counter()
    for doc in docs:
        for word in set(doc):
            df[word] += 1

    scores: dict[str, float] = {}
    for word, freq in df.items():
        idf = math.log((n_docs + 1) / (freq + 1)) + 1
        # A word is important if rare across sentences but used a lot in one
        best_tf = max((tf[word] for tf in tfs if word in tf), default=0)
        scores[word] = best_tf * idf
    return scores


# ---------------------------------------------------------------
# 4. Summary: pick the top-scored sentences (extractive summarization)
# ---------------------------------------------------------------
def summarize(text: str, n: int = 3) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    word_scores = tfidf_scores(sentences)
    ranked = sorted(
        sentences,
        key=lambda s: sum(word_scores.get(w, 0) for w in tokenize(s)) / max(len(tokenize(s)), 1),
        reverse=True,
    )
    return ranked[:n]


# ---------------------------------------------------------------
# 5. Quiz generator: fill-in-the-blank questions from top keywords
# ---------------------------------------------------------------
def make_quiz(text: str, n_questions: int = 5) -> list[dict]:
    sentences = split_sentences(text)
    word_scores = tfidf_scores(sentences)
    top_words = [w for w, _ in sorted(word_scores.items(), key=lambda x: -x[1])]

    questions = []
    used = set()
    for word in top_words:
        for sentence in sentences:
            if word in used:
                break
            if re.search(rf"\b{re.escape(word)}\b", sentence, re.IGNORECASE) and sentence not in [q["full"] for q in questions]:
                blanked = re.sub(rf"\b{re.escape(word)}\b", "_____", sentence, flags=re.IGNORECASE)
                questions.append({"question": blanked, "answer": word, "full": sentence})
                used.add(word)
                break
        if len(questions) >= n_questions:
            break
    return questions


# ---------------------------------------------------------------
# 6. Demo runner
# ---------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    Photosynthesis is the process by which green plants convert sunlight into chemical energy.
    Chlorophyll inside the chloroplasts absorbs light most strongly in the blue and red regions.
    The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH.
    Carbon dioxide enters the leaf through tiny pores called stomata, which are mostly on the underside.
    The Calvin cycle uses the ATP and NADPH to fix carbon dioxide into glucose in the stroma.
    Plants that receive insufficient light grow tall and thin because they stretch toward light sources.
    """
    print("=" * 60)
    print("📖 SUMMARY (top sentences by TF-IDF)")
    print("=" * 60)
    for s in summarize(sample, 3):
        print(f"• {s.strip()}")

    print()
    print("=" * 60)
    print("📝 QUIZ (fill in the blanks)")
    print("=" * 60)
    quiz = make_quiz(sample, 5)
    for i, q in enumerate(quiz, 1):
        print(f"Q{i}. {q['question']}")
    print()
    for i, q in enumerate(quiz, 1):
        print(f"Q{i} answer: {q['answer']}")