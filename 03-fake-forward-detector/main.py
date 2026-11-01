"""
FAKE FORWARD DETECTOR
Problem: WhatsApp misinformation (fake forwards) spreads panic — health scares,
fake job offers, made-up "news". People have no quick way to check credibility.

Solution: Paste any forwarded message; this Naive Bayes classifier gives a
REAL / FAKE verdict with a confidence score and tells you which words triggered it.

Model: Multinomial Naive Bayes, implemented from scratch.
- Bayes' theorem:  P(fake | words) ∝ P(words | fake) × P(fake)
- "Naive" = we assume words are independent, which works surprisingly well for spam/fakes.
- Laplace smoothing avoids zero probabilities for unseen words.
"""

import math
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------
# Training dataset — real-ish fake & real message patterns (expand freely)
# ---------------------------------------------------------------------
TRAINING_DATA = [
    # ---- FAKE patterns -------------------------------------------------
    ("fake", "URGENT! Government is giving free 5000 rupees to everyone. Click this link now to claim before midnight! Forward to all your contacts immediately."),
    ("fake", "Forward this message to 10 people or your account will be deleted tonight. WhatsApp has confirmed this. Do not ignore!!!"),
    ("fake", "Doctors are hiding this miracle cure! Big pharma doesn't want you to know. Share this before it gets banned."),
    ("fake", "Congratulations!! You have won a brand new iPhone in our lucky draw. Send your bank details and OTP to claim your prize NOW."),
    ("fake", "BREAKING: This medicine cures coronavirus in 2 days. Government is suppressing this. Forward to save lives!!!"),
    ("fake", "New whatsapp rule: blue ticks mean government is reading your messages. Forward this to everyone to stay safe."),
    ("fake", "This message was sent to you because your number is lucky today. Send this to 15 groups and money will come to you in 24 hours."),
    ("fake", "Rare one rupee coin can be sold for 2 lakh rupees! Send your details on this link and get instant cash."),
    ("fake", "Warning: Do not answer calls from this number, it is a death virus that will kill your phone. Forward urgently!!!"),
    ("fake", "Free internet for 3 months from government. Just register with your Aadhaar number on this unknown site before it expires."),
    # ---- REAL patterns -------------------------------------------------
    ("real", "The examination scheduled for Monday has been postponed to Friday due to heavy rain. Please check the school notice board."),
    ("real", "Your electricity bill for this month is 1450 rupees. The due date is the 20th. Payment can be made online or at the office."),
    ("real", "Reminder: parent-teacher meeting on Saturday from 10 AM to 1 PM in the school auditorium. Attendance is requested."),
    ("real", "The office will remain closed tomorrow on account of a public holiday. Regular working hours resume the day after."),
    ("real", "Train number 12345 is running 30 minutes late today. Passengers are advised to check the official railway website."),
    ("real", "Our shop is open from 9 AM to 8 PM on weekdays. Home delivery is available for orders above 500 rupees."),
    ("real", "The library has added new books this month. Students can borrow up to 3 books at a time for two weeks."),
    ("real", "Please note the updated clinic timings: Monday to Saturday, 10 AM to 6 PM. Sunday consultations by appointment."),
    ("real", "All students must carry their ID cards to the examination hall. Entry without ID will not be permitted."),
    ("real", "The monthly staff meeting has been shifted to the first Tuesday of next month at the same time."),
]

# ---------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return words

# ---------------------------------------------------------------------
# Naive Bayes model (from scratch)
# ---------------------------------------------------------------------
class NaiveBayesTextClassifier:
    def fit(self, data):
        self.class_docs = defaultdict(list)     # label -> list of token lists
        self.class_doc_count = Counter()       # label -> number of documents
        self.total_docs = 0
        self.vocab = set()
        self.word_counts = {}                   # label -> Counter of words

        for label, text in data:
            tokens = tokenize(text)
            self.class_docs[label].append(tokens)
            self.class_doc_count[label] += 1
            self.total_docs += 1
            self.vocab.update(tokens)
            counts = self.word_counts.setdefault(label, Counter())
            counts.update(tokens)

        # total word count per class (for denominator of P(word|class))
        self.class_word_totals = {c: sum(cnt.values()) for c, cnt in self.word_counts.items()}
        return self

    def predict(self, text: str, k: int = 5):
        tokens = [t for t in tokenize(text) if t in self.vocab]
        if not tokens:
            return "unknown", 0.0, {}

        scores = {}
        evidence = {}  # per-word contribution (for explainability)
        V = len(self.vocab)
        for label, doc_count in self.class_doc_count.items():
            # log prior P(class)
            score = math.log(doc_count / self.total_docs)
            word_ev = {}
            denom = self.class_word_totals[label] + V
            for token in tokens:
                # Laplace-smoothed likelihood P(word | class)
                count = self.word_counts[label].get(token, 0)
                like = (count + 1) / denom
                score += math.log(like)
                word_ev[token] = math.log(like)
            scores[label] = score
            evidence[label] = word_ev

        best = max(scores, key=scores.get)
        # confidence via softmax over log-scores (numerically safe)
        max_score = scores[best]
        exp = {l: math.exp(s - max_score) for l, s in scores.items()}
        total = sum(exp.values())
        confidence = exp[best] / total

        # words that pushed the verdict most strongly
        fake_ev = evidence.get("fake", {})
        real_ev = evidence.get("real", {})
        triggers = {}
        for t in tokens:
            diff = fake_ev.get(t, 0) - real_ev.get(t, 0)
            triggers[t] = diff
        return best, confidence, triggers


# ---------------------------------------------------------------------
# Test suite — measure accuracy before shipping
# ---------------------------------------------------------------------
def evaluate(model, test_set):
    correct = 0
    for label, text in test_set:
        pred, conf, _ = model.predict(text)
        if pred == label:
            correct += 1
    return correct / len(test_set)

if __name__ == "__main__":
    # hold out 6 messages for testing (train on the rest)
    train, test = TRAINING_DATA[:14], TRAINING_DATA[14:]
    model = NaiveBayesTextClassifier().fit(train)

    acc = evaluate(model, test)
    print(f"Model accuracy on held-out test set: {acc:.0%}\n")

    samples = [
        "URGENT!! Forward this to 20 people and you will get free recharge tonight. WhatsApp confirmed!",
        "The office will remain closed tomorrow due to the bandh called by the traders association.",
        "Send your OTP and bank details to claim your lottery prize of 10 lakh rupees immediately!!!",
        "Students of class 10 are requested to bring their lab manuals to school tomorrow.",
    ]
    for msg in samples:
        verdict, conf, triggers = model.predict(msg)
        top_triggers = sorted(triggers.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        print(f"message: {msg[:70]}...")
        print(f"  verdict: {verdict.upper()}  (confidence {conf:.0%})")
        print(f"  key words: {', '.join(w for w, _ in top_triggers)}")
        print()