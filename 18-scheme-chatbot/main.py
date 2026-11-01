"""
GOVERNMENT SCHEME ELIGIBILITY CHATBOT
Problem: India runs 3000+ welfare schemes (scholarships, pensions, housing,
LPG subsidy...) but crores of eligible citizens never apply — because they
don't know the schemes exist, and rules are buried in PDFs. The benefits go
unclaimed; the people who need them most lose out.

Solution: A chatbot that asks a few simple questions (age, income, student?,
state...) then lists EXACTLY the schemes you qualify for, with the documents
needed and where to apply. Plain language, no jargon.

Model: Intent classification (TF-IDF + cosine similarity, from scratch —
the same core math behind chat engines) + a structured eligibility-rules
engine. NLP understands the question; rules decide the answer.

  python main.py            # interactive chat
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------------
# Knowledge base: real schemes, simplified eligibility rules
# (Always verify current criteria on the official portal — rules change)
# ---------------------------------------------------------------------
SCHEMES = [
    {
        "name": "PM-KISAN (farmers' income support)",
        "match": ["farmer", "kisan", "agriculture", "farming"],
        "eligible": lambda p: p.get("occupation") == "farmer",
        "benefit": "₹6,000/year direct to bank account",
        "docs": "Land records, Aadhaar, bank passbook",
        "apply": "pmkisan.gov.in or your CSC centre",
    },
    {
        "name": "National Means-cum-Merit Scholarship (NMMS)",
        "match": ["scholarship", "student", "school", "study", "education"],
        "eligible": lambda p: p.get("student") and p.get("income", 1e9) < 350000
                              and 8 <= p.get("class", 0) <= 10,
        "benefit": "₹12,000/year scholarship for Class 9-12",
        "docs": "Income certificate, marksheet, school enrolment",
        "apply": "scholarships.gov.in",
    },
    {
        "name": "Sukanya Samriddhi Yojana (girl child savings)",
        "match": ["daughter", "girl", "girl child", "savings"],
        "eligible": lambda p: p.get("has_daughter") and p.get("daughter_age", 99) <= 10,
        "benefit": "High-interest savings account (8%+), tax-free",
        "docs": "Daughter's birth certificate, guardian ID",
        "apply": "Any post office or bank",
    },
    {
        "name": "Pradhan Mantri Awas Yojana (housing)",
        "match": ["house", "home", "housing", "awas"],
        "eligible": lambda p: p.get("income", 1e9) < 180000
                              and not p.get("owns_house", True),
        "benefit": "Subsidy up to ₹2.5 lakh on home loan",
        "docs": "Income certificate, Aadhaar, residence proof",
        "apply": "pmaymis.gov.in / your municipal office",
    },
    {
        "name": "Ayushman Bharat (PM-JAY health cover)",
        "match": ["health", "hospital", "medical", "insurance", "treatment"],
        "eligible": lambda p: p.get("income", 1e9) < 250000
                              and p.get("rural_or_bpl", False),
        "benefit": "₹5 lakh/year free hospital cover for family",
        "docs": "Aadhaar, ration card / SECC listing",
        "apply": "Check name at beneficiary.nha.gov.in or CSC",
    },
    {
        "name": "Atal Pension Yojana",
        "match": ["pension", "retirement", "old age"],
        "eligible": lambda p: 18 <= p.get("age", 0) <= 40 and p.get("income", 1e9) < 1500000,
        "benefit": "Guaranteed ₹1,000-5,000/month pension after 60",
        "docs": "Bank account, Aadhaar",
        "apply": "Your bank branch",
    },
]

# ---------------------------------------------------------------------
# NLP: intent matching via TF-IDF + cosine similarity (from scratch)
# ---------------------------------------------------------------------
def tokenize(t: str) -> list:
    return re.findall(r"[a-z]+", t.lower())

def build_vectors(query: str, docs: list):
    all_docs = docs + [tokenize(query)]
    vocab = sorted({w for d in all_docs for w in d})
    index = {w: i for i, w in enumerate(vocab)}

    def tfidf(doc: list) -> list:
        counts = Counter(doc)
        vec = [0.0] * len(vocab)
        for word, c in counts.items():
            n_docs = sum(1 for d in all_docs if word in d) or 1
            idf = math.log(len(all_docs) / n_docs) + 1
            vec[index[word]] = (c / len(doc)) * idf
        return vec

    def cosine(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1
        nb = math.sqrt(sum(y * y for y in b)) or 1
        return dot / (na * nb)

    return tfidf, cosine

def understand_question(query: str) -> str:
    """Map a free-text question to the closest scheme topic."""
    doc_tokens = [tokenize(" ".join(s["match"])) for s in SCHEMES]
    tfidf, cosine = build_vectors(query, doc_tokens)
    q_vec = tfidf(tokenize(query))
    scores = [cosine(tfidf(d), q_vec) for d in doc_tokens]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return SCHEMES[best]["name"] if scores[best] > 0.05 else ""

# ---------------------------------------------------------------------
# Profile collection → eligibility engine
# ---------------------------------------------------------------------
def collect_profile() -> dict:
    print("🤖 Namaste! I'll ask a few quick questions to find schemes for you.\n")
    p = {}
    p["age"] = int(input("1. Your age: ") or 0)
    income = input("2. Yearly family income in ₹ (approx, e.g. 200000): ")
    p["income"] = int(re.sub(r"[^0-9]", "", income) or "999999999")
    p["student"] = input("3. Are you a school student? (y/n): ").strip().lower() == "y"
    if p["student"]:
        p["class"] = int(input("   Which class/grade? (6-12): ") or 0)
    p["occupation"] = input("4. Occupation (farmer/shopkeeper/job/student/etc): ").strip().lower()
    p["has_daughter"] = input("5. Do you have a daughter under 10? (y/n): ").strip().lower() == "y"
    p["owns_house"] = input("6. Do you own a house? (y/n): ").strip().lower() == "y"
    p["rural_or_bpl"] = input("7. Rural area or BPL card? (y/n): ").strip().lower() == "y"
    return p

def find_schemes(profile: dict) -> list:
    matched = []
    for s in SCHEMES:
        try:
            if s["eligible"](profile):
                matched.append(s)
        except Exception:
            continue  # profile missing a field this scheme needs
    return matched

# ---------------------------------------------------------------------
if __name__ == "__main__":
    # NLP demo first: show the model understands free-text questions
    for q in ["I need scholarship money for school",
              "my father is a farmer",
              "hospital bill help scheme",
              "how do I get pension"]:
        topic = understand_question(q)
        print('🗣️ "%s"  → understood topic: %s' % (q, topic or "(not sure)"))
    print()

    profile = collect_profile()
    matched = find_schemes(profile)

    print("\n📋 YOU QUALIFY FOR:")
    if not matched:
        print("   No schemes matched — but check myScheme portal (myscheme.gov.in)")
        print("   which matches citizens to 3000+ central & state schemes.")
    for s in matched:
        print("\n   ✅ %s" % s["name"])
        print("      Benefit : %s" % s["benefit"])
        print("      Documents: %s" % s["docs"])
        print("      Apply at : %s" % s["apply"])
    print("\n⚠️ Rules change — always confirm on the official portal before applying.")