"""
SYMPTOM CHECKER (educational)
Problem: People Google symptoms, get terrified by worst-case results, and either
panic or ignore real warning signs. Self-diagnosis is broken.

Solution: Enter your symptoms → get the most likely common conditions with
matched symptoms and confidence, PLUS clear "see a doctor now" red-flag alerts.
Explicitly NOT a replacement for a doctor — it triages, it doesn't diagnose.

Model: Multi-label probabilistic scoring (Bayesian, from scratch):
  P(disease | symptoms) ∝ P(symptom) across that disease's profile × prior
Each disease has a symptom-profile with weights; input symptoms are matched
and scored with normalization. Red-flag symptoms short-circuit to URGENT.

Ethics built in: every output ends with a doctor disclaimer. Good AI knows
its limits — that's a feature universities love to hear about.
"""

# ---------------------------------------------------------------------
# Medical knowledge base (illustrative — simplified from public symptom
# databases; extend carefully and never use alone for real decisions)
# ---------------------------------------------------------------------
DISEASES = {
    "Common Cold": {
        "symptoms": {"runny nose": 3, "sneezing": 3, "sore throat": 2, "cough": 2,
                     "mild fever": 1, "headache": 1, "body ache": 1},
        "red_flags": [],
    },
    "Influenza (Flu)": {
        "symptoms": {"high fever": 3, "body ache": 3, "chills": 3, "fatigue": 3,
                     "cough": 2, "headache": 2, "sore throat": 1},
        "red_flags": [],
    },
    "Viral Fever (Dengue-like)": {
        "symptoms": {"high fever": 3, "headache": 3, "body ache": 3, "rash": 2,
                     "nausea": 2, "fatigue": 2, "eye pain": 2},
        "red_flags": ["bleeding gums", "severe abdominal pain"],
    },
    "Gastroenteritis": {
        "symptoms": {"vomiting": 3, "diarrhea": 3, "stomach pain": 2, "nausea": 2,
                     "mild fever": 1, "fatigue": 1},
        "red_flags": ["blood in stool"],
    },
    "Migraine": {
        "symptoms": {"severe headache": 3, "nausea": 2, "light sensitivity": 3,
                     "vomiting": 1, "blurred vision": 1},
        "red_flags": [],
    },
    "Anemia": {
        "symptoms": {"fatigue": 3, "pale skin": 3, "dizziness": 2, "shortness of breath": 2,
                     "headache": 1, "cold hands": 1},
        "red_flags": [],
    },
    "Typhoid": {
        "symptoms": {"high fever": 3, "stomach pain": 2, "headache": 2, "fatigue": 3,
                     "loss of appetite": 2, "constipation": 1, "rash": 1},
        "red_flags": [],
    },
}

RED_FLAG_SYMPTOMS = {
    "chest pain": "Possible cardiac emergency — call 108 (ambulance) NOW.",
    "difficulty breathing": "Breathing trouble — seek emergency care NOW.",
    "severe bleeding": "Uncontrolled bleeding — emergency care NOW.",
    "blood in stool": "Needs urgent medical evaluation today.",
    "bleeding gums": "With fever → suspected dengue — get tested today.",
    "severe abdominal pain": "Possible appendicitis/ulcer — see a doctor today.",
    "unconsciousness": "Call 108 immediately.",
}

# ---------------------------------------------------------------------
def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())

def check_symptoms(user_input: list[str]) -> None:
    symptoms = [normalize(s) for s in user_input]

    # 1) Red-flag short-circuit — safety first, always
    emergencies = [(s, RED_FLAG_SYMPTOMS[s]) for s in symptoms if s in RED_FLAG_SYMPTOMS]
    for disease, profile in DISEASES.items():
        for flag in profile.get("red_flags", []):
            if flag in symptoms:
                emergencies.append((flag, f"Warning sign of {disease} — see a doctor today."))

    # 2) Bayesian-style scoring
    results = []
    for disease, profile in DISEASES.items():
        weights = profile["symptoms"]
        matched = [s for s in symptoms if s in weights]
        if not matched:
            continue
        score = sum(weights[s] for s in matched) / sum(weights.values())
        results.append((disease, score, matched))

    results.sort(key=lambda r: r[1], reverse=True)
    top = results[:3]

    # 3) Output
    print("\n🩺 Based on:", ", ".join(symptoms))
    if emergencies:
        print("\n🚨 URGENT — these symptoms need a doctor, not an app:")
        for s, advice in emergencies:
            print(f"   • {s}: {advice}")

    if top:
        print("\n💡 Most likely common conditions (educational guesses only):")
        total = sum(r[1] for r in top)
        for disease, score, matched in top:
            share = score / total if total else 0
            print(f"   {disease:28s} {share:.0%} — matched: {', '.join(matched)}")
    else:
        print("\n🤔 No strong match. If symptoms persist more than 2 days, see a doctor.")

    print("\n⚕️  This tool is for awareness only. It is NOT a diagnosis.")
    print("    When in doubt, visit a real doctor. Health > anxiety.")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Symptom Checker (educational triage) ===")
    print("Enter your symptoms separated by commas (e.g. fever, headache, cough):")
    raw = input("> ")
    check_symptoms([s.strip() for s in raw.split(",") if s.strip()])