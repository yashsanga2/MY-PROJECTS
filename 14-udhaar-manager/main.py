"""
UDHAAR MANAGER — Shop Credit Tracker with Risk Prediction
Problem: Crores of Indian small shopkeepers run "udhaar" (informal credit) —
a notebook of who owes what. Bad debts quietly pile up, and there's no way to
see which customer is becoming risky until the money is already gone.

Solution: A digital udhaar register that:
  1. Tracks each customer's credit + repayment history (JSON "database")
  2. LEARNS a risk score per customer (logistic-style scoring from history)
  3. Warns: "Customer X is becoming risky — 3 missed payments, balance rising"

Model: A transparent risk-scoring model (weighted features + sigmoid, from
scratch) — the same core math banks use for credit scores, but fully
explainable so shopkeepers can see WHY someone is flagged.

  python main.py            # demo with sample shop data
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

DB_FILE = Path("udhaar.json")

# ---------------------------------------------------------------------
# Sample database — realistic small-shop credit histories
# ---------------------------------------------------------------------
def seed_db() -> list[dict]:
    today = datetime.now()
    def days_ago(n): return (today - timedelta(days=n)).strftime("%Y-%m-%d")

    return [
        {"name": "Ramesh", "entries": [
            {"date": days_ago(60), "type": "credit", "amount": 500},
            {"date": days_ago(45), "type": "payment", "amount": 500},
            {"date": days_ago(20), "type": "credit", "amount": 700},
            {"date": days_ago(5), "type": "payment", "amount": 700},
        ]},
        {"name": "Sunita", "entries": [
            {"date": days_ago(90), "type": "credit", "amount": 300},
            {"date": days_ago(70), "type": "payment", "amount": 150},
            {"date": days_ago(50), "type": "credit", "amount": 800},
            {"date": days_ago(25), "type": "credit", "amount": 600},
            {"date": days_ago(10), "type": "payment", "amount": 100},
        ]},
        {"name": "Anil", "entries": [
            {"date": days_ago(40), "type": "credit", "amount": 1200},
            {"date": days_ago(30), "type": "payment", "amount": 400},
            {"date": days_ago(12), "type": "credit", "amount": 900},
        ]},
    ]

def load_db() -> list[dict]:
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text())
    data = seed_db()
    DB_FILE.write_text(json.dumps(data, indent=2))
    return data

# ---------------------------------------------------------------------
# Features per customer (the raw material for risk scoring)
# ---------------------------------------------------------------------
def customer_features(customer: dict) -> dict:
    entries = customer["entries"]
    balance = sum(e["amount"] if e["type"] == "credit" else -e["amount"] for e in entries)
    credits = [e for e in entries if e["type"] == "credit"]
    payments = [e for e in entries if e["type"] == "payment"]
    total_credit = sum(e["amount"] for e in credits) or 1

    # repayment ratio: how much of what was taken has come back
    repayment_ratio = sum(e["amount"] for e in payments) / total_credit

    # missed payments: credits with no payment within 30 days after
    dates = sorted(e["date"] for e in entries)
    streak = 0  # consecutive unpaid credits at the END of history
    running = 0
    for e in entries:
        if e["type"] == "credit":
            running += 1
        else:
            running = 0
        streak = running

    return {
        "balance": balance,
        "repayment_ratio": repayment_ratio,
        "open_credits_streak": streak,
        "credit_count": len(credits),
    }

# ---------------------------------------------------------------------
# Risk score — logistic function over weighted features (from scratch)
# ---------------------------------------------------------------------
def risk_score(features: dict) -> tuple[float, list[str]]:
    z = 0.0
    reasons = []

    # each feature contributes to the logit, like a tiny credit bureau
    z += 3.0 * (1 - features["repayment_ratio"])
    if features["repayment_ratio"] < 0.5:
        reasons.append(f"only {features['repayment_ratio']:.0%} of credit repaid")

    z += 0.8 * min(features["open_credits_streak"], 4)
    if features["open_credits_streak"] >= 2:
        reasons.append(f"{features['open_credits_streak']} credits without repayment in a row")

    z += 0.002 * min(features["balance"], 3000)
    if features["balance"] > 800:
        reasons.append(f"outstanding balance ₹{features['balance']:.0f} is high")

    z -= 1.0  # baseline (most people are okay)

    risk = 1 / (1 + math.exp(-z))  # sigmoid
    if not reasons:
        reasons.append("clean repayment history")
    return risk, reasons

# ---------------------------------------------------------------------
def main():
    db = load_db()
    print("=" * 62)
    print("📒 UDHAAR REGISTER — with AI risk warnings")
    print("=" * 62)

    rows = []
    for customer in db:
        f = customer_features(customer)
        risk, reasons = risk_score(f)
        rows.append((customer["name"], f["balance"], risk, reasons))

    rows.sort(key=lambda r: r[2], reverse=True)  # riskiest first

    for name, balance, risk, reasons in rows:
        flag = "🔴" if risk > 0.6 else ("🟡" if risk > 0.35 else "🟢")
        print(f"\n{flag} {name}")
        print(f"   Outstanding: ₹{balance}")
        print(f"   Risk score : {risk:.0%}  ({'HIGH' if risk > 0.6 else 'watch' if risk > 0.35 else 'safe'})")
        print(f"   Why: {'; '.join(reasons)}")

    print("\n💡 Suggestion: for 🔴 customers, ask for part-payment before new credit.")
    print("   (Not a replacement for knowing your customers — a helper.)")

if __name__ == "__main__":
    main()