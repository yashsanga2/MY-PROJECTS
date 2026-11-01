"""
STUDENT DROPOUT RISK PREDICTOR
Problem: Students don't drop out suddenly — they fade: attendance slips, grades
slide, then they're gone. Schools notice only when the seat is empty, and by
then intervention is too late.

Solution: A model that flags at-risk students EARLY from attendance, grades,
and homework patterns — with the reasons attached, so a teacher knows HOW to
help (attendance problem vs. academic struggle need different interventions).

Model: Decision tree implemented from scratch (information gain / entropy),
chosen deliberately over a black box: teachers trust rules they can read,
like "attendance < 75% AND homework rate < 60% → at risk".

Trains on a realistic synthetic dataset (patterned after real dropout research
factors) and evaluates honestly with a held-out test split.

  python main.py
"""

import random
from collections import Counter

# ---------------------------------------------------------------------
# 1. Synthetic dataset — realistic patterns from dropout research
# ---------------------------------------------------------------------
FEATURES = ["attendance_pct", "avg_grade", "homework_rate", "absent_streak"]

def make_dataset(n: int = 400, seed: int = 7):
    random.seed(seed)
    X, y = [], []
    for _ in range(n):
        attendance = random.uniform(40, 100)
        grade = random.uniform(25, 95)
        homework = random.uniform(10, 100)
        streak = random.randint(0, 12)

        # dropout risk rises with: low attendance, low grades, low homework, long streak
        risk = (75 - attendance) * 0.05 + (55 - grade) * 0.04 + \
               (60 - homework) * 0.02 + streak * 0.09
        label = 1 if risk + random.gauss(0, 0.4) > 0.9 else 0
        X.append([round(attendance, 1), round(grade, 1), round(homework, 1), streak])
        y.append(label)
    return X, y

# ---------------------------------------------------------------------
# 2. Decision tree from scratch (entropy + information gain)
# ---------------------------------------------------------------------
def entropy(labels: list[int]) -> float:
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * __import__("math").log2(c / total) for c in counts.values())

def split_gain(X, y, feature_i, threshold):
    left_y = [yi for xi, yi in zip(X, y) if xi[feature_i] <= threshold]
    right_y = [yi for xi, yi in zip(X, y) if xi[feature_i] > threshold]
    if not left_y or not right_y:
        return 0.0
    parent = entropy(y)
    child = (len(left_y) * entropy(left_y) + len(right_y) * entropy(right_y)) / len(y)
    return parent - child

def best_split(X, y):
    best = (0, None, 0.0)  # feature, threshold, gain
    for f in range(len(FEATURES)):
        values = sorted({xi[f] for xi in X})
        for i in range(len(values) - 1):
            t = (values[i] + values[i + 1]) / 2
            g = split_gain(X, y, f, t)
            if g > best[2]:
                best = (f, t, g)
    return best

def build_tree(X, y, depth: int = 0, max_depth: int = 4):
    if depth >= max_depth or len(set(y)) == 1 or len(y) < 8:
        majority = Counter(y).most_common(1)[0][0]
        return {"leaf": majority}

    f, t, gain = best_split(X, y)
    if gain < 0.01:
        return {"leaf": Counter(y).most_common(1)[0][0]}

    left_idx = [i for i in range(len(X)) if X[i][f] <= t]
    right_idx = [i for i in range(len(X)) if X[i][f] > t]
    return {
        "feature": FEATURES[f],
        "threshold": t,
        "left": build_tree([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1, max_depth),
        "right": build_tree([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1, max_depth),
    }

def predict_one(tree, student: dict):
    node = tree
    while "leaf" not in node:
        go_left = student[node["feature"]] <= node["threshold"]
        node = node["left"] if go_left else node["right"]
    return node["leaf"]

def explain(tree, student: dict) -> list[str]:
    """Walk the tree and collect the rule path in plain English."""
    node, path = tree, []
    while "leaf" not in node:
        cond = student[node["feature"]] <= node["threshold"]
        path.append(f"{node['feature']} {'≤' if cond else '>'} {node['threshold']:.1f}")
        node = node["left"] if cond else node["right"]
    return path

# ---------------------------------------------------------------------
# 3. Train + evaluate + demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    X, y = make_dataset(400)
    split = 300
    X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

    tree = build_tree(X_train, y_train)

    preds = [predict_one(tree, dict(zip(FEATURES, xi))) for xi in X_test]
    acc = sum(p == t for p, t in zip(preds, y_test)) / len(y_test)
    print(f"🌳 Decision-tree accuracy on held-out test data: {acc:.0%}\n")

    print("📖 The model's readable rules (this is why a tree, not a neural net):")
    def show(node, indent=""):
        if "leaf" in node:
            print(f"{indent}→ {'⚠️ AT RISK' if node['leaf'] else '✅ okay'}")
            return
        print(f"{indent}if {node['feature']} ≤ {node['threshold']:.1f}:")
        show(node["left"], indent + "  ")
        print(f"{indent}else ({node['feature']} > {node['threshold']:.1f}):")
        show(node["right"], indent + "  ")
    show(tree)

    print("\n🔍 Screening 3 new students:")
    students = [
        {"attendance_pct": 82, "avg_grade": 61, "homework_rate": 70, "absent_streak": 2},
        {"attendance_pct": 58, "avg_grade": 42, "homework_rate": 30, "absent_streak": 7},
        {"attendance_pct": 91, "avg_grade": 78, "homework_rate": 85, "absent_streak": 1},
    ]
    for s in students:
        label = predict_one(tree, s)
        print(f"  {s} → {'⚠️ FLAG for counselling' if label else '✅ low risk'}")
        if label:
            print("     reason path:", " AND ".join(explain(tree, s)))