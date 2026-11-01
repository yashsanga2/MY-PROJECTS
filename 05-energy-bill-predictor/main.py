"""
ENERGY BILL PREDICTOR
Problem: Families get shocked by electricity bills — they never know how much
next month will cost, so they can't budget or adjust usage before it's too late.

Solution: Enter this month's appliance usage → predicts your bill, shows which
appliance costs the most, and what you'd save by cutting usage 10%.

Model: Linear regression, implemented from scratch with gradient descent.
  bill ≈ w1·(fan hours) + w2·(light hours) + w3·(AC hours) + ... + bias
The model LEARNS each appliance's cost-per-hour from sample household data.
"""

import random

# ---------------------------------------------------------------------
# Dataset: (fan_hrs, light_hrs, ac_hrs, fridge_hrs, tv_hrs) → bill in ₹
# Made realistic using Indian tariff ~₹6.5/unit and typical appliance watts:
#   fan 75W, LED bulbs ~40W total, AC 1.5 ton ~1500W, fridge 200W, TV 100W
# ---------------------------------------------------------------------
APPLIANCES = ["fan", "lights", "AC", "fridge", "TV"]
WATTS = {"fan": 75, "lights": 40, "AC": 1500, "fridge": 200, "TV": 100}

def true_bill(usage: list[float], tariff: float = 6.5) -> float:
    units = sum(hrs * WATTS[a] / 1000 for a, hrs in zip(APPLIANCES, usage))
    return round(units * tariff + 40, 2)  # + fixed charge

def make_dataset(n: int = 300) -> tuple[list[list[float]], list[float]]:
    random.seed(42)
    X, y = [], []
    for _ in range(n):
        usage = [
            random.uniform(0, 12),    # fan hours/day
            random.uniform(1, 8),     # light hours/day
            random.uniform(0, 10),    # AC hours/day
            24.0,                    # fridge always on
            random.uniform(0, 6),    # TV hours/day
        ]
        X.append(usage)
        y.append(true_bill(usage))
    return X, y

# ---------------------------------------------------------------------
# Linear regression with gradient descent (from scratch)
# ---------------------------------------------------------------------
class LinearRegressionGD:
    def __init__(self, lr: float = 0.0005, epochs: int = 3000):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for xi, yi in zip(X, y):
                pred = sum(wi * xij for wi, xij in zip(self.w, xi)) + self.b
                err = pred - yi
                for j in range(d):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(d):
                self.w[j] -= self.lr * 2 * grad_w[j] / n
            self.b -= self.lr * 2 * grad_b / n
        return self

    def predict(self, x: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

    def score_r2(self, X, y) -> float:
        preds = [self.predict(xi) for xi in X]
        mean = sum(y) / len(y)
        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
        ss_tot = sum((yi - mean) ** 2 for yi in y)
        return 1 - ss_res / ss_tot

# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    X, y = make_dataset(300)
    split = 250
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = LinearRegressionGD().fit(X_train, y_train)
    print(f"Model R² on test data: {model.score_r2(X_test, y_test):.3f}\n")

    # Learned cost-per-hour per appliance
    print("💸 What the model LEARNED (₹/hour of daily use, 30-day month):")
    for name, w in zip(APPLIANCES, model.w):
        print(f"  {name:7s} → ₹{w * 1:.2f} per extra hour/day")
    print(f"  (fixed charge learned: ₹{model.b:.2f})\n")

    # Predict YOUR month
    my_usage = [8, 6, 4, 24, 3]   # fan, lights, AC, fridge, TV hours/day
    bill = model.predict(my_usage)
    print(f"🔮 Predicted monthly bill for {my_usage}: ₹{bill:.0f}")

    # Savings advice
    top = max(range(5), key=lambda j: model.w[j])
    cut = model.predict([u if j != top else u * 0.9 for j, u in enumerate(my_usage)])
    print(f"💡 Your biggest cost is {APPLIANCES[top].upper()}.")
    print(f"   Use it 10% less → save ₹{bill - cut:.0f}/month (₹{(bill - cut) * 12:.0f}/year!)")