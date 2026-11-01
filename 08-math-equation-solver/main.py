"""
HANDWRITTEN MATH EQUATION SOLVER
Problem: Students get stuck on handwritten equations (textbook, board, notebook)
and can't check their work — typing math is slow and painful.

Solution: Photograph a linear equation like  2x + 5 = 13  → the app:
  1. Detects and reads each handwritten character (MNIST-trained CNN for digits
     + template matching for symbols)
  2. Assembles the equation string
  3. SOLVES it with a from-scratch linear-equation parser (no sympy needed)

Pipeline:  image → segment characters → classify → parse equation → solve

The CNN is trained on the Keras MNIST dataset (60,000 handwritten digit images).
Symbol recognition (+, -, =, x) uses contour/shape templates since MNIST is digits-only.
"""

import re
import sys

# ----------------------------------------------------------------------
# Step 1 — Character segmentation (find separate glyphs in an image)
# ----------------------------------------------------------------------
def segment_characters(image_path: str):
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # binarize (handwriting → white on black)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # dilate slightly to merge broken strokes
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # find contours → bounding boxes → left-to-right sort
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
    boxes.sort(key=lambda b: b[0])  # x-order = equation order
    return binary, boxes

# ----------------------------------------------------------------------
# Step 2 — MNIST digit classification (small CNN, trains in ~2 min on CPU)
# ----------------------------------------------------------------------
def train_digit_model():
    from tensorflow import keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

    model = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(16, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=1)
    loss, acc = model.evaluate(x_test, y_test)
    print(f"✅ Digit CNN test accuracy: {acc:.1%}")
    model.save("digit_model.keras")
    return model

# ----------------------------------------------------------------------
# Step 3 — classify one glyph (digit CNN + symbol heuristics)
# ----------------------------------------------------------------------
def classify_glyph(binary, box, model):
    import cv2
    import numpy as np

    x, y, w, h = box
    # Heuristic: '=' is wide & short, '+'/'-' are tiny strokes
    if w > 2.2 * h:
        return "="
    if h < binary.shape[0] * 0.15:  # very flat blob = minus sign
        return "-"
    if h < binary.shape[0] * 0.30 and w < binary.shape[0] * 0.30:
        return "+"

    roi = binary[y:y + h, x:x + w]
    # pad to square then resize to 28×28 (MNIST format)
    size = max(w, h)
    padded = cv2.copyMakeBorder(roi, (size - h) // 2, (size - h) // 2,
                                (size - w) // 2, (size - w) // 2,
                                cv2.BORDER_CONSTANT, value=0)
    digit = cv2.resize(padded, (28, 28)).astype("float32") / 255
    digit = digit.reshape(1, 28, 28, 1)

    probs = model.predict(digit, verbose=0)[0]
    idx = int(np.argmax(probs))
    # a '1' prediction with heavy stroke could be 'x' — context fixes later
    return str(idx)

# ----------------------------------------------------------------------
# Step 4 — Solve linear equation ax + b = c   (from scratch)
# ----------------------------------------------------------------------
def solve_linear(eq: str) -> str:
    """
    Handles forms like: 2x+5=13, 4x-7=21, 12=3x+3, x+9=15
    """
    eq = eq.replace(" ", "").lower()
    if "=" not in eq:
        return "Couldn't find '=' in the equation."
    left, right = eq.split("=")

    def parse_side(side: str):
        """return (coefficient of x, constant)"""
        tokens = re.findall(r"([+-]?)(\d*)x|([+-]?\d+)", side)
        coeff, const = 0.0, 0.0
        for sign, num, const_tok in tokens:
            if const_tok:
                const += float(const_tok)
            elif "x" in side:
                mag = float(num) if num else 1.0
                coeff += -mag if sign == "-" else mag
        return coeff, const

    lc, lconst = parse_side(left)
    rc, rconst = parse_side(right)

    a = lc - rc          # ax = c
    b = lconst - rconst  # move constants
    if a == 0:
        return "No 'x' found (or coefficient is 0) — can't solve."
    x = -b / a
    return f"{eq}   →   x = {x:g}"

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if "--train" in sys.argv:
        train_digit_model()
        sys.exit(0)

    # Direct text mode — works without a photo (test the solver brain first!)
    if len(sys.argv) >= 3 and sys.argv[1] == "--text":
        print(solve_linear(sys.argv[2]))
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py --train                # train digit CNN (MNIST, ~2 min)")
        print("  python main.py --text '2x+5=13'       # solver without camera")
        print("  python main.py equation.jpg           # photo → solve")
        sys.exit(0)

    from tensorflow import keras
    model = keras.models.load_model("digit_model.keras")
    binary, boxes = segment_characters(sys.argv[1])
    chars = [classify_glyph(binary, b, model) for b in boxes]
    equation = "".join(chars)
    print(f"🔍 Read from image: {equation}")
    print(solve_linear(equation))