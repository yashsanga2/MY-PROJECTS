"""
SIGN LANGUAGE → TEXT CONVERTER
Problem: Deaf and mute people struggle to communicate with those who don't know sign language.
Solution: Webcam tracks 21 hand landmarks per frame; a small classifier maps gesture → text.

Model: k-Nearest Neighbours (k-NN) on normalized MediaPipe hand landmarks.
- MediaPipe gives 21 (x, y, z) points per hand.
- We normalize them (translate wrist to origin, scale by hand size) so the
  gesture is invariant to where the hand is and how far it is from the camera.
- k-NN then classifies the landmark vector into a known sign.

This file has TWO parts:
  1) collect_data.py mode  — save landmarks for gestures you make (training data)
  2) main mode             — real-time recognition from webcam
"""

import csv
import json
import math
import os
from pathlib import Path

DATA_FILE = "sign_data.csv"
LABELS_FILE = "labels.json"


# ---------------------------------------------------------------
# Landmark normalization — the heart of making gestures recognizable
# ---------------------------------------------------------------
def normalize_landmarks(landmarks):
    """landmarks: list of 21 dicts {x, y, z} → rotation/scale-invariant features."""
    if not landmarks or len(landmarks) < 21:
        return None

    # 1. Translate: make wrist (point 0) the origin
    wrist = landmarks[0]
    pts = [(p["x"] - wrist["x"], p["y"] - wrist["y"], p["z"] - wrist["z"]) for p in landmarks]

    # 2. Scale: divide by distance wrist → middle-finger MCP (point 9)
    scale = math.dist(pts[0][:2], pts[9][:2]) or 1.0
    pts = [(x / scale, y / scale, z / scale) for x, y, z in pts]

    # 3. Rotate: make the wrist→MCP vector point along the +Y axis
    #    (undo hand tilt so the same gesture always looks the same)
    ax, ay = pts[9][0], pts[9][1]
    angle = math.atan2(ax, ay)
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    out = []
    for x, y, z in pts:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        out.extend([round(rx, 4), round(ry, 4), round(z, 4)])
    return out


# ---------------------------------------------------------------
# Training data collection (run with --collect GESTURE_NAME)
# ---------------------------------------------------------------
def collect(label: str):
    import mediapipe as mp
    import cv2
    import numpy as np

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    new_file = not os.path.exists(DATA_FILE)
    stored = 0
    print(f"Hold the '{label}' sign in front of the camera. Press 's' to save samples, 'q' to quit.")

    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([f"f{i}" for i in range(63)] + ["label"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm = [{"x": p.x, "y": p.y, "z": p.z} for p in result.multi_hand_landmarks[0].landmark]
                feats = normalize_landmarks(lm)
                if feats and cv2.waitKey(1) & 0xFF == ord("s"):
                    writer.writerow(feats + [label])
                    stored += 1
                    print(f"Saved sample #{stored} for '{label}'")

            cv2.imshow("Collect — s=save, q=quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {stored} samples saved for '{label}'.")


# ---------------------------------------------------------------
# k-NN classifier (pure Python — no sklearn needed)
# ---------------------------------------------------------------
def load_dataset():
    X, y = [], []
    if not os.path.exists(DATA_FILE):
        return X, y
    with open(DATA_FILE) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) == 64:
                X.append([float(v) for v in row[:63]])
                y.append(row[63])
    return X, y


def knn_predict(sample, X, y, k=5):
    if not X:
        return None, 0.0
    dists = sorted(((math.dist(sample, xi), yi) for xi, yi in zip(X, y)))
    votes = {}
    for _, label in dists[:k]:
        votes[label] = votes.get(label, 0) + 1
    best = max(votes, key=votes.get)
    return best, votes[best] / min(k, len(dists))


# ---------------------------------------------------------------
# Real-time recognition
# ---------------------------------------------------------------
def recognize():
    import mediapipe as mp
    import cv2

    X, y = load_dataset()
    if not X:
        print("No training data found. Run:  python main.py --collect hello")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    sentence = []

    print("Show a sign to the camera. 'c' clears text, 'q' quits.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            lm = [{"x": p.x, "y": p.y, "z": p.z} for p in result.multi_hand_landmarks[0].landmark]
            feats = normalize_landmarks(lm)
            if feats:
                label, conf = knn_predict(feats, X, y)
                if label and conf >= 0.6 and (not sentence or sentence[-1] != label):
                    sentence.append(label)

        cv2.putText(frame, " ".join(sentence), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Sign → Text", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--collect":
        collect(sys.argv[2])
    else:
        recognize()