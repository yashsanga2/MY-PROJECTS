"""
TRAFFIC SIGN RECOGNIZER FOR INDIAN ROADS
Problem: New drivers, tourists, and driving-licence learners can't instantly
recognize every sign — Indian roads use 100+ sign types, and missing one
causes real accidents (wrong-way entries, missed speed limits, ignoring
give-way rules).

Solution: A CNN that classifies a photographed road sign and explains what the
driver must do — tuned for the Indian Motor Vehicles Act sign conventions.

Model: Small custom CNN (3 conv blocks) — trains on the German Traffic Sign
Recognition Benchmark (GTSRB, 43 classes, 50k images) whose signs closely match
Indian conventions (circular prohibitory, triangular warning, blue mandatory).

  python main.py --train       # trains and evaluates the CNN
  python main.py sign.jpg      # recognizes a sign image
"""

import sys

# What the driver must DO — this mapping is the real-world value-add
ACTION = {
    0:  ("Speed limit 20", "Slow down to 20 km/h."),
    1:  ("Speed limit 30", "Slow down to 30 km/h — residential zone."),
    2:  ("Speed limit 50", "Max 50 km/h."),
    3:  ("Speed limit 60", "Max 60 km/h."),
    4:  ("Speed limit 70", "Max 70 km/h."),
    5:  ("Speed limit 80", "Max 80 km/h."),
    6:  ("End of 80 limit", "Normal speed limits resume."),
    7:  ("Speed limit 100", "Max 100 km/h — highway."),
    8:  ("Speed limit 120", "Max 120 km/h."),
    9:  ("No overtaking", "Stay in lane; overtaking prohibited."),
    10: ("No trucks overtaking", "Trucks must not overtake."),
    11: ("Priority road", "You have right of way."),
    12: ("Priority road ahead", "Yield to oncoming traffic at junction."),
    13: ("Yield / Give way", "Slow down, let traffic on main road pass."),
    14: ("STOP", "Full stop; proceed only when clear."),
    15: ("No vehicles", "Entry prohibited."),
    16: ("No trucks >3.5t", "Heavy vehicles prohibited."),
    17: ("No entry", "Do not enter — one-way against you."),
    18: ("General caution", "Slow down, watch for hazards."),
    19: ("Dangerous curve left", "Reduce speed, sharp left ahead."),
    20: ("Dangerous curve right", "Reduce speed, sharp right ahead."),
    21: ("Double curve", "Slow down, winding road."),
    22: ("Bumpy road", "Slow down for rough surface."),
    23: ("Slippery road", "No sudden braking — road slippery."),
    24: ("Road narrows", "Merge carefully, road narrows on right."),
    25: ("Road work", "Slow down; workers on road."),
    26: ("Traffic signals ahead", "Signals ahead — prepare to stop."),
    27: ("Pedestrian crossing", "Slow down; yield to pedestrians."),
    28: ("Children crossing", "Extra caution — school zone."),
    29: ("Cyclists", "Watch for cyclists."),
    30: ("Ice/snow hazard", "Drive carefully in bad weather."),
    31: ("Wild animals", "Watch for animals crossing."),
    32: ("End of restrictions", "Previous limits cancelled."),
    33: ("Turn right ahead", "Mandatory right turn ahead."),
    34: ("Turn left ahead", "Mandatory left turn ahead."),
    35: ("Ahead only", "Go straight only."),
    36: ("Straight or right only", "Straight or right permitted."),
    37: ("Straight or left only", "Straight or left permitted."),
    38: ("Keep right", "Pass obstacle on right."),
    39: ("Keep left", "Pass obstacle on left."),
    40: ("Roundabout", "Enter roundabout; yield to traffic inside."),
    41: ("End of no-overtaking", "Overtaking allowed again."),
    42: ("End of truck-overtaking ban", "Trucks may overtake again."),
}

def build_cnn(num_classes: int = 43):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input((48, 48, 3)),
        # conv block 1
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        # conv block 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        # conv block 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        # head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train():
    from tensorflow import keras
    import numpy as np

    # GTSRB format: train_images.npy (float32, normalized, 48x48x3), train_labels.npy
    try:
        X = np.load("train_images.npy")
        y = np.load("train_labels.npy")
    except FileNotFoundError:
        print("Dataset not found. See README for the one-line GTSRB download.")
        return

    model = build_cnn()
    model.fit(X, y, validation_split=0.15, epochs=15, batch_size=64,
              callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    model.save("traffic_model.keras")
    print("✅ Model saved to traffic_model.keras")

def predict(image_path: str):
    import cv2
    import numpy as np
    from tensorflow import keras

    model = keras.models.load_model("traffic_model.keras")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48)).astype("float32") / 255
    probs = model.predict(img[None])[0]
    idx = int(np.argmax(probs))
    name, action = ACTION[idx]
    print(f"\n🚦 {name}  ({probs[idx]:.0%} confident)")
    print(f"   → {action}")

if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    elif len(sys.argv) >= 2:
        predict(sys.argv[-1])
    else:
        print("Usage:\n  python main.py --train\n  python main.py sign.jpg")