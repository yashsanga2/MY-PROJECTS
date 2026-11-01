"""
WASTE CLASSIFIER — "Which bin?"
Problem: Waste segregation fails at the household level. People genuinely
don't know if a pizza box, a milk packet, or a broken toy is recyclable,
so everything goes into one bin and recycling fails downstream.

Solution: Photo of any item → (1) classify material category with a CNN,
(2) apply local rules → tell you the EXACT bin: wet / dry / recyclable /
hazardous / e-waste, plus why.

Model: Transfer learning on MobileNetV2 trained with Kaggle's TrashNet or
Garbage Classification dataset (cardboard, glass, metal, paper, plastic,
trash, + optionally battery/e-waste classes).

The intelligence is two-layered:
  Layer 1 (ML):  photo → material class        (learned)
  Layer 2 (rules): material → bin + why       (expert knowledge, Indian SWM Rules 2016)

  python main.py --train
  python main.py item.jpg
"""

import sys

# ---------------------------------------------------------------------
# Layer 2: expert rules — material → bin (India Solid Waste Management
# Rules 2016 categories). Edit per your city's colour-coded bins.
# ---------------------------------------------------------------------
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash",
               "battery", "e_waste"]

BIN_RULES = {
    "cardboard": ("♻️ Recyclable bin (BLUE)", "Clean cardboard — flatten it to save space."),
    "glass":     ("♻️ Recyclable bin (BLUE)", "Glass is 100% recyclable. Don't mix broken glass loosely — wrap it."),
    "metal":     ("♻️ Recyclable bin (BLUE)", "Metals recycle forever — cans, foil, tins all qualify."),
    "paper":     ("♻️ Recyclable bin (BLUE)", "Keep paper dry; wet paper goes to compost instead."),
    "plastic":    ("♻️ Recyclable bin (BLUE) — CHECK code", "Look for recycling code 1,2,5 (recyclable). Code 3,6,7 often rejected in India."),
    "trash":     ("🗑️ Reject bin (BLACK/other)", "Non-recyclable mixed waste — landfill. Avoid buying these!"),
    "battery":   ("⚠️ Hazardous / e-waste drop-off", "Batteries NEVER go in regular bins — take to a collection point (e.g. a local e-waste drive)."),
    "e_waste":   ("⚠️ E-waste collection", "Electronics have valuable + toxic parts. Find your nearest authorised e-waste recycler."),
}

# ---------------------------------------------------------------------
def build_model(num_classes: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    base = keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                          include_top=False, weights="imagenet")
    base.trainable = False

    model = keras.Sequential([
        layers.Input((224, 224, 3)),
        keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)]),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train(dataset_dir: str = "dataset"):
    import tensorflow as tf
    from tensorflow import keras

    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="training", seed=42,
        image_size=(224, 224), batch_size=32, label_mode="int")
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=(224, 224), batch_size=32, label_mode="int")

    model = build_model(len(train_ds.class_names))
    model.fit(train_ds.prefetch(tf.data.AUTOTUNE),
              validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
              epochs=10,
              callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                         keras.callbacks.ModelCheckpoint("waste_model.keras", save_best_only=True)])
    open("class_names.txt", "w").write("\n".join(train_ds.class_names))
    print("✅ Saved waste_model.keras")

def classify(image_path: str):
    import numpy as np
    from tensorflow import keras

    try:
        names = open("class_names.txt").read().splitlines()
    except FileNotFoundError:
        names = CLASS_NAMES

    model = keras.models.load_model("waste_model.keras")
    img = keras.utils.load_img(image_path, target_size=(224, 224))
    x = keras.applications.mobilenet_v2.preprocess_input(
        np.asarray(keras.utils.img_to_array(img))[None, ...])

    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    material = names[idx] if idx < len(names) else CLASS_NAMES[idx]
    bin_name, why = BIN_RULES.get(material, BIN_RULES["trash"])

    print(f"\n🗑️ Item: {image_path}")
    print(f"   Material: {material} ({probs[idx]:.0%} confident)")
    print(f"   → {bin_name}")
    print(f"     {why}")

if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    elif len(sys.argv) >= 2:
        classify(sys.argv[-1])
    else:
        print("Usage:\n  python main.py --train\n  python main.py item.jpg")