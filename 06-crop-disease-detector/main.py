"""
CROP DISEASE DETECTOR
Problem: Farmers lose 30-40% of crops to diseases they identify too late.
Agriculture extension officers can't visit every village in time.

Solution: Photograph an affected leaf → CNN classifies the disease and
suggests treatment, offline-friendly, built for smallholder farmers.

Model: Transfer learning with MobileNetV2 (pretrained on ImageNet).
- MobileNetV2 is tiny (~3.4 MB quantized) — runs on cheap Android phones.
- We freeze its convolutional layers (they already know edges, textures,
  spots, blight patterns) and train only a new classification head.
- Output classes use PlantVillage conventions (38 classes) — demo trains
  on a subset; instructions for the full dataset are in the README.

Two modes:
  python main.py --train          # fine-tune on your dataset
  python main.py --predict leaf.jpg
"""

import sys

CLASS_NAMES = [
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___Healthy", "Corn___Common_rust", "Corn___Healthy",
]

TREATMENT = {
    "Early_blight": "Remove infected leaves; spray mancozeb 2g/L every 7 days.",
    "Late_blight": "Act fast — remove and burn infected plants; spray metalaxyl + mancozeb.",
    "Leaf_Mold": "Improve airflow, reduce humidity; spray chlorothalonil.",
    "Common_rust": "Use resistant hybrids; spray sulfur-based fungicide.",
    "Healthy": "No disease detected. Keep monitoring weekly. 🌱",
}

# ----------------------------------------------------------------------
# Model definition (shared by train & predict)
# ----------------------------------------------------------------------
def build_model(num_classes: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze pretrained convolution layers

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ----------------------------------------------------------------------
# Training — expects dataset/disease_XXX/ folders of leaf images
# ----------------------------------------------------------------------
def train(dataset_dir: str = "dataset"):
    import tensorflow as tf
    print("Loading images from dataset/ (folder per class) ...")
    train_ds = keras.utils.image_dataset_from_directory(  # noqa
        dataset_dir, validation_split=0.2, subset="training",
        seed=42, image_size=(224, 224), batch_size=32, label_mode="int")
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="validation",
        seed=42, image_size=(224, 224), batch_size=32, label_mode="int")

    model = build_model(len(train_ds.class_names))
    class_names = train_ds.class_names

    # Early stopping + export best model
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("crop_model.keras", save_best_only=True),
    ]
    model.fit(train_ds.prefetch(tf.data.AUTOTUNE),
              validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
              epochs=10, callbacks=callbacks)

    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print(f"✅ Training done. Model saved to crop_model.keras ({len(class_names)} classes)")

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def predict(image_path: str, model_path: str = "crop_model.keras"):
    import numpy as np
    from tensorflow import keras

    try:
        names = open("class_names.txt").read().splitlines()
    except FileNotFoundError:
        names = CLASS_NAMES

    model = keras.models.load_model(model_path)
    img = keras.utils.load_img(image_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)[np.newaxis, ...]
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    label = names[idx] if idx < len(names) else CLASS_NAMES[idx]
    disease_key = label.split("___")[-1]

    print(f"\n🔬 Result for {image_path}")
    print(f"   Disease: {label}  (confidence {probs[idx]:.0%})")
    print(f"   💊 {TREATMENT.get(disease_key, 'Consult your local Krishi Vigyan Kendra.')}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    elif len(sys.argv) >= 2:
        predict(sys.argv[-1])
    else:
        print("Usage:\n  python main.py --train\n  python main.py leaf.jpg")