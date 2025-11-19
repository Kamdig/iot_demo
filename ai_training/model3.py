import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pathlib
import os
from PIL import Image

# === Paths ===
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "train"
MODEL_PATH = BASE_DIR / "model.keras"
TFLITE_QUANT_PATH = BASE_DIR / "model_int8.tflite"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

# === Parameters ===
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 15
VAL_SPLIT = 0.2
PATIENCE = 3
FINE_TUNE_AT = 200


def prepare_datasets():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    with open(CLASS_NAMES_PATH, "w") as f:
        f.write("\n".join(class_names))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.2),
    ])

    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        pooling="avg",
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_aug(inputs)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs), base_model


def train_model():
    train_ds, val_ds, class_names = prepare_datasets()
    model, base_model = build_model(len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    print("Training classification head...")
    model.compile(optimizer=keras.optimizers.AdamW(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

    print("Fine-tuning with some layers unfrozen...")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.AdamW(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    convert_to_tflite(model, train_ds)


def convert_to_tflite(model, train_ds):
    print("Converting to INT8 TFLite...")

    def rep_gen():
        for images, _ in train_ds.take(100):
            yield [images]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen

    # FLOAT input, INT8 weights — works everywhere
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    TFLITE_QUANT_PATH.write_bytes(tflite_model)

    print(f"Saved TFLite INT8 model: {TFLITE_QUANT_PATH}")


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Class names not found.")
    return [line.strip() for line in CLASS_NAMES_PATH.read_text().splitlines()]


