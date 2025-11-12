import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pathlib
import os
from PIL import Image
from typing import cast

# === Paths ===
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "train"
MODEL_PATH = BASE_DIR / "model.keras"
TFLITE_PATH = BASE_DIR / "model.tflite"
TFLITE_QUANT_PATH = BASE_DIR / "model_int8.tflite"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

# === Parameters ===
IMG_SIZE = (160, 160)     # smaller input = faster training/inference
BATCH_SIZE = 16
EPOCHS = 15
VAL_SPLIT = 0.2
PATIENCE = 3
FINE_TUNE_AT = 200  # number of layers to keep frozen when fine-tuning


def prepare_datasets():
    # Assemble cached train/validation datasets from the local image folder.
    """Load dataset and split into training and validation sets."""
    # Refuse to train when the dataset path is missing.
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
    print(f"📂 Classes found: {class_names}")

    # Save class names for later inference
    with open(CLASS_NAMES_PATH, "w") as f:
        f.write("\n".join(class_names))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.2),
    ])

    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        pooling="avg",
        weights="imagenet"
    )
    base_model.trainable = False  # freeze pretrained weights

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    return model, base_model


def train_model():
    # Full training pipeline plus TensorFlow Lite export/quantization steps.
    """Train and export TensorFlow + TFLite models with transfer learning."""
    train_ds, val_ds, class_names = prepare_datasets()
    num_classes = len(class_names)

    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH.with_suffix(".h5")), save_best_only=True, save_format="h5"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    print("🧠 Training top layers...")
    model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

    # === Fine-tuning ===
    print("🔓 Unfreezing base model for fine-tuning...")
    base_model.trainable = True
    # Preserve most pretrained layers during fine-tuning to avoid overfitting.
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    print(f"✅ Saved trained model to {MODEL_PATH}")

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    print(f"✅ Saved trained model to {MODEL_PATH}")

    # === Convert to TensorFlow Lite with full int8 quantization ===
    def representative_dataset_gen():
        for images, _ in train_ds.take(100):
            yield [images]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_quant_model = converter.convert()
    with open(str(TFLITE_QUANT_PATH), "wb") as f:
        f.write(tflite_quant_model)

    print(f"✅ Quantized int8 model saved as {TFLITE_QUANT_PATH}")





# Read the saved class names list from disk for inference.
def load_class_names():
    """Load saved class names."""
    # Ensure the label file exists before attempting to read it.
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Class names file not found. Train the model first.")
    with open(CLASS_NAMES_PATH) as f:
        return [line.strip() for line in f.readlines()]


# Classify a single image with either the Keras or TFLite model.
def predict_image(image_path: str, use_tflite: bool = False):
    """Predict image class using either TF model or TFLite."""
    # Validate that the image path refers to an existing file.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    class_names = load_class_names()
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Choose the inference runtime based on the CLI flag.
    if use_tflite:
        # === Run inference with TensorFlow Lite ===
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_QUANT_PATH))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"])[0]
    else:
        # === Run inference with TensorFlow ===
        # === Run inference with TensorFlow ===
        model = keras.models.load_model(MODEL_PATH)
        prediction = model.predict(img_array, verbose=0)[0]

    top_idx = np.argmax(prediction)
    confidence = prediction[top_idx] * 100
    print(f"🖼️ Predicted class: {class_names[top_idx]} ({confidence:.2f}%)")

    print("Class probabilities:")
    # Enumerate each class probability for debugging visibility.
    for name, prob in zip(class_names, prediction):
        print(f"  {name}: {prob * 100:.2f}%")


if __name__ == "__main__":
    # Run inference when an image path argument is supplied, otherwise train.
    import sys
    # Switch to prediction mode whenever an image path argument is provided.
    if len(sys.argv) > 1:
        predict_image(sys.argv[1], use_tflite=False)
    else:
        train_model()
