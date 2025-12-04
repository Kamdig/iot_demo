from keras import layers
import tensorflow as tf
import pathlib
import keras

# ==========================================
# PATHS
# ==========================================
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "train"
MODEL_PATH = BASE_DIR / "model.keras"
TFLITE_QUANT_PATH = BASE_DIR / "model_int8.tflite"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

# ==========================================
# PARAMETERS
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 15
VAL_SPLIT = 0.2
PATIENCE = 4


# ==========================================
# DATA PIPELINE
# ==========================================
def prepare_datasets():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = keras.utils.image_dataset_from_directory(
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

    # Show class counts
    print("Dataset class counts:")
    for cname in class_names:
        count = len(list((DATASET_DIR / cname).glob("*.jpg")))
        print(f"{cname}: {count}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(2000).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# ==========================================
# MODEL DEFINITION
# ==========================================
def build_model(num_classes):
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ])

    base = keras.applications.MobileNetV3Small(
        include_top=False,
        pooling="avg",
        input_shape=IMG_SIZE + (3,),
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_aug(inputs)
    x = keras.applications.mobilenet_v3.preprocess_input(x)
    x = base(x, training=False)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs), base


# ==========================================
# TRAINING
# ==========================================
def train_model():
    train_ds, val_ds, class_names = prepare_datasets()
    model, base = build_model(len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    # Train head
    print("\n=== Training classification head ===")
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

    # Fine-tuning: unfreeze last 30% of the network
    print("\n=== Fine-tuning ===")
    base.trainable = True
    num_layers = len(base.layers)
    fine_tune_from = int(num_layers * 0.7)

    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.AdamW(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks)

    model.save(MODEL_PATH)
    print("\nSaved model:", MODEL_PATH)

    convert_to_tflite(model, train_ds)


# ==========================================
# TFLITE CONVERSION
# ==========================================
def convert_to_tflite(model, dataset):
    print("\nConverting to TFLite...")

    def rep():
        for imgs, _ in dataset.unbatch().take(300):  # more samples, better quantization
            yield [imgs[None, ...]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    TFLITE_QUANT_PATH.write_bytes(tflite_model)

    print("Saved INT8 TFLite:", TFLITE_QUANT_PATH)


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    train_model()
