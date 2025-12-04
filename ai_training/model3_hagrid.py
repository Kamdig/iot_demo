from keras import layers
import tensorflow as tf
import numpy as np
import pathlib
import keras

# ==========================================
# PATHS
# ==========================================
BASE_DIR = pathlib.Path(__file__).resolve().parent

# Folder that contains "dislike", "like", etc.
# For example, if you have:
#   /home/felix/Downloads/dislike/dislike_train.npz
#   /home/felix/Downloads/like/like_train.npz
# then set DATASET_DIR to "/home/felix/Downloads"
DATASET_DIR = BASE_DIR / ("hagrid_npz")  # <-- CHANGE THIS

MODEL_PATH        = BASE_DIR / "model.keras"
TFLITE_QUANT_PATH = BASE_DIR / "model_int8.tflite"
CLASS_NAMES_PATH  = BASE_DIR / "class_names.txt"
MAX_TRAIN_PER_CLASS = 5000   # or even 3000 if RAM is tight
MAX_VALID_PER_CLASS = 2000


# ==========================================
# PARAMETERS
# ==========================================
IMG_SIZE = (224, 224)   # MobileNetV3Small expects this
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 15
PATIENCE = 4

# Names of your gesture classes AND folder names
# e.g. you likely have "/home/felix/Downloads/dislike" and "/home/felix/Downloads/like"
CLASS_NAMES = ["thumbs_down", "thumbs_up", "neutral"]   # add more if you have them


# ==========================================
# DATA PIPELINE (from per-class NPZ files)
# ==========================================
def make_split_dataset(split: str, shuffle: bool) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset for a given split ("train" or "valid")
    by loading each class's `class_split.npz` file, SUBSAMPLING,
    fixing channel order, and assigning labels.
    """
    datasets = []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        npz_path = DATASET_DIR / class_name / f"{class_name}_{split}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")

        data = np.load(npz_path)  # npz => we have to fully load once
        key = data.files[0]
        imgs = data[key]  # shape: (N, 3, 224, 224) uint8
        N = imgs.shape[0]

        # How many samples to keep for this split?
        if split == "train":
            max_per = MAX_TRAIN_PER_CLASS
        else:  # "valid"
            max_per = MAX_VALID_PER_CLASS

        k = min(N, max_per)
        print(f"{split} – class '{class_name}': using {k} / {N} samples")

        # Random subset indices
        rng = np.random.default_rng(42 + label_idx)  # deterministic per class
        idx = rng.choice(N, size=k, replace=False)

        # Subsample
        imgs_small = imgs[idx]  # still (k, 3, 224, 224)
        labels = np.full((k,), label_idx, dtype=np.int64)

        # We don't need the full big array anymore
        del imgs, data

        # Build dataset for this class
        ds = tf.data.Dataset.from_tensor_slices((imgs_small, labels))
        datasets.append(ds)

    # Concatenate all class datasets for this split
    ds_all = datasets[0]
    for ds in datasets[1:]:
        ds_all = ds_all.concatenate(ds)

    AUTOTUNE = tf.data.AUTOTUNE

    # Convert (3, 224, 224) -> (224, 224, 3) and cast to float32
    def preprocess_example(img, label):
        img = tf.transpose(img, [1, 2, 0])
        img = tf.cast(img, tf.float32)
        return img, label

    ds_all = ds_all.map(preprocess_example, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds_all = ds_all.shuffle(10_000)

    ds_all = ds_all.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds_all



def prepare_datasets():
    # Build train and validation datasets
    train_ds = make_split_dataset("train", shuffle=True)
    val_ds   = make_split_dataset("valid", shuffle=False)

    # Save class names for the Pi (index order must match labels we created)
    with open(CLASS_NAMES_PATH, "w") as f:
        f.write("\n".join(CLASS_NAMES))

    print("Class names (index order):", CLASS_NAMES)
    return train_ds, val_ds, CLASS_NAMES


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

    model = keras.Model(inputs, outputs)
    return model, base


# ==========================================
# TRAINING
# ==========================================
def train_model():
    train_ds, val_ds, class_names = prepare_datasets()
    model, base = build_model(len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=PATIENCE,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            save_best_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=2,
            monitor="val_loss",
        ),
    ]

    # ------------------------------
    # Stage 1: Train classification head
    # ------------------------------
    print("\n=== Training classification head ===")
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # ------------------------------
    # Stage 2: Fine-tune MobileNetV3Small
    # ------------------------------
    print("\n=== Fine-tuning ===")
    base.trainable = True
    num_layers = len(base.layers)
    fine_tune_from = int(num_layers * 0.7)

    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.AdamW(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    print("\nSaved model:", MODEL_PATH)

    convert_to_tflite(model, train_ds)


# ==========================================
# TFLITE CONVERSION
# ==========================================
def convert_to_tflite(model, dataset):
    print("\nConverting to TFLite...")

    def representative_data_gen():
        # Use a subset for quantization calibration
        for imgs, _ in dataset.unbatch().take(300):
            yield [tf.cast(imgs[None, ...], tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
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
