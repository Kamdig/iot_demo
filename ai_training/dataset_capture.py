from pathlib import Path
import numpy as np
import cv2
import os

# ==========================================
# CONFIG
# ==========================================
RTSP_URL = os.environ.get("RTSP_URL")
if not RTSP_URL:
    raise RuntimeError(
        "RTSP_URL environment variable is required "
        "(e.g. rtsp://user:pass@host/path)"
    )

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "train"          # as you asked
RAW_NEUTRAL_DIR = DATASET_DIR / "neutral_raw"
NPZ_NEUTRAL_DIR = BASE_DIR / "hagrid_npz" / "neutral"

IMG_SIZE = (224, 224)
N_FRAMES = 3000          # how many neutral frames to capture
TRAIN_SPLIT = 0.8        # 80% train, 20% valid

RAW_NEUTRAL_DIR.mkdir(parents=True, exist_ok=True)
NPZ_NEUTRAL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# STEP 1: CAPTURE NEUTRAL FRAMES
# ==========================================
def capture_neutral_frames():
    print(f"Opening RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP: {RTSP_URL}")

    count = 0
    while count < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        # Save raw frame as JPG
        out_path = RAW_NEUTRAL_DIR / f"neutral_{count:05d}.jpg"
        cv2.imwrite(str(out_path), frame)

        count += 1
        print(f"\rSaved neutral frame {count}/{N_FRAMES}", end="")

    cap.release()
    print("\nDone capturing neutral frames.")


# ==========================================
# STEP 2: CONVERT TO NPZ (CHW, uint8)
# ==========================================
def build_neutral_npz():
    images = []

    jpg_paths = sorted(RAW_NEUTRAL_DIR.glob("*.jpg"))
    if not jpg_paths:
        raise RuntimeError(f"No JPGs found in {RAW_NEUTRAL_DIR}, did capture run?")

    for img_path in jpg_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize full frame to 224x224
        img = cv2.resize(img, IMG_SIZE)

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))  # (3, 224, 224)
        images.append(img)

    images = np.stack(images).astype("uint8")
    print("Collected neutral images:", images.shape)

    # Train/valid split
    n = images.shape[0]
    split = int(n * TRAIN_SPLIT)
    train_imgs = images[:split]
    valid_imgs = images[split:]

    np.savez(NPZ_NEUTRAL_DIR / "neutral_train.npz", train_imgs)
    np.savez(NPZ_NEUTRAL_DIR / "neutral_valid.npz", valid_imgs)

    print("Saved:")
    print("  ", NPZ_NEUTRAL_DIR / "neutral_train.npz")
    print("  ", NPZ_NEUTRAL_DIR / "neutral_valid.npz")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    capture_neutral_frames()
    build_neutral_npz()
