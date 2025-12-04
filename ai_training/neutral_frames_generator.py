from pathlib import Path
import numpy as np
import cv2

IMG_SIZE = (224, 224)
FRAMES_DIR = Path("neutral_frames")
OUT_DIR = Path("hagrid_npz/neutral")
OUT_DIR.mkdir(parents=True, exist_ok=True)

images = []

for img_path in sorted(FRAMES_DIR.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))  # (3, 224, 224)
    images.append(img)

images = np.stack(images).astype("uint8")
print("Collected neutral images:", images.shape)

# Simple split: 80% train, 20% valid
n = images.shape[0]
split = int(n * 0.8)
train_imgs = images[:split]
valid_imgs = images[split:]

np.savez(OUT_DIR / "neutral_train.npz", train_imgs)
np.savez(OUT_DIR / "neutral_valid.npz", valid_imgs)
print("Saved neutral_train.npz and neutral_valid.npz")
