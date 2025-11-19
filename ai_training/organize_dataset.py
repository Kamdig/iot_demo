from pathlib import Path
from PIL import Image

DATASET = Path("train")

def analyze_dataset():
    print("=== Dataset Summary ===")
    for cls_dir in DATASET.iterdir():
        if cls_dir.is_dir():
            images = list(cls_dir.glob("*.jpg"))
            print(f"{cls_dir.name}: {len(images)} images")
    print("========================")

def remove_corrupted():
    print("Checking for corrupted files...")
    for cls_dir in DATASET.iterdir():
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.glob("*.jpg"):
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                print(f"Removing bad file: {img_path}")
                img_path.unlink()

analyze_dataset()
remove_corrupted()
analyze_dataset()
