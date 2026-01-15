import random, shutil
from pathlib import Path

BASE = Path("dataset_yolo")
IMG = BASE/"images"
LBL = BASE/"labels"

OUT = Path("yolo_data")
train_img = OUT/"images/train"
val_img   = OUT/"images/val"
train_lbl = OUT/"labels/train"
val_lbl   = OUT/"labels/val"

for p in [train_img, val_img, train_lbl, val_lbl]:
    p.mkdir(parents=True, exist_ok=True)

imgs = sorted([p for p in IMG.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
random.seed(42)
random.shuffle(imgs)

split = int(0.8 * len(imgs))
train_set = imgs[:split]
val_set   = imgs[split:]

def copy_pair(img_path, img_dst, lbl_dst):
    lbl_path = LBL / (img_path.stem + ".txt")
    shutil.copy2(img_path, img_dst / img_path.name)
    shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

for p in train_set:
    copy_pair(p, train_img, train_lbl)

for p in val_set:
    copy_pair(p, val_img, val_lbl)

print("✅ Train:", len(train_set), " Val:", len(val_set))
print("➡️ Çıktı:", OUT.resolve())
