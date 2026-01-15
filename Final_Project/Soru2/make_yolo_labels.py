import cv2
from pathlib import Path

# =========================
# AYARLAR
# =========================
SRC = Path("dataset")          # içinde 0..9 klasörleri olan yer
OUT = Path("dataset_yolo")
IMG_OUT = OUT / "images"
LBL_OUT = OUT / "labels"
IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

def find_bbox(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # temizle
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    if w * h < 200:   # çok küçükse gürültü
        return None

    return x, y, w, h

def to_yolo(x, y, w, h, W, H):
    xc = (x + w/2) / W
    yc = (y + h/2) / H
    ww = w / W
    hh = h / H
    return xc, yc, ww, hh

ok, fail = 0, 0
for class_dir in sorted(SRC.iterdir()):
    if not class_dir.is_dir():
        continue
    if not class_dir.name.isdigit():
        continue

    class_id = int(class_dir.name)
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs += list(class_dir.glob(ext))

    for img_path in sorted(imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            print("Okunamadı:", img_path)
            fail += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bbox = find_bbox(gray)
        if bbox is None:
            print("BBox yok:", img_path)
            fail += 1
            continue

        x, y, w, h = bbox
        H, W = gray.shape[:2]
        xc, yc, ww, hh = to_yolo(x, y, w, h, W, H)

        # tekil isim verelim: 1_veri_0001.jpg gibi
        out_stem = f"{class_id}_{img_path.stem}"
        out_img = IMG_OUT / f"{out_stem}{img_path.suffix.lower()}"
        out_lbl = LBL_OUT / f"{out_stem}.txt"

        # resmi kopyala (yeniden yaz)
        cv2.imwrite(str(out_img), img)

        # label yaz
        out_lbl.write_text(f"{class_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        ok += 1

print(f"✅ Label oluşturuldu: {ok} | ❌ Hata: {fail}")
print("➡️ Çıktı klasörü:", OUT.resolve())
