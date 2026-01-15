from pathlib import Path

# TFLite model yolu (gerekirse adını değiştir)
TFLITE_PATH = Path("runs/detect/train/weights/best_saved_model/best_float32.tflite")
OUT_PATH = Path("model_data.h")
ARRAY_NAME = "g_yolo_model_data"


def main():
    if not TFLITE_PATH.is_file():
        raise FileNotFoundError(f"TFLite dosyası bulunamadı: {TFLITE_PATH}")

    data = TFLITE_PATH.read_bytes()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("#include <cstdint>\n\n")
        f.write(f"const unsigned char {ARRAY_NAME}[] = {{\n  ")

        for i, b in enumerate(data):
            f.write(str(b))
            if i != len(data) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")

        f.write("\n};\n")
        f.write(f"const unsigned int {ARRAY_NAME}_len = {len(data)};\n")

    print(f"Oluşturuldu: {OUT_PATH} (boyut: {len(data)} byte)")


if __name__ == "__main__":
    main()
