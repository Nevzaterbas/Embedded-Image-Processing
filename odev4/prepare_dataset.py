import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def write_idx_images(path, images):
    """Görüntüleri (Images) IDX3 formatında kaydeder."""
    print(f"Yazılıyor: {path} | Boyut: {images.shape}")
    # IDX3 magic number = 2051 (0x00000803)
    # Header: Magic(4) + Count(4) + Height(4) + Width(4)
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, images.shape[0], images.shape[1], images.shape[2]))
        f.write(images.astype(np.uint8).tobytes())

def write_idx_labels(path, labels):
    """Etiketleri (Labels) IDX1 formatında kaydeder."""
    print(f"Yazılıyor: {path} | Adet: {labels.shape[0]}")
    # IDX1 magic number = 2049 (0x00000801)
    # Header: Magic(4) + Count(4)
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, labels.shape[0]))
        f.write(labels.astype(np.uint8).tobytes())

def verify_and_visualize(image_path, label_path, num_samples=5):
    """Oluşturulan dosyaları okuyup sağlamasını yapar ve örnek gösterir."""
    print("\n--- Doğrulama ve Görselleştirme ---")
    
    # Görüntü dosyasını okuma (Test amaçlı)
    with open(image_path, 'rb') as f_img:
        magic, num, rows, cols = struct.unpack(">IIII", f_img.read(16))
        # Kalan veriyi oku
        buffer = f_img.read(num * rows * cols)
        data = np.frombuffer(buffer, dtype=np.uint8).reshape(num, rows, cols)
    
    # Etiket dosyasını okuma
    with open(label_path, 'rb') as f_lbl:
        magic_lbl, num_lbl = struct.unpack(">II", f_lbl.read(8))
        buffer_lbl = f_lbl.read(num_lbl)
        labels = np.frombuffer(buffer_lbl, dtype=np.uint8)

    print(f"Dosyadan okunan: {num} görüntü, {rows}x{cols} piksel.")
    
    # İlk 'num_samples' kadar örneği çizdir
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 3))
    for i in range(num_samples):
        axes[i].imshow(data[i], cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    
    plt.suptitle(f"{os.path.basename(image_path)} Dosyasından İlk {num_samples} Örnek")
    plt.show()

def main():
    # 1. Veri setini yükle
    print("MNIST veri seti indiriliyor/yükleniyor...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    outdir = "MNIST-dataset"
    os.makedirs(outdir, exist_ok=True)

    # Dosya yolları
    train_img_path = os.path.join(outdir, "train-images.idx3-ubyte")
    train_lbl_path = os.path.join(outdir, "train-labels.idx1-ubyte")
    test_img_path = os.path.join(outdir, "t10k-images.idx3-ubyte")
    test_lbl_path = os.path.join(outdir, "t10k-labels.idx1-ubyte")

    # 2. Dosyaları IDX formatında yaz
    write_idx_images(train_img_path, train_images)
    write_idx_labels(train_lbl_path, train_labels)
    write_idx_images(test_img_path, test_images)
    write_idx_labels(test_lbl_path, test_labels)

    print(f"\nİşlem tamamlandı. Dosyalar '{outdir}' klasörüne kaydedildi.")

    # 3. Yazılan dosyayı doğrula (Opsiyonel ama önerilir)
    verify_and_visualize(test_img_path, test_lbl_path)

if __name__ == "__main__":
    main()