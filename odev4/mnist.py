import struct
import numpy as np
import os

def load_idx_images(path: str) -> np.ndarray:
    """
    IDX3 formatındaki görüntü dosyasını okur.
    Başlıktaki boyut bilgilerini kullanarak dinamik reshape yapar.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    with open(path, "rb") as f:
        # Başlık (Header) Okuma: 
        # Magic Number (4 byte) + Count (4 byte) + Rows (4 byte) + Cols (4 byte) = 16 bytes
        # >IIII : Big-endian formatında 4 tane unsigned int
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # Magic Number Kontrolü (IDX3 images için 2051 olmalı)
        if magic != 2051:
            raise ValueError(f"Geçersiz görüntü dosyası! Beklenen: 2051, Okunan: {magic}")
            
        # Veriyi oku ve numpy dizisine çevir
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        
        # Dosyadan okunan boyutlara göre şekillendir (N, Rows, Cols)
        images = data.reshape(num_images, rows, cols)
        
    return images

def load_idx_labels(path: str) -> np.ndarray:
    """
    IDX1 formatındaki etiket dosyasını okur.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    with open(path, "rb") as f:
        # Başlık Okuma: Magic (4 byte) + Count (4 byte) = 8 bytes
        magic, num_labels = struct.unpack(">II", f.read(8))
        
        # Magic Number Kontrolü (IDX1 labels için 2049 olmalı)
        if magic != 2049:
            raise ValueError(f"Geçersiz etiket dosyası! Beklenen: 2049, Okunan: {magic}")
            
        # Veriyi oku
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        
    return labels

# --- Kullanım Örneği ---
if __name__ == "__main__":
    # Önceki adımda oluşturduğun klasör yolu
    base_dir = "MNIST-dataset"
    
    img_path = os.path.join(base_dir, "train-images.idx3-ubyte")
    lbl_path = os.path.join(base_dir, "train-labels.idx1-ubyte")

    try:
        images = load_idx_images(img_path)
        labels = load_idx_labels(lbl_path)

        print(f"Başarıyla Yüklendi!")
        print(f"Görüntü Verisi Şekli: {images.shape}")  # Örn: (60000, 28, 28)
        print(f"Etiket Verisi Şekli: {labels.shape}")  # Örn: (60000,)
        print(f"İlk Etiket: {labels[0]}")
        
    except Exception as e:
        print(f"Hata oluştu: {e}")