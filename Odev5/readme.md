# Ödev Raporu — Kısa Şablon

## 1. Kişisel Bilgiler
- Nevzat Tarık Erbaş - 150720058


## 2. Amaç
- Bölüm 12 uygulamalarından
  - Keyword Spotting (FSDD)
  - Handwritten Digit Recognition (MNIST)

## 3. Veri Setleri ve Önişleme
- FSDD serbest konuşulan rakamlar (.wav)
  - sampling rate 8000 Hz
  - MFCC n_mfcc=13, n_fft=512, hop_length=256
  - sabit time frames 32 (padtrim)
- MNIST
  - normalize 255
  - input shape (28,28,1)

## 4. Modeller
- KWS (Conv2D 16)-MaxPool-(Conv2D 32)-MaxPool-Flatten-Dense64-Dense10(softmax)
  - Parametre sayısı ~17.8k
- HDR (MNIST) Conv2D 32 - Conv2D 64 - Dense128 - Dense10

## 5. Eğitim Ayarları
- Optimizer Adam
- Loss sparse_categorical_crossentropy
- Callbacks ModelCheckpoint (best), EarlyStopping (patience)
- Batch size  Epochs belirtilen komutlarda

## 6. Sonuçlar
- KWS (örnek, 30 epoch) Test accuracy ...
- MNIST (örnek, 12 epoch) Test accuracy ...

(Eğitim grafiklerini ve confusion matrix görsellerini ekle)

## 7. Quantizasyon & MCU
- TFLite float, dynamic quant ve full int8 (representative dataset) üretildi.
- TFLite - C array script kullanılabilir.

## 8. Karşılaşılan Problemler
- TensorFlow  NumPy uyumsuzluğu (çözüm numpy2)
- librosa n_fft uyarıları (çözüm n_fft azaltıldı)

## 9. Nasıl Çalıştırılır (kısa)
README.md içindeki adımları takip et.

## 10. Ekler
- models_kws ve models_mnist klasörleri (model dosyaları)
- training_history.csv
