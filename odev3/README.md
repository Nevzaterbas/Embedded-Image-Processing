# EE4065 - Gömülü Sayısal Görüntü İşleme: Proje Raporu

**Proje Durumu:** Tamamlandı (Completed) ✅  
**Ders:** EE4065 Embedded Digital Image Processing  
**Teslim Tarihi:** 19 Aralık 2025  
**Platform:** STM32 Nucleo-F446RE & PC (Python)  
**Protokol:** UART (115200 Baud Rate)

---

## 1. Proje Sonuç Özeti
Bu proje kapsamında, STM32 mikrodenetleyicisi üzerinde gerçek zamanlı görüntü işleme algoritmaları başarıyla gerçeklenmiştir. 

Sistem, PC'den gönderilen ham görüntü verilerini UART üzerinden almakta, mikrodenetleyici üzerinde **Otsu Eşikleme** ve **Morfolojik İşlemleri** (Aşındırma ve Genişletme) gerçekleştirmekte ve işlenen görüntüyü analiz için tekrar PC'ye iletmektedir. Yapılan testler sonucunda algoritmaların kararlı çalıştığı ve görüntü verilerinin kayıpsız aktarıldığı doğrulanmıştır.

---

## 2. Gerçeklenen Algoritmalar ve Teorik Altyapı

### 2.1 Otsu Metodu ile Otomatik Eşikleme (Q1)
STM32 üzerinde koşan C kodu, gelen gri seviye görüntünün histogramını çıkarmış ve sınıflar arası varyansı (inter-class variance) maksimize eden optimum eşik değerini otomatik olarak hesaplamıştır.
* **Sonuç:** Farklı ışık koşullarına sahip görüntülerde bile nesne ve arka plan ayrımı (binarization) başarıyla sağlanmıştır.

### 2.2 Morfolojik İşlemler (Q3)
3x3 boyutunda bir yapısal eleman (kernel) kullanılarak aşağıdaki işlemler uygulanmıştır:
* **Erosion (Aşındırma):** Görüntüdeki küçük beyaz gürültüler (noise) başarıyla temizlenmiş ve nesne sınırları inceltilmiştir.
* **Dilation (Genişletme):** Nesne üzerindeki küçük kopukluklar birleştirilmiş ve delikler doldurulmuştur.
* **Opening/Closing:** Bu iki temel işlem ardışık kullanılarak gürültü temizleme ve yapısal bütünlük sağlama testleri yapılmıştır.

---

## 3. Donanım ve Yazılım Mimarisi

Sistem uçtan uca (End-to-End) aşağıdaki mimari ile çalışmaktadır:

1.  **PC Arayüzü (Python):** `OpenCV` kullanılarak görüntü 64x64 piksel boyutuna indirgenmiş ve `PySerial` kütüphanesi ile paketlenerek gönderilmiştir.
2.  **İletişim Hattı:** USB-TTL dönüştürücü üzerinden 115200 baud hızında asenkron seri haberleşme sağlanmıştır.
3.  **Gömülü İşlemci (STM32):** Gelen veri DMA veya Kesme (Interrupt) kullanılmadan, bloklayıcı modda (Blocking Mode) işlenmiş ve doğrudan RAM üzerinde manipüle edilmiştir.

---

## 4. Deneysel Sonuçlar ve Performans

Sistem, standart test görüntüsü olan **"Lena"** üzerinde test edilmiştir. Aşağıdaki görseller, STM32'den geri dönen gerçek işlem çıktılarıdır.

### Test 1: Otsu Eşikleme Sonucu
Orijinal gri tonlamalı resim STM32'ye gönderilmiş, MCU histogramı analiz ederek eşik değerini belirlemiş ve sonucu binary (siyah-beyaz) olarak döndürmüştür.

| Orijinal Girdi (PC) | İşlenmiş Çıktı (STM32) |
| :---: | :---: |
| ![alt text](original_input.png) | ![alt text](otsu_output.png) |


### Test 2: Morfolojik İşlem Sonuçları
Eşiklenmiş görüntü üzerinde yapılan yapısal işlemlerin sonuçları aşağıdadır:

| İşlem Tipi | Sonuç Görüntüsü | Gözlem |
| :--- | :---: | :--- |
| **Erosion** | ![alt text](erosion_output.png) | Görüntüdeki detaylar incelmiş, arka plan gürültüleri yok olmuştur. |
| **Dilation** | ![alt text](dilation_output.png) | Ana hatlar kalınlaşmış, nesne bütünlüğü artmıştır. |

---

## 5. Dosya Yapısı

Depo (Repository) aşağıdaki dosya düzenine sahiptir:

* `Core/Src/main.c`: Otsu ve Morfoloji algoritmalarını içeren ana C kodu.
* `odev3_pc.py`: Görüntü transferini ve görselleştirmeyi sağlayan Python betiği.
* `lena.jpg`: Testlerde kullanılan kaynak görüntü.
* `results/`: İşlem sonuçlarına ait ekran görüntüleri.

---

