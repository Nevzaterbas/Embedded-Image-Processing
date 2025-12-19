import serial
import cv2
import numpy as np
import time
import sys
import os

# ==========================================
# AYARLAR
# ==========================================
SERIAL_PORT = 'COM3'  # Yarın Aygıt Yöneticisi'nden kontrol edip burayı değiştirin!
BAUD_RATE   = 115200 
TIMEOUT     = 5       

# Dosya adını tam olarak buraya yazın
IMAGE_NAME  = 'lena.jpg'  # Eğer dosyayı png olarak kaydettiyseniz 'lena.png' yapın

WIDTH  = 64
HEIGHT = 64
IMG_SIZE = WIDTH * HEIGHT

def main():
    # 1. Seri Port Bağlantısı
    print(f"[-] {SERIAL_PORT} portuna bağlanılıyor...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"[-] Bağlantı başarılı! Port oturması için bekleniyor...")
        time.sleep(2) 
    except serial.SerialException:
        print(f"[!] HATA: {SERIAL_PORT} bulunamadı veya açılamadı.")
        print("    -> Kabloyu kontrol edin.")
        print("    -> Aygıt Yöneticisi'nden doğru COM numarasını (Örn: COM5) öğrenin.")
        sys.exit()

    # 2. Resmi Kontrol Et ve Oku
    if not os.path.exists(IMAGE_NAME):
        print(f"[!] HATA: '{IMAGE_NAME}' dosyası bu klasörde yok!")
        print(f"    -> Lütfen resmin adının kodda yazan '{IMAGE_NAME}' ile aynı olduğundan emin olun.")
        ser.close()
        sys.exit()

    img = cv2.imread(IMAGE_NAME, 0) # Gri tonlamalı oku

    # Resmi 64x64'e küçült
    img_resized = cv2.resize(img, (WIDTH, HEIGHT))
    img_data = img_resized.flatten()

    print(f"[-] Resim hazırlandı ve gönderiliyor... ({len(img_data)} byte)")

    # 3. Gönder ve Al
    try:
        ser.reset_input_buffer()
        ser.write(img_data.tobytes()) # Gönder

        print("[-] STM32 işliyor... Yanıt bekleniyor...")
        
        # Tam veri gelene kadar bekle
        received_data = ser.read(IMG_SIZE)
        
        if len(received_data) != IMG_SIZE:
            print(f"[!] HATA: Eksik veri! Gelen: {len(received_data)}, Beklenen: {IMG_SIZE}")
            print("    -> STM32 reset düğmesine basıp tekrar deneyin.")
        else:
            print("[-] Veri başarıyla alındı!")
            
            # 4. Göster
            result_img = np.frombuffer(received_data, dtype=np.uint8).reshape((HEIGHT, WIDTH))

            # Ekranda büyük gözükmesi için 5 kat büyütelim
            scale = 5
            disp_orig = cv2.resize(img_resized, (WIDTH*scale, HEIGHT*scale), interpolation=cv2.INTER_NEAREST)
            disp_res  = cv2.resize(result_img,  (WIDTH*scale, HEIGHT*scale), interpolation=cv2.INTER_NEAREST)

            cv2.imshow(f"Girdi: {IMAGE_NAME}", disp_orig)
            cv2.imshow("Cikti: STM32 Otsu/Morfoloji", disp_res)
            
            print("[-] Sonuç ekranda. Çıkış için bir tuşa basın.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"[!] Bir hata oluştu: {e}")
    finally:
        ser.close()

if __name__ == "__main__":
    main()