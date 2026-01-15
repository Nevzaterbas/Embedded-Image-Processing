import cv2      
import numpy as np 

def parlak_nesne_tespiti(resim_yolu):
    # 1. ADIM: Resmi Oku
    img = cv2.imread(resim_yolu, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Hata: Resim bulunamadı!")
        return

    # --- EKSTRA: RESİM ÇOK BÜYÜKSE KÜÇÜLTELİM ---
    # Eğer resim çok büyükse işlem uzun sürer ve ekrana sığmaz.
    # Resmi daha makul bir boyuta (örneğin genişlik 800px olacak şekilde) oranlı küçültelim.
    yukseklik, genislik = img.shape
    if genislik > 1000: # Eğer resim 1000 pikselden genişse
        oran = 800 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        img = cv2.resize(img, (800, yeni_yukseklik))
        print("Uyarı: Resim ekrana sığması için küçültüldü.")

    # Soruda istenen kural: Sadece 1000 piksel seçilecek
    HEDEF_PIKSEL_SAYISI = 1000 
    
    # 2. ADIM: Histogram Hesapla
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 3. ADIM: Eşik Değerini Bul
    toplam_piksel = 0
    esik_degeri = 0

    for i in range(255, -1, -1):
        toplam_piksel += hist[i]
        if toplam_piksel >= HEDEF_PIKSEL_SAYISI:
            esik_degeri = i
            break 

    print(f"Hesaplanan Dinamik Eşik Değeri: {esik_degeri}")
    print(f"Bu değerin üzerindeki piksel sayısı: {int(toplam_piksel)}")

    # 4. ADIM: Resmi Kesip Biç (Thresholding)
    _, sonuc_resmi = cv2.threshold(img, esik_degeri, 255, cv2.THRESH_BINARY)

    # --- PENCERE AYARLARI (Sorunu Çözen Kısım) ---
    # Pencereleri oluşturup boyutlandırılabilir yapıyoruz
    cv2.namedWindow('Orijinal Resim', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'En Parlak {HEDEF_PIKSEL_SAYISI} Piksel', cv2.WINDOW_NORMAL)

    # Pencereleri ekrana sığacak boyuta (örneğin 600x400) getiriyoruz
    cv2.resizeWindow('Orijinal Resim', 600, 400)
    cv2.resizeWindow(f'En Parlak {HEDEF_PIKSEL_SAYISI} Piksel', 600, 400)

    # Sonuçları Göster
    cv2.imshow('Orijinal Resim', img)
    cv2.imshow(f'En Parlak {HEDEF_PIKSEL_SAYISI} Piksel', sonuc_resmi)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Fonksiyonu çalıştır
parlak_nesne_tespiti('test_resmi.jpg')

# ... (kodun en alt kısmı) ...

# SAĞLAMA: Beyaz pikselleri say
beyaz_piksel_sayisi = cv2.countNonZero(sonuc_resmi)
print(f"RESİMDEKİ TOPLAM BEYAZ PİKSEL SAYISI: {beyaz_piksel_sayisi}")
