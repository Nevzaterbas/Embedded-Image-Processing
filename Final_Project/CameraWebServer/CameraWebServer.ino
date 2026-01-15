#include "esp_camera.h"

// ==========================================
// 1. PIN TANIMLAMALARI (AI THINKER MODELİ)
// ==========================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM       5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Soru gereği hedeflenen piksel sayısı
#define HEDEF_PIKSEL_SAYISI 1000

void setup() {
  Serial.begin(115200);
  Serial.println("\n--- Soru 1-b: Adaptif Esikleme Baslatiliyor ---");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  
  // SİZİN KAMERANIZ İÇİN ÖZEL AYAR (RHYX Modeli)
  config.xclk_freq_hz = 10000000; // 10 MHz (Bağlantı hatasını önler)
  
  // Görüntü İşleme İçin En İyisi: GRAYSCALE
  // Her piksel 0-255 arası tek bir sayıdır. İşlemesi çok hızlıdır.
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  
  // Çözünürlük: QVGA (320x240) -> Toplam 76.800 piksel
  config.frame_size = FRAMESIZE_QVGA; 
  
  config.fb_location = CAMERA_FB_IN_PSRAM; // PSRAM kullan (Hızlı erişim)
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_count = 1;

  // Kamerayı Başlat
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Kamera Hatasi! Kod: 0x%x\n", err);
    return;
  }
  
  Serial.println("Kamera basariyla baslatildi.");
}

void loop() {
  // 1. Görüntüyü Yakala (Frame Buffer al)
  camera_fb_t * fb = esp_camera_fb_get();
  
  if (!fb) {
    Serial.println("Goruntu alinamadi!");
    delay(1000);
    return;
  }

  // --- ALGORİTMA BAŞLIYOR (Soru 1-b Çözümü) ---

  // 2. Histogram Dizisini Sıfırla
  // Hangi parlaklıktan kaç adet olduğunu sayacağız.
  // static tanımladık ki her seferinde bellekte yer açmasın, hızlı olsun.
  static int histogram[256]; 
  memset(histogram, 0, sizeof(histogram)); // Diziyi temizle

  // 3. Pikselleri Say (Histogram Analizi)
  // fb->buf : Resim verisinin olduğu dizi
  // fb->len : Resimdeki toplam piksel sayısı
  for (size_t i = 0; i < fb->len; i++) {
    uint8_t parlaklik = fb->buf[i]; // O pikselin değeri (0-255)
    histogram[parlaklik]++;         // O değerin sayacını 1 arttır
  }

  // 4. Adaptif Eşik Değerini Hesapla
  int toplam_piksel = 0;
  int hesaplanan_esik = 0;

  // En parlak (255) değerden geriye doğru sayıyoruz
  for (int i = 255; i >= 0; i--) {
    toplam_piksel += histogram[i];
    
    // Eğer hedef sayıyı (1000) geçtikse, eşiği bulduk!
    if (toplam_piksel >= HEDEF_PIKSEL_SAYISI) {
      hesaplanan_esik = i;
      break; 
    }
  }

  // 5. Sonuçları Serial Monitor'e Yaz
  // Bunu görerek sistemin ışığa tepki verip vermediğini anlayacağız.
  Serial.printf("Hedef: %d -> Bulunan Esik: %d | (Gercek Toplam: %d)\n", 
                HEDEF_PIKSEL_SAYISI, hesaplanan_esik, toplam_piksel);

  // 6. (Opsiyonel) Görüntüyü Eşikleme (Thresholding)
  // Hafızadaki resmi gerçekten Siyah-Beyaz yapıyoruz.
  // Bu işlem "extracted based on its size" şartını yerine getirir.
  for (size_t i = 0; i < fb->len; i++) {
    if (fb->buf[i] >= hesaplanan_esik) {
      fb->buf[i] = 255; // BEYAZ (Parlak Nesne)
    } else {
      fb->buf[i] = 0;   // SİYAH (Arka Plan)
    }
  }

  // Hafızayı serbest bırak (Çok önemli!)
  esp_camera_fb_return(fb);

  // Biraz bekle (Çok hızlı akmasın)
  delay(500); 
}