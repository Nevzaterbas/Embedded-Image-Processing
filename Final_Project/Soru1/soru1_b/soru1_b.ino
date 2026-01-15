#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_http_server.h"

// ==========================================
// 1. WI-FI BILGILERINI GIRIN
// ==========================================
const char* ssid = "FiberHGW_ZYC761";
const char* password = "skXVYh7RjWfp";

#define HEDEF_PIKSEL_SAYISI 1000

// ==========================================
// 2. PIN TANIMLAMALARI (AI THINKER)
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

httpd_handle_t stream_httpd = NULL;

// ==========================================
// 3. GÖRÜNTÜ ISLEME VE YAYIN FONKSIYONU
// ==========================================
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
  if(res != ESP_OK) return res;

  while(true){
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Kamera yakalama hatasi");
      res = ESP_FAIL;
    } else {
      
      // --- ALGORITMA BASLIYOR ---
      static int histogram[256];
      memset(histogram, 0, sizeof(histogram));
      
      for (size_t i = 0; i < fb->len; i++) {
        histogram[ fb->buf[i] ]++;
      }

      int toplam = 0;
      int esik = 0;
      for (int i = 255; i >= 0; i--) {
        toplam += histogram[i];
        if (toplam >= HEDEF_PIKSEL_SAYISI) {
          esik = i;
          break;
        }
      }

      // --- DEĞİŞİKLİK BURADA YAPILDI ---
      // Artık sadece Hedef ve Eşik Değeri yazılıyor.
      Serial.printf("Hedef: %d | Hesaplanan Esik: %d\n", HEDEF_PIKSEL_SAYISI, esik);

      // Görüntüyü Siyah-Beyaz Yap (Thresholding)
      for (size_t i = 0; i < fb->len; i++) {
        if (fb->buf[i] >= esik) {
          fb->buf[i] = 255; // Beyaz
        } else {
          fb->buf[i] = 0;   // Siyah
        }
      }
      
      // JPEG Çevirimi
      bool jpeg_converted = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_GRAYSCALE, 80, &_jpg_buf, &_jpg_buf_len);
      
      esp_camera_fb_return(fb); 
      fb = NULL;

      if(!jpeg_converted){
        Serial.println("JPEG cevirme hatasi!");
        res = ESP_FAIL;
      }
    }

    if(res == ESP_OK){
      size_t hlen = snprintf((char *)part_buf, 64, "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, (const char *)"\r\n--frame\r\n", 12);
    }

    if(_jpg_buf){
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    
    if(res != ESP_OK){
      break;
    }
  }
  return res;
}

void startCameraServer(){
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); 
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println();

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
  
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE; 
  config.frame_size = FRAMESIZE_QVGA;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Kamera Hatasi 0x%x", err);
    return;
  }

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Baglandi");

  startCameraServer();

  Serial.print("Canli Yayin Linki: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");
}

void loop() {
  delay(10000);
}