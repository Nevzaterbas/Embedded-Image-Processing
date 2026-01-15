#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "img_converters.h" 

// ==========================================
// AĞ BİLGİLERİ
// ==========================================
const char* ssid = "Rize2.4";
const char* password = "22399342";

// ==========================================
// PIN AYARLARI
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

// =============================================================
// 1. FONKSİYON: RESIZE (Nearest Neighbor Algoritması)
// =============================================================
bool resizeImage(uint8_t* src_buf, int src_w, int src_h, float scale, 
                 uint8_t** out_buf, int* out_w, int* out_h) {
    
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);

    // PSRAM kullanarak hafıza ayır
    uint8_t* new_buf = (uint8_t*)heap_caps_malloc(new_w * new_h, MALLOC_CAP_SPIRAM);
    if (!new_buf) return false;

    // En Yakın Komşu (Nearest Neighbor) Mantığı
    for (int y = 0; y < new_h; y++) {
        int src_y = (int)(y / scale);
        if (src_y >= src_h) src_y = src_h - 1;

        for (int x = 0; x < new_w; x++) {
            int src_x = (int)(x / scale);
            if (src_x >= src_w) src_x = src_w - 1;

            new_buf[y * new_w + x] = src_buf[src_y * src_w + src_x];
        }
    }

    *out_buf = new_buf;
    *out_w = new_w;
    *out_h = new_h;
    return true;
}

// =============================================================
// 2. FONKSİYON: PASTE (Resimleri Birleştirme)
// =============================================================
void pasteImage(uint8_t* canvas, int canvas_w, int canvas_h, 
                uint8_t* img, int img_w, int img_h, int off_x, int off_y) {
    
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            if (off_x + x < canvas_w && off_y + y < canvas_h) {
                int canvas_idx = (off_y + y) * canvas_w + (off_x + x);
                int img_idx = y * img_w + x;
                canvas[canvas_idx] = img[img_idx];
            }
        }
    }
}

// =============================================================
// 3. HTML KODU (CSS İLE ZORLA BÜYÜTME EKLENDİ)
// =============================================================
static const char PROGMEM INDEX_HTML[] = R"rawliteral(
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ESP32-CAM Resize Demo</title>
  <style>
    body {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
      background-color: #1a1a1a; /* Koyu gri arka plan */
      overflow: hidden; /* Kaydırma çubuklarını gizle */
      font-family: Arial, sans-serif;
    }
    .stream-container {
      position: relative;
      display: inline-block;
      /* Kapsayıcıyı içeriğe göre değil, ekran yüksekliğine göre ayarla */
      height: 90vh; 
      /* Görüntü oranını koruyarak genişliği otomatik ayarla */
      aspect-ratio: 240 / 300; 
      box-shadow: 0 10px 30px rgba(0,0,0,0.8);
      border: 2px solid #444;
    }
    img {
      display: block;
      /* DÜZELTME BURADA: Resmi ekranın %90 yüksekliğine zorla */
      height: 100%; 
      width: 100%;
      /* DÜZELTME: Bulanıklığı önlemek için Pixelated rendering */
      image-rendering: pixelated;
      image-rendering: -moz-crisp-edges;
      image-rendering: crisp-edges;
    }
    .label {
      position: absolute;
      background-color: rgba(220, 20, 60, 0.9); /* Kırmızı belirgin etiket */
      color: white;
      padding: 0.5vh 1vh; /* Boyuta göre ölçeklenen yazı alanı */
      font-size: 2.5vh;   /* Yazı boyutunu da ekranla büyüt */
      font-weight: bold;
      border-radius: 4px;
      pointer-events: none;
      text-transform: uppercase;
      box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    /* Konumlandırma (% ile) */
    .lbl-orig { top: 0; left: 0; background-color: rgba(0, 100, 255, 0.8); }
    .lbl-down { top: 10%; left: 66.7%; }
    .lbl-up   { top: 40%; left: 0; background-color: rgba(255, 69, 0, 0.9); }
  </style>
</head>
<body>
  <div class="stream-container">
    <img src="/stream" alt="Yayin">
    <div class="label lbl-orig">Orijinal</div>
    <div class="label lbl-down">Kucuk (0.5x)</div>
    <div class="label lbl-up">Buyuk (1.5x)</div>
  </div>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

// =============================================================
// 4. STREAM HANDLER (Görüntü İşleme)
// =============================================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];
  
  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if(res != ESP_OK) return res;

  // TUVAL BOYUTLARI (240x300)
  const int CANVAS_W = 240;
  const int CANVAS_H = 300;
  
  uint8_t* canvas = (uint8_t*)heap_caps_malloc(CANVAS_W * CANVAS_H, MALLOC_CAP_SPIRAM);

  while(true){
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
    } else {
      
      // A. Tuvali Temizle
      memset(canvas, 0, CANVAS_W * CANVAS_H);

      // B. Orijinal (Sol Üst)
      pasteImage(canvas, CANVAS_W, CANVAS_H, fb->buf, fb->width, fb->height, 0, 0);

      // C. Küçültme (Sağ Üst)
      uint8_t* small_buf = NULL;
      int s_w, s_h;
      if (resizeImage(fb->buf, fb->width, fb->height, 0.5, &small_buf, &s_w, &s_h)) {
          pasteImage(canvas, CANVAS_W, CANVAS_H, small_buf, s_w, s_h, 160, 30);
          free(small_buf);
      }

      // D. Büyütme (Alt Taraf)
      uint8_t* big_buf = NULL;
      int b_w, b_h;
      if (resizeImage(fb->buf, fb->width, fb->height, 1.5, &big_buf, &b_w, &b_h)) {
          pasteImage(canvas, CANVAS_W, CANVAS_H, big_buf, b_w, b_h, 0, 120);
          free(big_buf);
      }

      // E. JPEG Gönder
      uint8_t *jpg_buf = NULL;
      size_t jpg_len = 0;
      
      // Kaliteyi artırdık (30 -> daha net)
      bool converted = fmt2jpg(canvas, CANVAS_W * CANVAS_H, CANVAS_W, CANVAS_H, PIXFORMAT_GRAYSCALE, 30, &jpg_buf, &jpg_len);
      
      esp_camera_fb_return(fb);

      if(converted){
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, jpg_len);
        res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
        free(jpg_buf);
      }
    }
    
    if(res != ESP_OK) break;
  }
  free(canvas);
  return res;
}

void startCameraServer(){
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.stack_size = 32768; 

  httpd_start(&stream_httpd, &config);

  httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = index_handler,
    .user_ctx  = NULL
  };
  httpd_register_uri_handler(stream_httpd, &index_uri);

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };
  httpd_register_uri_handler(stream_httpd, &stream_uri);
}

void setup() {
  Serial.begin(115200);
  
  if(psramInit()){
    Serial.println("PSRAM Basarili!");
  } else {
    Serial.println("PSRAM Hatasi!");
  }

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
  
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QQVGA; 
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_count = 2;

  if(esp_camera_init(&config) != ESP_OK) {
    Serial.println("Kamera hatasi!");
    return;
  }
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.print("\nIP Adresi: http://");
  Serial.println(WiFi.localIP());

  startCameraServer();
}

void loop() {
  delay(10000);
}