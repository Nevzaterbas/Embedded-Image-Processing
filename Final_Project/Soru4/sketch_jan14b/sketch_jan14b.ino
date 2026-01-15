#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "img_converters.h" 
// Brownout (Güç Hatası) için gerekli kütüphaneler:
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// KÜTÜPHANELER
#include <EloquentTinyML.h> 
#include "squeezenet_model.h"

const char* ssid = "Rize2.4";
const char* password = "22399342";

// AYARLAR
#define NUMBER_OF_INPUTS  (28 * 28 * 1)
#define NUMBER_OF_OUTPUTS 10
// SqueezeNet biraz daha yer isteyebilir, 50KB yapalim garanti olsun.
#define TENSOR_ARENA_SIZE 50 * 1024 

using namespace Eloquent::TinyML;

TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> *ml_squeeze;

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

void preprocessImage(camera_fb_t *fb, float *input_buffer) {
    int w = fb->width;
    int h = fb->height;
    int crop_size = (w < h) ? w : h; 
    int start_x = (w - crop_size) / 2;
    int start_y = (h - crop_size) / 2;
  
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int orig_x = start_x + (x * crop_size / 28);
            int orig_y = start_y + (y * crop_size / 28);
            int p_idx = (orig_y * w) + orig_x;
            uint8_t pixel = fb->buf[p_idx];
            float norm_pixel = (255 - pixel) / 255.0;
            if (norm_pixel < 0.50) norm_pixel = 0; 
            else norm_pixel = 1.0; 
            input_buffer[(y * 28) + x] = norm_pixel;
        }
    }
}

// HTML
static const char PROGMEM INDEX_HTML[] = R"rawliteral(
<!doctype html>
<html>
<head><title>SqueezeNet</title></head>
<body><h1>SqueezeNet Test</h1><img src="/stream"></body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];
  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);

  static float input[NUMBER_OF_INPUTS];
  static float output[NUMBER_OF_OUTPUTS];
  long frame_count = 0;

  while(true){
    fb = esp_camera_fb_get();
    if (!fb) { res = ESP_FAIL; break; }

    if (frame_count++ % 10 == 0) { 
        preprocessImage(fb, input);
        
        Serial.println("\n--- TAHMIN YAPILIYOR ---");
        ml_squeeze->predict(input, output);
        
        int best = 0;
        float score = 0;
        for (int i=0; i<10; i++) {
            if (output[i] > score) {
                score = output[i];
                best = i;
            }
        }
        Serial.printf("SqueezeNet: %d (Guven: %.2f)\n", best, score);
    }

    uint8_t *jpg_buf = NULL;
    size_t jpg_len = 0;
    bool converted = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_GRAYSCALE, 30, &jpg_buf, &jpg_len);
    esp_camera_fb_return(fb);
    
    if(converted){
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, jpg_len);
        res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
        free(jpg_buf);
    }
    if(res != ESP_OK) break;
  }
  return res;
}

void startCameraServer(){
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  httpd_start(&stream_httpd, &config);
  httpd_uri_t index_uri = { .uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = NULL };
  httpd_register_uri_handler(stream_httpd, &index_uri);
  httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };
  httpd_register_uri_handler(stream_httpd, &stream_uri);
}

void setup() {
  // --- KRITIK DUZELTME: Brownout Korumasini Kapat ---
  // Bu satir, voltaj dusse bile reset atmasini engeller.
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); 
  
  Serial.begin(115200);
  Serial.println("Sistem baslatiliyor...");
  
  if(!psramInit()){
    Serial.println("HATA: PSRAM Acilmadi!");
    return;
  }
  Serial.println("PSRAM Tamam.");

  ml_squeeze = new TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE>();

  if(ml_squeeze->begin(squeezenet_data)) {
    Serial.println("SqueezeNet Yuklendi!");
  } else {
    Serial.println("HATA: Model Yuklenemedi!");
    while(1); 
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
  config.frame_size = FRAMESIZE_QVGA; 
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_count = 2;

  esp_camera_init(&config);
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Baglandi! IP: http://");
  Serial.println(WiFi.localIP());

  startCameraServer();
}

void loop() { delay(10000); }