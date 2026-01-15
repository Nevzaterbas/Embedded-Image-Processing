#include <Arduino.h>
#include "esp_camera.h"

// ==== MODEL ====
#include "model_data.h"  // <-- xxd -i ile ürettiğin dosya

#include <TensorFlowLite.h>
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// =====================
// KAMERA PINLERİ (AI Thinker ESP32-CAM)
// =====================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// =====================
// MODEL INPUT BOYUTU (PC’de train/export ettiğin imgsz ile aynı olmalı)
// Öneri: ESP32 için 160 veya 128
// =====================
static const int IN_W = 160;
static const int IN_H = 160;
static const int IN_CH = 3;

// YOLOv8 output: (1, 14, 2100) = (1, 4 bbox + 10 class, N)
// N = (IN_W/8 * IN_H/8) + (IN_W/16 * IN_H/16) + (IN_W/32 * IN_H/32)
// 160 için: 20*20 + 10*10 + 5*5 = 400 + 100 + 25 = 525
// 320 için: 40*40 + 20*20 + 10*10 = 1600 + 400 + 100 = 2100
// Senin log’da 2100 gördük -> IN_W/IN_H büyük ihtimal 320.
// Eğer modelin 320 ise burada 320 yazmalısın (ama ESP32 için ağır olabilir).
// Şu an 160 yazdım; eğer modelin 320 export ise IN_W=320 yap.
static const int NC = 10; // 0..9

// =====================
// TFLM arena (model ağırsa büyütmen gerekebilir)
// PSRAM varsa daha rahat.
// =====================
static const int ARENA_SIZE = 420 * 1024; // yetmezse 520KB dene
static uint8_t tensor_arena[ARENA_SIZE];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Kamera frame’i QQVGA (160x120) alacağız, sonra 160x160’e resize edeceğiz.
static uint8_t rgb_input[IN_W * IN_H * 3];

// =====================
// Kamera init
// =====================
static bool init_camera() {
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

  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  // Daha hızlı: GRAYSCALE alıp RGB’ye kopyalayacağız
  config.pixel_format = PIXFORMAT_GRAYSCALE;

  // 160x120 (QQVGA) hafif
  config.frame_size = FRAMESIZE_QQVGA;

  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

// =====================
// Basit nearest-neighbor resize: gray (sw x sh) -> RGB (IN_W x IN_H)
// =====================
static void resize_gray_to_rgb(const uint8_t* src, int sw, int sh, uint8_t* dst) {
  for (int y = 0; y < IN_H; y++) {
    int sy = (y * sh) / IN_H;
    for (int x = 0; x < IN_W; x++) {
      int sx = (x * sw) / IN_W;
      uint8_t v = src[sy * sw + sx];
      int i = (y * IN_W + x) * 3;
      dst[i + 0] = v;
      dst[i + 1] = v;
      dst[i + 2] = v;
    }
  }
}

// =====================
// Input tensor’a yaz: uint8 veya int8 olabilir
// =====================
static void write_input_tensor() {
  if (input_tensor->type == kTfLiteUInt8) {
    // 0..255 direkt
    memcpy(input_tensor->data.uint8, rgb_input, IN_W * IN_H * 3);
  } else if (input_tensor->type == kTfLiteInt8) {
    // int8 quant model: genelde zero_point= -128..127 aralığı
    int8_t* p = input_tensor->data.int8;
    for (int i = 0; i < IN_W * IN_H * 3; i++) {
      p[i] = (int16_t)rgb_input[i] - 128;
    }
  } else {
    Serial.println("Unsupported input tensor type!");
  }
}

// =====================
// YOLOv8 decode (sadece en iyi skor)
// output: shape (1, 14, N) => [x,y,w,h, c0..c9] * N
// Biz N içinden en yüksek class score’u buluyoruz.
// =====================
static void decode_and_print() {
  // output tipi: float32 veya int8 olabilir
  // Basitlik: float32 bekliyoruz. int8 ise ölçek/zero_point ile float’a çeviriyoruz.
  int dims = output_tensor->dims->size;
  if (dims != 3) {
    Serial.printf("Unexpected output dims=%d\n", dims);
    return;
  }

  int C = output_tensor->dims->data[1]; // 14
  int N = output_tensor->dims->data[2]; // 2100 (320 ise), 525 (160 ise)
  if (C != (4 + NC)) {
    Serial.printf("Unexpected C=%d (expected %d)\n", C, 4 + NC);
    // yine de devam edelim
  }

  float best_score = -1.0f;
  int best_class = -1;
  int best_i = -1;
  float bx=0, by=0, bw=0, bh=0;

  auto get_val = [&](int c, int i) -> float {
    int idx = (0 * C * N) + (c * N) + i;
    if (output_tensor->type == kTfLiteFloat32) {
      return output_tensor->data.f[idx];
    } else if (output_tensor->type == kTfLiteInt8) {
      int8_t q = output_tensor->data.int8[idx];
      float scale = output_tensor->params.scale;
      int zp = output_tensor->params.zero_point;
      return (q - zp) * scale;
    } else if (output_tensor->type == kTfLiteUInt8) {
      uint8_t q = output_tensor->data.uint8[idx];
      float scale = output_tensor->params.scale;
      int zp = output_tensor->params.zero_point;
      return (int(q) - zp) * scale;
    }
    return 0.0f;
  };

  for (int i = 0; i < N; i++) {
    // bbox
    float x = get_val(0, i);
    float y = get_val(1, i);
    float w = get_val(2, i);
    float h = get_val(3, i);

    // class scores (YOLOv8 genelde sigmoid sonrası olabilir; biz en büyük değeri alıyoruz)
    for (int cls = 0; cls < NC; cls++) {
      float s = get_val(4 + cls, i);
      if (s > best_score) {
        best_score = s;
        best_class = cls;
        best_i = i;
        bx=x; by=y; bw=w; bh=h;
      }
    }
  }

  Serial.printf("Best: class=%d score=%.3f (idx=%d)\n", best_class, best_score, best_i);
  Serial.printf("BBox(raw): x=%.3f y=%.3f w=%.3f h=%.3f\n", bx, by, bw, bh);
}

// =====================
// Setup
// =====================
void setup() {
  Serial.begin(115200);
  delay(300);

  if (!init_camera()) {
    Serial.println("Camera FAIL");
    while (1) delay(1000);
  }
  Serial.println("Camera OK");

  // ===== TFLM init =====
  // model_data.h içinde isimler farklı olabilir: burada kendi isimlerini kullan
  // ÖRNEK: digit_yolo_int8_tflite / digit_yolo_int8_tflite_len
  const tflite::Model* model = t*
