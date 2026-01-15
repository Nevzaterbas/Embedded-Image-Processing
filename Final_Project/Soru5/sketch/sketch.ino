/*
  ESP32-CAM + Edge Impulse FOMO + Web Server (GRAYSCALE sensor, NO JPEG)
  Endpoints:
    /         -> simple page
    /capture  -> PGM grayscale snapshot
    /detect   -> runs inference once and returns detections as JSON
*/

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

// Edge Impulse export header
#include <nevzaterbas-project-1_inferencing.h>

// ---------------- WiFi ----------------
// !!! Buraya kendi SSID/PASS'ini yaz, şifreyi chat'e atma !!!
const char* WIFI_SSID = "FiberHGW_ZYC761";
const char* WIFI_PASS = "skXVYh7RjWfp";

// --------------- Web Server -----------
WebServer server(80);

// --------- Camera pin map (AI Thinker ESP32-CAM) ---------
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

// --------- EI input size ---------
#ifndef EI_CLASSIFIER_INPUT_WIDTH
#define EI_CLASSIFIER_INPUT_WIDTH  96
#endif
#ifndef EI_CLASSIFIER_INPUT_HEIGHT
#define EI_CLASSIFIER_INPUT_HEIGHT 96
#endif

// 96x96 grayscale buffer
static uint8_t gray96[EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT];

// Forward declarations
static bool camera_init();
static bool capture_and_preprocess();
static int ei_get_data(size_t offset, size_t length, float *out_ptr);

// ---------- Web handlers ----------
void handle_root() {
  String html =
    "<html><body>"
    "<h2>ESP32-CAM FOMO Digit Detection</h2>"
    "<p>Sensor: GRAYSCALE (no JPEG)</p>"
    "<ul>"
    "<li><a href='/capture'>/capture</a> (PGM snapshot)</li>"
    "<li><a href='/detect'>/detect</a> (run inference, JSON)</li>"
    "</ul>"
    "</body></html>";
  server.send(200, "text/html", html);
}

// Return PGM grayscale image (P5)
void handle_capture() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  // fb->buf is raw grayscale bytes because PIXFORMAT_GRAYSCALE
  String header = "P5\n" + String(fb->width) + " " + String(fb->height) + "\n255\n";

  server.setContentLength(header.length() + fb->len);
  server.send(200, "image/x-portable-graymap", "");

  WiFiClient client = server.client();
  client.write((const uint8_t*)header.c_str(), header.length());
  client.write(fb->buf, fb->len);

  esp_camera_fb_return(fb);
}

// Run inference once and return JSON detections
void handle_detect() {
  if (!capture_and_preprocess()) {
    server.send(500, "application/json",
                "{\"ok\":false,\"msg\":\"capture/preprocess failed\"}");
    return;
  }

  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
  signal.get_data = &ei_get_data;

  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    String j = String("{\"ok\":false,\"msg\":\"run_classifier err ") + (int)err + "\"}";
    server.send(500, "application/json", j);
    return;
  }

  String json = "{\"ok\":true,\"detections\":[";
  bool first = true;

  // FOMO -> bounding boxes list
  for (size_t i = 0; i < result.bounding_boxes_count; i++) {
    auto bb = result.bounding_boxes[i];
    if (bb.value <= 0.0f) continue;

    if (!first) json += ",";
    first = false;

    json += "{";
    json += "\"label\":\"" + String(bb.label) + "\",";
    json += "\"score\":" + String(bb.value, 3) + ",";
    json += "\"x\":" + String(bb.x) + ",";
    json += "\"y\":" + String(bb.y) + ",";
    json += "\"w\":" + String(bb.width) + ",";
    json += "\"h\":" + String(bb.height);
    json += "}";
  }

  json += "]}";
  server.send(200, "application/json", json);
}

// ---------- Setup / Loop ----------
void setup() {
  Serial.begin(115200);
  delay(500);

  // Camera init (GRAYSCALE)
  if (!camera_init()) {
    Serial.println("ERROR: Camera init failed");
    while (1) delay(1000);
  }
  Serial.println("Camera OK");

  // WiFi connect
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("WiFi connecting");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - t0) < 20000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi FAILED. If router is WPA3-only, set WPA2 or WPA2/WPA3 mixed.");
    while (1) delay(1000);
  }

  Serial.print("WiFi OK, IP: ");
  Serial.println(WiFi.localIP());

  // Routes
  server.on("/", handle_root);
  server.on("/capture", handle_capture);
  server.on("/detect", handle_detect);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}

// ---------- Camera init ----------
static bool camera_init() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  // Your sensor: NO JPEG -> use GRAYSCALE
  config.pixel_format = PIXFORMAT_GRAYSCALE;

  // 160x120
  config.frame_size   = FRAMESIZE_QQVGA;
  config.fb_count     = 1;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  return (err == ESP_OK);
}

// Resize 160x120 -> 96x96 grayscale (nearest neighbor)
static bool capture_and_preprocess() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return false;

  const int src_w = fb->width;   // 160
  const int src_h = fb->height;  // 120
  const uint8_t *src = fb->buf;  // grayscale bytes

  const int dst_w = EI_CLASSIFIER_INPUT_WIDTH;   // 96
  const int dst_h = EI_CLASSIFIER_INPUT_HEIGHT;  // 96

  for (int y = 0; y < dst_h; y++) {
    int sy = (y * src_h) / dst_h;
    for (int x = 0; x < dst_w; x++) {
      int sx = (x * src_w) / dst_w;
      gray96[y * dst_w + x] = src[sy * src_w + sx];
    }
  }

  esp_camera_fb_return(fb);
  return true;
}

// EI input normalization 0..255 -> 0..1
static int ei_get_data(size_t offset, size_t length, float *out_ptr) {
  for (size_t i = 0; i < length; i++) {
    out_ptr[i] = gray96[offset + i] / 255.0f;
  }
  return 0;
}
