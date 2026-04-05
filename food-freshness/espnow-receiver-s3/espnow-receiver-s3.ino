/**
 * ESP-NOW 接收端 - 智能食材新鲜度监测系统
 * 适配 tanakamasayuki/Arduino_TensorFlowLite_ESP32 版本
 * 功能：
 * - 接收 ESP-NOW 数据（传感器数据和预热状态）
 * - 实时推理输出预测结果和新鲜度评分
 * - SD卡存储传感器历史数据（CSV格式，包含预测结果）
 * - 支持串口命令
 *
 * SD卡引脚：SCK=GPIO12, MOSI=GPIO13, MISO=GPIO14, CS=GPIO15 (符合要求)
 * 模型文件：model.tflite 放在SD卡根目录
 */

#include <esp_now.h>
#include <WiFi.h>
#include <SD.h>
#include <SPI.h>
#include <time.h>
#include <Arduino.h>

// tanakamasayuki/Arduino_TensorFlowLite_ESP32 - all includes handled by main header
#include "TensorFlowLite_ESP32.h"

// ==================== 数据结构（与发送端一致） ====================
#define STATUS_ADS1115_OK 0x01
#define STATUS_MHZ19C_OK 0x02
#define STATUS_BME680_OK 0x04
#define PKT_TYPE_SENSOR 0x01
#define PKT_TYPE_WARMUP 0x02
#define PKT_TYPE_COMMAND 0x03

typedef struct __attribute__((packed)) {
  uint8_t dataType;
  float odor_ppm;
  float hcho_ppm;
  float co_ppm;
  float voc_ppm;
  uint16_t co2_ppm;
  int16_t co2_temp;
  float env_temp;
  float humidity;
  uint8_t sensor_status;
  uint32_t timestamp;
} SensorData;

typedef struct __attribute__((packed)) {
  uint8_t dataType;
  uint16_t remainingSec;
  uint32_t timestamp;
} WarmupStatus;

typedef struct __attribute__((packed)) {
  uint8_t dataType;
  char command[32];
  uint32_t timestamp;
} CommandPacket;

// ==================== 配置 ====================
// 发送端 MAC 地址
uint8_t sensorMac[] = {0x90, 0xE5, 0xB1, 0xCC, 0x3C, 0x78};
const unsigned long CONNECTION_TIMEOUT = 45000;
const unsigned long STARTUP_GRACE = 120000;
const int OFFLINE_CONFIRM_COUNT = 2;

// SD卡配置 - 严格按照要求的引脚
#define SD_SCK  12
#define SD_MISO 14
#define SD_MOSI 13
#define SD_CS   15

// 模型路径
#define MODEL_FILE "/model.tflite"

//  arena大小 - 足够小模型
#define TENSOR_ARENA_SIZE 100000

// ==================== 全局变量 ====================
unsigned long startTime = 0;
SensorData latestData;
bool hasValidSensorData = false;

// 连接监控
struct PeerMonitor {
  uint8_t mac[6];
  unsigned long lastSeen;
  bool wasConnected;
  char name[16];
};
PeerMonitor peers[4];
int peerCount = 0;

// TensorFlow Lite - all types already in tflite namespace via main header
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::AllOpsResolver resolver;

  const tflite::Model* model;
  tflite::MicroInterpreter* interpreter;
  alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

bool model_loaded = false;

// 推理结果缓存
float last_confidence = 0.0f;
int last_predicted = 0;

// SD卡
SPIClass *sd_spi = nullptr;
bool sd_ready = false;
File data_file;

// ==================== 类别名称 - 根据你的训练修改 ====================
const char *classNames[] = {
  "apple_fresh",
  "apple_stale",
  "apple_rotten"
};
const int numClasses = sizeof(classNames) / sizeof(classNames[0]);

// ==================== 函数声明 ====================
bool initSD();
bool loadModelFromSD();
void logSensorDataToSD(const SensorData &data, int pred_class, float freshness);
void addPeer(const uint8_t *mac, const char *name);
void onReceive(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len);
void checkAllConnections();
int runInference(const SensorData &data);
float getConfidence();
int calculateFreshnessScore(float confidence, int predictedClass);
void serialTask(void *arg);

// ==================== setup ====================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("\n===== ESP-NOW 接收端 - 智能食材新鲜度监测 =====");
  startTime = millis();

  // 初始化SD卡
  if (!initSD()) {
    Serial.println("❌ SD卡初始化失败，无法加载模型和记录数据");
    while (1) delay(1000);
  }
  Serial.println("✅ SD卡初始化完成");

  // 检查test文件夹存在
  bool testDirExists = SD.exists("/test");
  Serial.printf("📁 test文件夹 exists: %d\n", testDirExists);
  if (!testDirExists) {
    bool ok = SD.mkdir("/test");
    Serial.printf("📁 创建test文件夹 [%d]\n", ok);
  }

  // 加载模型从SD卡
  if (!loadModelFromSD()) {
    Serial.printf("❌ 模型加载失败: %s\n", "check if model.tflite exists on SD root");
  } else {
    Serial.println("✅ 模型加载完成，准备推理");
    model_loaded = true;
  }

  // 创建日志文件
  char filename[64];
  time_t unixTime = time(NULL);
  if (unixTime > 1000000000) {
    struct tm *tm;
    tm = gmtime(&unixTime);
    snprintf(filename, sizeof(filename), "/test/sensor_log_%04d%02d%02d_%02d%02d.csv", 
      tm->tm_year+1900, tm->tm_mon+1, tm->tm_mday, tm->tm_hour, tm->tm_min);
  } else {
    String mac = WiFi.macAddress();
    mac.replace(":", "");
    snprintf(filename, sizeof(filename), "/test/sensor_log_%s_%lu.csv", 
      mac.c_str(), (unsigned long)millis());
  }
  Serial.printf("📝 创建日志文件: %s\n", filename);
  data_file = SD.open(filename, FILE_WRITE);
  if (data_file) {
    data_file.println("timestamp,odor_ppm,hcho_ppm,co_ppm,voc_ppm,co2_ppm,co2_temp,env_temp,humidity,sensor_status,prediction_class,freshness_score");
    data_file.flush();
    Serial.printf("✅ 日志文件创建成功\n");
  } else {
    Serial.printf("⚠️ 无法创建日志文件\n");
  }

  // 初始化ESP-NOW
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW 初始化失败");
    while (1) delay(100);
  }
  esp_now_register_recv_cb(onReceive);

  // 添加发送端
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, sensorMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("添加发送端失败");
  }

  addPeer(sensorMac, "气体发送端");

  // 后台串口任务处理命令
  xTaskCreatePinnedToCore(serialTask, "serialTask", 8192, NULL, 1, NULL, 1);

  Serial.print("本机 MAC: ");
  Serial.println(WiFi.macAddress());
  Serial.println("等待数据...");
}

// ==================== SD卡初始化 ====================
bool initSD() {
  sd_spi = new SPIClass();
  sd_spi->begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  pinMode(SD_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH);

  if (!SD.begin(SD_CS, *sd_spi)) {
    sd_ready = false;
    return false;
  }

  sd_ready = true;
  return true;
}

// ==================== 从SD卡加载模型 ====================
bool loadModelFromSD() {
  if (!sd_ready) return false;

  File file = SD.open(MODEL_FILE);
  if (!file) {
    Serial.printf("❌ Cannot open %s\n", MODEL_FILE);
    return false;
  }

  size_t modelSize = file.size();
  Serial.printf("📄 Model file size: %zu bytes (%.1f KB)\n", modelSize, (float)modelSize / 1024);

  // 分配buffer存储模型
  uint8_t *model_buffer = (uint8_t*)heap_caps_malloc(modelSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!model_buffer) {
    Serial.println("❌ malloc failed for model buffer");
    file.close();
    return false;
  }

  // 读取模型
  size_t read = file.read(model_buffer, modelSize);
  file.close();
  if (read != modelSize) {
    Serial.println("❌ Read incomplete");
    free(model_buffer);
    return false;
  }

  // 初始化TFLite
  model = tflite::GetModel(model_buffer);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Schema version mismatch");
    free(model_buffer);
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println("❌ AllocateTensors failed");
    free(model_buffer);
    return false;
  }

  Serial.println("✅ TFLite initialization complete");
  return true;
}

// ==================== 记录传感器数据到SD卡 ====================
void logSensorDataToSD(const SensorData &data, int pred_class, float freshness) {
  if (!sd_ready || !data_file) {
    return;
  }
  data_file.printf("%u,%.2f,%.2f,%.2f,%.2f,%u,%d,%.2f,%.2f,%u,%d,%.2f\n",
    data.timestamp,
    data.odor_ppm,
    data.hcho_ppm,
    data.co_ppm,
    data.voc_ppm,
    data.co2_ppm,
    data.co2_temp,
    data.env_temp,
    data.humidity,
    data.sensor_status,
    pred_class,
    freshness
  );
  if (data_file.position() > 4096) {
    data_file.flush();
  }
}

// ==================== 后台串口任务 ====================
void serialTask(void *arg) {
  while (true) {
    if (Serial.available() > 0) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd.equalsIgnoreCase("info model")) {
        Serial.println("\n===== 模型信息 =====");
        if (model_loaded && interpreter) {
          Serial.println("✅ Model loaded");
          int input_size = interpreter->inputs().size();
          int output_size = interpreter->outputs().size();
          Serial.printf("  Input size: %d\n", input_size);
          Serial.printf("  Output size: %d\n", output_size);
        } else {
          Serial.println("❌ Model not loaded");
        }
        Serial.println("====================\n");
      } else if (cmd.length() > 0) {
        // 转发命令到发送端
        CommandPacket pkt;
        pkt.dataType = PKT_TYPE_COMMAND;
        strncpy(pkt.command, cmd.c_str(), 31);
        pkt.command[31] = '\0';
        pkt.timestamp = millis();
        esp_err_t result = esp_now_send(sensorMac, (uint8_t*)&pkt, sizeof(pkt));
        if (result == ESP_OK) {
          Serial.printf("命令已发送: %s\n", cmd.c_str());
        } else {
          Serial.printf("命令发送失败, code: %d\n", result);
        }
      }
    }
    delay(10);
  }
}

// ==================== 命令发送到发送端 ====================
void sendCommand(const char* cmd) {
  CommandPacket pkt;
  pkt.dataType = PKT_TYPE_COMMAND;
  strncpy(pkt.command, cmd, 31);
  pkt.command[31] = '\0';
  pkt.timestamp = millis();
  esp_err_t result = esp_now_send(sensorMac, (uint8_t*)&pkt, sizeof(pkt));
  if (result == ESP_OK) {
    Serial.printf("命令已发送到发送端: %s\n", cmd);
  } else {
    Serial.printf("命令发送失败，错误码: %d\n", result);
  }
}

// ==================== loop ====================
void loop() {
  checkAllConnections();
  delay(100);
}

// ==================== ESP-NOW 接收回调 ====================
void onReceive(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  for (int i = 0; i < peerCount; i++) {
    if (memcmp(info->src_addr, peers[i].mac, 6) == 0) {
      peers[i].lastSeen = millis();
      break;
    }
  }

  if (len == sizeof(SensorData)) {
    SensorData tmp;
    memcpy(&tmp, incomingData, sizeof(tmp));
    if (tmp.dataType == PKT_TYPE_SENSOR) {
      latestData = tmp;
      hasValidSensorData = true;
      Serial.println("\n========== 接收到传感器数据 ==========");
      Serial.print("发送方 MAC: ");
      for (int i = 0; i < 6; i++) {
        Serial.printf("%02X", info->src_addr[i]);
        if (i < 5) Serial.print(":");
      }
      Serial.println();
      Serial.printf("发送端时间戳: %u ms\n", latestData.timestamp);
      Serial.printf("传感器状态: %s\n", getStatusString(tmp.sensor_status).c_str());
      Serial.printf("Odor: %.2f ppm\n", tmp.odor_ppm);
      Serial.printf("HCHO: %.2f ppm\n", tmp.hcho_ppm);
      Serial.printf("CO: %.2f ppm\n", tmp.co_ppm);
      Serial.printf("VOC: %.2f ppm\n", tmp.voc_ppm);
      Serial.printf("CO2: %d ppm\n", tmp.co2_ppm);
      Serial.printf("CO2温度: %d °C\n", tmp.co2_temp);
      Serial.printf("环境温度: %.2f °C\n", tmp.env_temp);
      Serial.printf("湿度: %.2f %%\n", tmp.humidity);
      
      int predictedClass = 0;
      float confidence = 0;
      int freshnessScore = 0;

      if (model_loaded) {
        predictedClass = runInference(latestData);
        confidence = getConfidence();
        freshnessScore = calculateFreshnessScore(confidence, predictedClass);
        
        Serial.println("\n================ 推理结果 ================");
        Serial.printf("预测类别: %s\n", classNames[predictedClass]);
        Serial.printf("置信度: %.1f%%\n", confidence * 100);
        Serial.printf("新鲜度评分: %d/100\n", freshnessScore);
        Serial.println("========================================\n");
      } else {
        Serial.println("\n⚠️  模型未加载，无法推理\n");
      }
      
      if (model_loaded && data_file) {
        logSensorDataToSD(tmp, predictedClass, freshnessScore);
      }
      Serial.println("======================================\n");
    }
  }
  else if (len == sizeof(WarmupStatus)) {
    WarmupStatus warmup;
    memcpy(&warmup, incomingData, sizeof(warmup));
    if (warmup.dataType == PKT_TYPE_WARMUP) {
      Serial.printf("\n[预热状态] 剩余: %d 秒\n", warmup.remainingSec);
      if (warmup.remainingSec == 0) {
        Serial.println("预热完成，开始发送数据。");
      }
    }
  }
}

// ==================== 连接管理 ====================
void addPeer(const uint8_t *mac, const char *name) {
  if (peerCount >= 4) return;
  memcpy(peers[peerCount].mac, mac, 6);
  peers[peerCount].lastSeen = 0;
  peers[peerCount].wasConnected = false;
  strncpy(peers[peerCount].name, name, 15);
  peerCount++;
}

void checkAllConnections() {
  static unsigned long lastCheck = 0;
  unsigned long now = millis();
  if (now - lastCheck < 1000) return;
  lastCheck = now;

  bool inGrace = (now - startTime) < STARTUP_GRACE;
  static int offlineCnt[4] = {0};

  for (int i = 0; i < peerCount; i++) {
    bool online = (now - peers[i].lastSeen) <= CONNECTION_TIMEOUT;
    if (inGrace) online = true;

    if (!online && peers[i].wasConnected) {
      offlineCnt[i]++;
      if (offlineCnt[i] >= OFFLINE_CONFIRM_COUNT) {
        Serial.printf("\n[警告] %s 已离线！\n", peers[i].name);
        peers[i].wasConnected = false;
        offlineCnt[i] = 0;
      }
    } else if (online && !peers[i].wasConnected) {
      offlineCnt[i] = 0;
      Serial.printf("\n[连接] %s 已恢复在线。\n", peers[i].name);
      peers[i].wasConnected = true;
    }
  }

  if (!hasValidSensorData && (now - lastCheck > 30000)) {
    Serial.println("[提醒] 尚未收到传感器数据");
  }
}

// ==================== 帮助函数 ====================
String getStatusString(uint8_t status) {
  String s = "";
  if (status & STATUS_ADS1115_OK) s += "ADS1115 ";
  if (status & STATUS_MHZ19C_OK) s += "MHZ19C ";
  if (status & STATUS_BME680_OK) s += "BME680 ";
  if (s.length() == 0) s = "None";
  return s;
}

// ==================== 模型推理 ====================
void preprocessInput(const SensorData &data, float input[5]) {
  input[0] = data.odor_ppm;
  input[1] = data.hcho_ppm;
  input[2] = data.co_ppm;
  input[3] = data.voc_ppm;
  input[4] = (float)data.co2_ppm;
}

int runInference(const SensorData &data) {
  if (!interpreter || !model_loaded) {
    last_confidence = 0.0f;
    last_predicted = 0;
    return 0;
  }

  float input[5];
  preprocessInput(data, input);
  
  // typed_input supported in this version
  for (int i = 0; i < 5; i++) {
    interpreter->typed_input<float>(input[i], &i);
  }
  
  interpreter->Invoke();
  
  int predictedClass = 0;
  float maxProb = 0.0f;
  int output_size = interpreter->outputs().size();
  for (int i = 0; i < output_size; i++) {
    float prob = interpreter->typed_output<float>(i);
    if (prob > maxProb) {
      maxProb = prob;
      predictedClass = i;
    }
  }
  
  last_confidence = maxProb;
  last_predicted = predictedClass;
  return predictedClass;
}

float getConfidence() {
  return last_confidence;
}

int calculateFreshnessScore(float confidence, int predictedClass) {
  int baseScore = (int)(confidence * 100);
  int score = baseScore - predictedClass * 20;
  if (score < 0) score = 0;
  if (score > 100) score = 100;
  return score;
}
