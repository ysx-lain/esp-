/**
 * ESP-NOW 接收端 - 智能食材新鲜度监测系统
 * 功能：
 * - 接收 ESP-NOW 数据（传感器数据和预热状态）
 * - 实时在串口打印接收的数据（多行人类可读格式）
 * - SD卡存储传感器历史数据（CSV格式）
 * - 支持串口命令转发（set interval, skip warmup, status, update model, info model）
 * - 连接状态检测（带防抖）
 * - 支持flash自定义分区模型OTA升级，第二核心后台处理，ESP-NOW不受影响
 * - 接收完成直接生效，无需重启ESP32
 *
 * 分工：
 * - 模型 → flash自定义分区（256KB），读取快，无需重启
 * - 传感器历史数据 → SD卡存储（CSV格式）
 * 适配：chirale/TensorFlowLite_ESP32
 */

#include <esp_now.h>
#include <WiFi.h>
#include <SD.h>
#include <SPI.h>
#include <time.h>

// TensorFlow Lite for Microcontrollers - chirale library
// All headers are already included by TensorFlowLite_ESP32.h
#include <TensorFlowLite_ESP32.h>

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
// 发送端 MAC 地址（必须与发送端代码中的 receiverMac 一致）
uint8_t sensorMac[] = {0x90, 0xE5, 0xB1, 0xCC, 0x3C, 0x78}; // 请修改为您的发送端 MAC
const unsigned long CONNECTION_TIMEOUT = 45000;
const unsigned long STARTUP_GRACE = 120000;
const int OFFLINE_CONFIRM_COUNT = 2;

// SD卡配置 - SPI引脚
#define SD_SCK  12
#define SD_MISO 14
#define SD_MOSI 13
#define SD_CS   15

// 模型配置
#define MAX_MODEL_SIZE  (256 * 1024)  // 256KB足够

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

// ==================== TensorFlow Lite 全局变量（直接在这里） ====================
// 模型分区
const esp_partition_t *model_partition = nullptr;
uint8_t *model_buffer = nullptr;
uint8_t *tensor_arena = nullptr;
MicroInterpreter *interpreter = nullptr;
MicroErrorReporter *error_reporter = nullptr;
bool model_initialized = false;
char last_error[128] = "";

// 推理结果缓存
float last_confidence = 0.0f;
int last_predicted = 0;

// SD卡 - 存储传感器历史数据
SPIClass *sd_spi = nullptr;
bool sd_ready = false;
File data_file;

// ==================== 函数声明 ====================
bool findModelPartition();
size_t getModelSize();
bool loadModelFromFlash();
bool receiveAndWriteModel(Stream &stream, int timeoutSeconds = 60);
bool initSD();
void logSensorDataToSD(const SensorData &data, int pred_class, float freshness);
void logSensorDataToSD(const SensorData &data);
void addPeer(const uint8_t *mac, const char *name);
void onReceive(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len);
void checkAllConnections();
// 模型推理函数声明
int runInference(const SensorData &data);
float getConfidence();
int calculateFreshnessScore(float confidence, int predictedClass);
void modelUpgradeTask(void *arg);

// 类别名称 - 根据训练时的类别顺序定义
// 如果你的类别不同，请修改这里
const char *classNames[] = {
  "apple_fresh",
  "apple_stale",
  "apple_rotten",
  // 添加更多类别...
};
int numClasses = sizeof(classNames) / sizeof(classNames[0]);

void sendCommand(const char* cmd);
void printSensorData();
void printWarmupStatus(const WarmupStatus &w);
String getStatusString(uint8_t status);

// ==================== setup ====================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("\n===== ESP-NOW 接收端 - 智能食材新鲜度监测 =====");
  Serial.println("可用命令: set interval <秒>, skip warmup, status, update model, info model");
  startTime = millis();

  // 查找并初始化flash分区模型
  if (!findModelPartition()) {
    Serial.printf("⚠️ 模型分区初始化失败: %s\n", last_error);
    Serial.println("可以继续使用，发送 'update model' 升级模型");
    Serial.println("请检查 partitions.csv 是否添加了model分区");
  } else {
    Serial.printf("✅ 找到模型分区: %s, 大小: %d bytes (%.1f KB)\n", 
      model_partition->label, model_partition->size, (float)model_partition->size / 1024);
    // 如果分区中有模型，加载它
    if (getModelSize() > 0 && getModelSize() <= MAX_MODEL_SIZE) {
      if (loadModelFromFlash()) {
        Serial.printf("📦 模型加载完成，大小: %zu bytes\n", getModelSize());
      } else {
        Serial.printf("⚠️ 加载模型失败: %s\n", last_error);
      }
    } else {
      Serial.println("⚠️ 分区中没有找到有效模型");
      Serial.println("发送 'update model' 开始串口升级模型");
    }
  }

  // 初始化SD卡 - 存储传感器历史数据
  if (initSD()) {
    Serial.println("✅ SD卡初始化完成，传感器数据将记录到SD");
    
    // SD卡已挂载，确保test文件夹存在
    bool testDirExists = SD.exists("/test");
    Serial.printf("📁 test文件夹 exists: %d\n", testDirExists);
    if (!testDirExists) {
      bool ok = SD.mkdir("/test");
      Serial.printf("📁 创建test文件夹 [%d]\n", ok);
    }

    // 创建日志文件 - test/device_mac_YYYYMMDD_HHMM.csv
    char filename[64];
    // 获取当前时间如果有RTC，这里简单用开机时间戳
    time_t unixTime = time(NULL);
    if (unixTime > 1000000000) {
      // 如果有RTC设置了时间，用日期命名
      struct tm *tm;
      tm = gmtime(&unixTime);
      snprintf(filename, sizeof(filename), "/test/sensor_log_%04d%02d%02d_%02d%02d.csv", 
        tm->tm_year+1900, tm->tm_mon+1, tm->tm_mday, tm->tm_hour, tm->tm_min);
    } else {
      // 否则用MAC+开机毫秒，去掉MAC中的冒号（冒号不能当文件名）
      String mac = WiFi.macAddress();
      mac.replace(":", "");
      snprintf(filename, sizeof(filename), "/test/sensor_log_%s_%lu.csv", 
        mac.c_str(), (unsigned long)millis());
    }
    Serial.printf("📝 尝试创建文件: %s\n", filename);
    data_file = SD.open(filename, FILE_WRITE);
    if (data_file) {
      // 写入CSV头 - 包含预测字段
      data_file.println("timestamp,odor_ppm,hcho_ppm,co_ppm,voc_ppm,co2_ppm,co2_temp,env_temp,humidity,sensor_status,prediction_class,freshness_score");
      data_file.flush();
      Serial.printf("✅ 日志文件创建成功: %s\n", filename);
    } else {
      Serial.printf("⚠️ 无法创建日志文件 [%s], 数据不会记录\n", filename);
    }
  } else {
    Serial.println("⚠️ SD卡初始化失败，传感器数据不会记录到SD");
  }

  // 启动第二核心处理模型升级，ESP-NOW在Core 0运行互不干扰
  xTaskCreatePinnedToCore(modelUpgradeTask, "modelUpgrade", 16384, NULL, 1, NULL, 1);
  Serial.println("✅ 模型升级任务已在Core 1启动，ESP-NOW不受影响");

  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW 初始化失败");
    while (1) delay(100);
  }
  esp_now_register_recv_cb(onReceive);

  // 添加发送端为对等设备
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, sensorMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("添加发送端失败");
  }

  // 添加监控设备
  addPeer(sensorMac, "气体发送端");

  Serial.print("本机 MAC: ");
  Serial.println(WiFi.macAddress());
  Serial.println("等待数据...");
}

// ==================== 查找模型分区 ====================
bool findModelPartition() {
  last_error[0] = '\0';
  // 查找标签为"model"的分区
  const esp_partition_t *found = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
  if (found) {
    model_partition = found;
  } else {
    // 尝试找任何足够大的数据分区
    esp_partition_iterator_t it = esp_partition_find(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, NULL);
    while (it) {
      found = esp_partition_get(it);
      if (found->size >= MAX_MODEL_SIZE) {
        model_partition = found;
        break;
      }
      it = esp_partition_next(it);
    }
  }

  if (!model_partition) {
    strncpy(last_error, "No model partition found, check partitions.csv", sizeof(last_error)-1);
    return false;
  }
  return true;
}

// ==================== 获取模型大小 ====================
size_t getModelSize() {
  if (!model_partition) return 0;
  size_t size = 0;
  esp_err_t err = esp_partition_read(model_partition, 0, &size, sizeof(size_t));
  if (err != ESP_OK) {
    return 0;
  }
  return size;
}

// ==================== 从flash加载模型 ====================
bool loadModelFromFlash() {
  last_error[0] = '\0';
  if (!model_partition) {
    strncpy(last_error, "No model partition", sizeof(last_error)-1);
    return false;
  }
  size_t storedSize = getModelSize();
  if (storedSize == 0) {
    strncpy(last_error, "No model stored", sizeof(last_error)-1);
    return false;
  }
  if (storedSize > MAX_MODEL_SIZE) {
    strncpy(last_error, "Model too large for buffer", sizeof(last_error)-1);
    return false;
  }

  // 如果之前有加载模型，释放内存
  if (interpreter) {
    delete interpreter;
    interpreter = nullptr;
  }
  if (model_buffer) {
    free(model_buffer);
    model_buffer = nullptr;
  }
  if (tensor_arena) {
    free(tensor_arena);
    tensor_arena = nullptr;
  }

  // 分配模型缓冲区
  model_buffer = (uint8_t*)heap_caps_malloc(storedSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!model_buffer) {
    strncpy(last_error, "malloc failed for model buffer", sizeof(last_error)-1);
    return false;
  }

  // 跳过前4字节（存储大小），读取模型数据
  esp_err_t err = esp_partition_read(model_partition, sizeof(size_t), model_buffer, storedSize);
  if (err != ESP_OK) {
    strncpy(last_error, esp_err_to_name(err), sizeof(last_error)-1);
    free(model_buffer);
    model_buffer = nullptr;
    return false;
  }

  // 初始化TensorFlow Lite Micro
  const Model* model = GetModel(model_buffer);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    strncpy(last_error, "Model schema version mismatch", sizeof(last_error)-1);
    free(model_buffer);
    model_buffer = nullptr;
    return false;
  }

  // 注册所有操作
  static AllOpsResolver resolver;

  // 内存分配 - 100KB足够小模型
  constexpr size_t tensorArenaSize = 100000;
  tensor_arena = (uint8_t*)heap_caps_malloc(tensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    strncpy(last_error, "malloc failed for tensor arena", sizeof(last_error)-1);
    free(model_buffer);
    model_buffer = nullptr;
    return false;
  }

  // 创建错误reporter
  static MicroErrorReporter microErrorReporter;
  error_reporter = &microErrorReporter;

  // 创建解释器
  interpreter = new MicroInterpreter(model, resolver, tensor_arena, tensorArenaSize, error_reporter);
  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    strncpy(last_error, "AllocateTensors failed", sizeof(last_error)-1);
    free(model_buffer);
    free(tensor_arena);
    model_buffer = nullptr;
    tensor_arena = nullptr;
    interpreter = nullptr;
    return false;
  }

  model_initialized = true;
  Serial.println("✅ TFLite模型初始化完成");
  return true;
}

// ==================== 通过串口接收新模型，写入flash分区 ====================
bool receiveAndWriteModel(Stream &stream, int timeoutSeconds) {
  last_error[0] = '\0';
  if (!model_partition) {
    strncpy(last_error, "No model partition", sizeof(last_error)-1);
    return false;
  }

  Serial.println("\n=== 开始接收模型数据 ===");
  Serial.println("请从电脑发送二进制模型文件...");
  Serial.printf("超时: %d 秒，最大: %d bytes\n", timeoutSeconds, MAX_MODEL_SIZE);

  // 使用静态buffer，不占用动态内存
  // 放在flash只读段，不占用DRAM
  static uint8_t tempBuffer[MAX_MODEL_SIZE] __attribute__((aligned(4), section(".rodata")));
  unsigned long start = millis();
  size_t bytesReceived = 0;

  while (millis() - start < (unsigned long)timeoutSeconds * 1000) {
    if (stream.available()) {
      while (stream.available() && bytesReceived < MAX_MODEL_SIZE) {
        tempBuffer[bytesReceived] = stream.read();
        bytesReceived++;
        if (bytesReceived % 1024 == 0) {
          Serial.printf("Received %zu KB...\n", bytesReceived / 1024);
        }
      }
      start = millis();  // 收到数据就重置超时
    }
    delay(1);
  }

  Serial.printf("\n=== 接收完成 ===\n");
  Serial.printf("Total: %zu bytes\n", bytesReceived);

  if (bytesReceived == 0) {
    strncpy(last_error, "No data received within timeout", sizeof(last_error)-1);
    return false;
  }

  if (bytesReceived >= MAX_MODEL_SIZE) {
    strncpy(last_error, "Model too large, increase MAX_MODEL_SIZE", sizeof(last_error)-1);
    return false;
  }

  // 写入flash: 先擦除，再写大小，再写数据
  size_t size = bytesReceived;
  esp_err_t err;

  err = esp_partition_erase_range(model_partition, 0, model_partition->size);
  if (err != ESP_OK) {
    strncpy(last_error, esp_err_to_name(err), sizeof(last_error)-1);
    return false;
  }
  Serial.printf("⚡ 分区擦除完成\n");

  err = esp_partition_write(model_partition, 0, &size, sizeof(size_t));
  if (err != ESP_OK) {
    strncpy(last_error, esp_err_to_name(err), sizeof(last_error)-1);
    return false;
  }

  err = esp_partition_write(model_partition, sizeof(size_t), tempBuffer, bytesReceived);
  if (err != ESP_OK) {
    strncpy(last_error, esp_err_to_name(err), sizeof(last_error)-1);
    return false;
  }

  Serial.printf("✅ 模型写入flash完成: %zu bytes\n", bytesReceived);

  // 重新加载新模型
  bool ok = loadModelFromFlash();
  if (ok) {
    Serial.println("🎉 新模型已加载并生效，无需重启！");
  } else {
    Serial.printf("⚠️ 新模型写入成功，但加载失败: %s\n", getLastError());
  }

  return ok;
}

// 获取最后错误
const char* getLastError() { return last_error; }
bool isModelInitialized() { return model_initialized; }

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

// ==================== 记录传感器数据到SD卡 ====================
// 重载1: 只存原始数据
void logSensorDataToSD(const SensorData &data) {
  if (!sd_ready || !data_file) {
    return;
  }
  // 没有预测结果时只存原始数据
  data_file.printf("%u,%.2f,%.2f,%.2f,%.2f,%u,%d,%.2f,%.2f,%u,,\n",
    data.timestamp,
    data.odor_ppm,
    data.hcho_ppm,
    data.co_ppm,
    data.voc_ppm,
    data.co2_ppm,
    data.co2_temp,
    data.env_temp,
    data.humidity,
    data.sensor_status
  );
  // 定期flush确保数据写入
  if (data_file.position() > 4096) {
    data_file.flush();
  }
}

// 重载2: 存原始数据 + 预测结果
void logSensorDataToSD(const SensorData &data, int pred_class, float freshness) {
  if (!sd_ready || !data_file) {
    return;
  }
  // CSV格式: timestamp,odor,hcho,co,voc,co2,co2_temp,temp,humidity,status,prediction,freshness
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
  // 定期flush确保数据写入
  if (data_file.position() > 4096) {
    data_file.flush();
  }
}

// ==================== 模型升级任务（运行在Core 1） ====================
void modelUpgradeTask(void *arg) {
  while (true) {
    if (Serial.available() > 0) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd.equalsIgnoreCase("update model")) {
        // 清除缓冲区，直接开始接收二进制数据
        while (Serial.available()) {
          Serial.read();
          delay(1);
        }
        bool ok = receiveAndWriteModel(Serial);
        if (ok) {
          Serial.println("✅ 模型接收成功！已写入flash分区，无需重启直接生效");
        } else {
          Serial.printf("❌ 模型接收失败: %s\n", getLastError());
        }
      } else if (cmd.equalsIgnoreCase("info model")) {
        // 查询模型信息 - 实时读取flash分区
        Serial.println("\n===== 模型信息 =====");
        if (!model_partition) {
          Serial.println("❌ 没有找到model分区，请检查partitions.csv");
        } else if (getModelSize() > 0 && getModelSize() <= MAX_MODEL_SIZE) {
          Serial.printf("✅ flash分区存有模型\n");
          Serial.printf("📦 模型大小: %zu bytes (%.1f KB)\n", 
            getModelSize(), (float)getModelSize() / 1024);
          if (model_initialized) {
            Serial.println("✅ 模型已初始化，可以推理");
          } else {
            Serial.println("⚠️ 模型加载失败");
          }
        } else {
          Serial.println("⚠️ 分区中没有模型，需要发送 'update model'");
          if (getLastError()[0] != '\0') {
            Serial.printf("💡 错误信息: %s\n", getLastError());
          }
        }
        Serial.println("====================\n");
      } else if (cmd.length() > 0) {
        // 其他命令转发给发送端（ESP-NOW）
        sendCommand(cmd.c_str());
      }
    }
    delay(10);
  }
}

// ==================== loop ====================
void loop() {
  checkAllConnections();

  // 主循环（Core 0）只处理ESP-NOW，模型升级在Core 1
  delay(100);
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

// ==================== ESP-NOW 接收回调 ====================
void onReceive(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  // 更新监控设备的最后活跃时间
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
      printSensorData();
      
      // 运行模型推理
      int predictedClass = runInference(latestData);
      float confidence = getConfidence();
      int freshnessScore = calculateFreshnessScore(confidence, predictedClass);
      
      // 打印推理结果
      Serial.println("\n================ 推理结果 ================");
      if (model_initialized) {
        Serial.printf("预测类别: %s\n", classNames[predictedClass]);
        Serial.printf("置信度: %.1f%%\n", confidence * 100);
        Serial.printf("新鲜度评分: %d/100\n", freshnessScore);
      } else {
        Serial.println("⚠️ 模型未初始化，无法推理");
      }
      Serial.println("========================================\n");
      
      // 记录到SD卡（带预测结果）
      if (model_initialized) {
        logSensorDataToSD(tmp, predictedClass, freshnessScore);
      } else {
        logSensorDataToSD(tmp);
      }
      Serial.println("======================================\n");
    }
  }
  else if (len == sizeof(WarmupStatus)) {
    WarmupStatus warmup;
    memcpy(&warmup, incomingData, sizeof(warmup));
    if (warmup.dataType == PKT_TYPE_WARMUP) {
      printWarmupStatus(warmup);
    }
  }
  else if (len == sizeof(CommandPacket)) {
    // 忽略接收端自己发出的命令包
  }
  else {
    Serial.printf("未知数据大小: %d\n", len);
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
    } else if (online) {
      offlineCnt[i] = 0;
    }
  }

  if (!hasValidSensorData && (now - lastCheck > 30000)) {
    Serial.println("[提醒] 尚未收到传感器数据");
  }
}

// ==================== 打印函数 ====================
void printSensorData() {
  Serial.printf("发送端时间戳: %u ms\n", latestData.timestamp);
  Serial.printf("传感器状态: %s\n", getStatusString(latestData.sensor_status).c_str());
  Serial.printf("Odor: %.2f ppm\n", latestData.odor_ppm);
  Serial.printf("HCHO: %.2f ppm\n", latestData.hcho_ppm);
  Serial.printf("CO: %.2f ppm\n", latestData.co_ppm);
  Serial.printf("VOC: %.2f ppm\n", latestData.voc_ppm);
  Serial.printf("CO2: %d ppm\n", latestData.co2_ppm);
  Serial.printf("CO2温度: %d °C\n", latestData.co2_temp);
  Serial.printf("环境温度: %.2f °C\n", latestData.env_temp);
  Serial.printf("湿度: %.2f %%\n", latestData.humidity);
}

void printWarmupStatus(const WarmupStatus &w) {
  Serial.printf("\n[预热状态] 剩余: %d 秒\n", w.remainingSec);
  if (w.remainingSec == 0) {
    Serial.println("预热完成，开始发送数据。");
  }
}

String getStatusString(uint8_t status) {
  String s = "";
  if (status & STATUS_ADS1115_OK) s += "ADS1115 ";
  if (status & STATUS_MHZ19C_OK) s += "MHZ19C ";
  if (status & STATUS_BME680_OK) s += "BME680 ";
  if (s.length() == 0) s = "None";
  return s;
}

// ==================== 模型推理 ====================
// 预处理输入数据
void preprocessInput(const SensorData &data, float input[5]) {
  // 使用和训练时相同的输入顺序
  input[0] = data.odor_ppm;
  input[1] = data.hcho_ppm;
  input[2] = data.co_ppm;
  input[3] = data.voc_ppm;
  input[4] = (float)data.co2_ppm;
}

// 运行推理，返回预测类别
int runInference(const SensorData &data) {
  if (!interpreter || !model_initialized) {
    last_confidence = 0.0f;
    last_predicted = 0;
    return 0;
  }

  float input[5];
  preprocessInput(data, input);
  
  // 设置输入
  for (int i = 0; i < 5; i++) {
    interpreter->typed_input<float>(input[i], &i);
  }
  
  // 运行推理
  interpreter->Invoke();
  
  // 获取输出，找到最大概率类别
  int predictedClass = 0;
  float maxProb = 0.0f;
  int outputSize = interpreter->outputs().size;
  for (int i = 0; i < outputSize; i++) {
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

// 获取最后一次推理的置信度
float getConfidence() {
  return last_confidence;
}

// 计算新鲜度评分：置信度 * 100，新鲜类别得分高
int calculateFreshnessScore(float confidence, int predictedClass) {
  // 如果是新鲜类别，得分 = 置信度 * 100
  // 如果是不新鲜，得分 = 置信度 * (100 - 基分)，这里简单处理
  // 可以根据你的类别顺序调整，假设第一个类别是新鲜，后续是不同程度不新鲜
  int baseScore = (int)(confidence * 100);
  // predictedClass越大，新鲜度越低
  int score = baseScore - predictedClass * 20;
  if (score < 0) score = 0;
  if (score > 100) score = 100;
  return score;
}
