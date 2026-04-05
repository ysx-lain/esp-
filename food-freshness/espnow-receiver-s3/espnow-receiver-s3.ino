/**
 * ESP-NOW 接收端 - 智能食材新鲜度监测系统
 * 功能：
 * - 接收 ESP-NOW 数据（传感器数据和预热状态）
 * - 实时在串口打印接收的数据（多行人类可读格式）
 * - 支持串口命令转发（set interval, skip warmup, status, update model）
 * - 连接状态检测（带防抖）
 * - 支持SD卡模型OTA升级，第二核心后台处理，ESP-NOW不受影响
 */

#include <esp_now.h>
#include <WiFi.h>
#include "model_manager.h"

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

// SD卡模型升级配置
#define SD_CS_PIN 15  // 根据你的硬件修改CS引脚

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

// 模型管理
ModelManager *modelMgr = nullptr;

// ==================== 函数声明 ====================
void addPeer(const uint8_t *mac, const char *name);
void onReceive(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len);
void checkAllConnections();
void sendCommand(const char* cmd);
void printSensorData();
void printWarmupStatus(const WarmupStatus &w);
String getStatusString(uint8_t status);
void modelUpgradeTask(void *arg);

// ==================== setup ====================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("\n===== ESP-NOW 接收端 - 智能食材新鲜度监测 =====");
  Serial.println("可用命令: set interval <秒>, skip warmup, status, update model, info model");
  startTime = millis();

  // 初始化SD卡模型管理
  modelMgr = new ModelManager(SD_CS_PIN);
  if (!modelMgr->begin()) {
    Serial.printf("⚠️ SD卡模型管理初始化失败: %s\n", modelMgr->getLastError());
    Serial.println("可以继续使用内置模型，无法OTA升级");
  } else {
    Serial.println("✅ SD卡模型管理初始化完成");
    if (modelMgr->hasModel()) {
      Serial.println("📦 SD卡上找到模型文件，启动后加载");
    } else {
      Serial.println("⚠️ SD卡上没有找到模型，使用内置模型");
      Serial.println("发送 'update model' 开始串口升级模型");
    }
    // 启动第二核心处理模型升级，ESP-NOW在Core 0运行互不干扰
    xTaskCreatePinnedToCore(modelUpgradeTask, "modelUpgrade", 8192, NULL, 1, NULL, 1);
    Serial.println("✅ 模型升级任务已在Core 1启动，ESP-NOW不受影响");
  }

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

// ==================== 模型升级任务（运行在Core 1） ====================
void modelUpgradeTask(void *arg) {
  while (true) {
    if (Serial.available() > 0) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd.equalsIgnoreCase("update model")) {
        // 开始接收新模型
        if (!modelMgr) {
          Serial.println("❌ 模型管理未初始化");
          continue;
        }
        bool ok = modelMgr->receiveAndSaveModel(Serial);
        if (ok) {
          Serial.println("✅ 模型接收保存成功！重启后加载新模型");
        } else {
          Serial.printf("❌ 模型接收失败: %s\n", modelMgr->getLastError());
        }
      } else if (cmd.equalsIgnoreCase("info model")) {
        // 查询模型信息
        if (!modelMgr) {
          Serial.println("❌ 模型管理未初始化");
        } else {
          Serial.println("\n===== 模型信息 =====");
          if (modelMgr->hasModel()) {
            Serial.printf("✅ SD卡有模型文件\n");
            Serial.printf("📦 模型大小: %u bytes (%.1f KB)\n", 
              modelMgr->getModelSize(), (float)modelMgr->getModelSize() / 1024);
            Serial.printf("💾 存储路径: %s\n", modelMgr->getModelPath());
          } else {
            Serial.println("⚠️ 没有SD卡模型，使用内置模型");
          }
          Serial.println("====================\n");
        }
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
