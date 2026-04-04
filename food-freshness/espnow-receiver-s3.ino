/**
 * ESP-NOW 接收端 - 智能食材新鲜度监测系统
 * 功能：
 * - 接收 ESP-NOW 数据（传感器数据和预热状态）
 * - 实时在串口打印接收的数据（多行人类可读格式）
 * - 支持串口命令转发（set interval, skip warmup, status）
 * - 连接状态检测（带防抖）
 * - 制冷片控制（可配置S3直驱 / UART发给ESP32-P4）
 * - 预留AI推理接口
 */

#include <esp_now.h>
#include <WiFi.h>
#include <driver/ledc.h>

// ==================== 配置选项 ====================
#define ENABLE_P4_CONTROL 0    // 0=S3直接PWM控制制冷片, 1=UART发给ESP32-P4

// ==================== 引脚定义（根据你的硬件修改）====================
// 制冷片PWM（S3直驱模式）
#define COOLER_PWM_PIN   GPIO_NUM_8
#define COOLER_PWM_FREQ  1000
#define COOLER_PWM_CH    LEDC_CHANNEL_0

// P4 UART（P4控制模式）
#define P4_UART          UART_NUM_2
#define P4_UART_TX       GPIO_NUM_17
#define P4_UART_RX       GPIO_NUM_18
#define P4_BAUD_RATE     115200

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

// 给P4的控制帧
typedef struct __attribute__((packed)) {
  uint8_t magic[2];
  uint8_t cmd;        // 0=关, 1=开, 2=设置功率
  uint8_t power;      // 0-100%
  uint8_t checksum;
} P4CoolerCmd;

// 推理结果
typedef struct {
  uint8_t item_id;         // 物品种类ID
  uint8_t freshness_level; // 0-差, 1-一般, 2-新鲜
  float confidence;        // 置信度
} InferenceResult;

// ==================== 配置 ====================
// 发送端 MAC 地址（必须与发送端代码中的 receiverMac 一致）
uint8_t sensorMac[] = {0x90, 0xE5, 0xB1, 0xCC, 0x3C, 0x78}; // 请修改为您的发送端 MAC
const unsigned long CONNECTION_TIMEOUT = 45000;
const unsigned long STARTUP_GRACE = 120000;
const int OFFLINE_CONFIRM_COUNT = 2;
const unsigned long DATA_PRINT_INTERVAL = 5000; // 打印间隔，避免刷屏

unsigned long startTime = 0;
SensorData latestData;
bool hasValidSensorData = false;
unsigned long lastDataPrintTime = 0;
InferenceResult lastInference;

// 连接监控
struct PeerMonitor {
  uint8_t mac[6];
  unsigned long lastSeen;
  bool wasConnected;
  char name[16];
};
PeerMonitor peers[4];
int peerCount = 0;

// ==================== 函数声明 ====================
void addPeer(const uint8_t *mac, const char *name);
void onReceive(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len);
void checkAllConnections();
void sendCommand(const char* cmd);
void printSensorData();
void printWarmupStatus(const WarmupStatus &w);
String getStatusString(uint8_t status);
void coolerInit();
void setCoolerPower(uint8_t percent);
bool runInference(const SensorData &sensor, InferenceResult &result);

// ==================== PWM / UART 初始化 ====================
#if !ENABLE_P4_CONTROL
static void pwm_cooler_init() {
  ledc_timer_config_t cfg = {
    .speed_mode = LEDC_HIGH_SPEED_MODE,
    .duty_resolution = LEDC_TIMER_8_BIT,
    .timer_num = LEDC_TIMER_0,
    .freq_hz = COOLER_PWM_FREQ,
    .clk_cfg = LEDC_AUTO_CLK
  };
  ledc_timer_config(&cfg);

  ledc_channel_config_t ch_cfg = {
    .gpio_num = COOLER_PWM_PIN,
    .speed_mode = LEDC_HIGH_SPEED_MODE,
    .channel = COOLER_PWM_CH,
    .timer_sel = LEDC_TIMER_0,
    .duty = 0,
    .hpoint = 0
  };
  ledc_channel_config(&ch_cfg);
}

void set_cooler_pwm(uint8_t percent) {
  uint32_t duty = (percent * 255UL) / 100UL;
  ledc_set_duty(LEDC_HIGH_SPEED_MODE, COOLER_PWM_CH, duty);
  ledc_update_duty(LEDC_HIGH_SPEED_MODE, COOLER_PWM_CH);
}
#else
static void p4_uart_init() {
  uart_config_t uart_config = {
    .baud_rate = P4_BAUD_RATE,
    .data_bits = UART_DATA_8_BITS,
    .parity = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
  };
  uart_param_config(P4_UART, &uart_config);
  uart_set_pin(P4_UART, P4_UART_TX, P4_UART_RX, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
  uart_driver_install(P4_UART, 256, 0, 0, NULL, 0);
}

void send_cooler_to_p4(uint8_t cmd, uint8_t power) {
  P4CoolerCmd frame;
  frame.magic[0] = 0x55;
  frame.magic[1] = 0xAA;
  frame.cmd = cmd;
  frame.power = power;
  frame.checksum = frame.magic[0] ^ frame.magic[1] ^ frame.cmd ^ frame.power;
  uart_write_bytes(P4_UART, (const char*)&frame, sizeof(frame));
}
#endif

void coolerInit() {
#if ENABLE_P4_CONTROL
  p4_uart_init();
  send_cooler_to_p4(0, 0); // 初始关闭
#else
  pwm_cooler_init();
  set_cooler_pwm(0); // 初始关闭
#endif
}

void setCoolerPower(uint8_t percent) {
#if ENABLE_P4_CONTROL
  if (percent == 0) {
    send_cooler_to_p4(0, 0);
  } else {
    send_cooler_to_p4(1, percent);
  }
#else
  set_cooler_pwm(percent);
#endif
}

// ==================== setup ====================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("\n===== ESP-NOW 接收端 - 智能食材监测 =====");
  Serial.println("可用命令: set interval <秒>, skip warmup, status, cooler <0-100>");

  startTime = millis();
  coolerInit();

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

  addPeer(sensorMac, "气体发送端");

  Serial.print("本机 MAC: ");
  Serial.println(WiFi.macAddress());
  Serial.println("配置：");
#if ENABLE_P4_CONTROL
  Serial.println("→ 制冷片控制：ESP32-P4（UART）");
#else
  Serial.println("→ 制冷片控制：S3 直接PWM");
#endif
  Serial.println("等待数据...");
}

// ==================== loop ====================
void loop() {
  checkAllConnections();

  // 处理本地串口命令（转发到发送端）
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) {
      // 本地命令处理
      if (cmd.startsWith("cooler ")) {
        int p = cmd.substring(7).toInt();
        p = constrain(p, 0, 100);
        setCoolerPower(p);
        Serial.printf("制冷片功率：%d%%\n", p);
      } else {
        // 其他命令转发给发送端
        sendCommand(cmd.c_str());
      }
    }
  }

  // 当有新数据时，运行AI推理
  static unsigned long lastInferTime = 0;
  unsigned long now = millis();
  if (hasValidSensorData && (now - lastInferTime) > 1000) {
    lastInferTime = now;

    if (runInference(latestData, lastInference)) {
      // 根据推理结果自动控制制冷
      // 示例：不新鲜 → 开启制冷保鲜
      if (lastInference.freshness_level == 0) {
        setCoolerPower(80); // 80%功率
      } else if (lastInference.freshness_level == 1) {
        setCoolerPower(40); // 中等功率
      } else {
        setCoolerPower(0);  // 关闭
      }

      // 打印推理结果
      Serial.println("\n========== AI推理结果 ==========");
      Serial.printf("物品ID: %d\n", lastInference.item_id);
      Serial.printf("新鲜度: %d级\n", lastInference.freshness_level);
      Serial.printf("置信度: %.2f\n", lastInference.confidence);
      Serial.println("================================");
    }
  }

  // 定期打印数据，避免刷屏
  if (hasValidSensorData && (now - lastDataPrintTime > DATA_PRINT_INTERVAL)) {
    lastDataPrintTime = now;
    printSensorData();
  }

  if (!hasValidSensorData && (now - lastDataPrintTime > 30000)) {
    lastDataPrintTime = now;
    Serial.println("[提醒] 尚未收到传感器数据");
  }

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
    }
  }
}

// ==================== 打印函数 ====================
void printSensorData() {
  Serial.println("\n========== 传感器数据 ==========");
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
  Serial.println("================================");
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

// ==================== AI推理占位（替换为你的模型推理）====================
bool runInference(const SensorData &sensor, InferenceResult &result) {
  // 这里填入你的 TFLite 模型推理代码
  // 示例：
  // float input[8] = {sensor.odor_ppm, sensor.hcho_ppm, sensor.co_ppm, sensor.voc_ppm,
  //                   sensor.co2_ppm, (float)sensor.co2_temp, sensor.env_temp, sensor.humidity};
  // 模型推理 → 填充 result
  // 返回 true 表示推理成功

  // 占位返回，实际项目请删除替换
  result.item_id = 1;
  result.freshness_level = 2;
  result.confidence = 0.92f;
  return true;
}
