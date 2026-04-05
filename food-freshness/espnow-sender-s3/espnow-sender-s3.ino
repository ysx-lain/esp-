/**
 * 综合气体监测系统 - 发送端 (ESP32-S3)
 * 支持远程命令控制，最小间隔 1 秒，高精度定时
 * 接收端 MAC: 30:ED:A0:A9:71:20
 */

#include <DFRobot_ADS1115.h>
#include <Wire.h>
#include <MHZ19.h>
#include <SPI.h>
#include <bme68xLibrary.h>
#include <esp_now.h>
#include <WiFi.h>

// ==================== 引脚定义 ====================
#define I2C_SDA 17
#define I2C_SCL 18
#define MHZ_RX 15
#define MHZ_TX 16
#define BME_MOSI 10
#define BME_MISO 11
#define BME_SCK 12
#define BME_CS 5
#define VCC 3.3

// ==================== 传感器参数 ====================
const float RL[4] = {4700.0, 10000.0, 4700.0, 4700.0};
const float A[4] = {2.357, 0.351, 4.082, 2.357};
const float B_const = 0.5;

// ==================== 状态位 ====================
#define STATUS_ADS1115_OK 0x01
#define STATUS_MHZ19C_OK 0x02
#define STATUS_BME680_OK 0x04

// ==================== 数据包类型 ====================
#define PKT_TYPE_SENSOR 0x01
#define PKT_TYPE_WARMUP 0x02
#define PKT_TYPE_COMMAND 0x03

// ==================== ESP-NOW ====================
uint8_t receiverMac[] = {0x30, 0xED, 0xA0, 0xA9, 0x71, 0x20};
#define ESP_NOW_MAX_RETRY 2

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

// ==================== 全局变量 ====================
DFRobot_ADS1115 ads(&Wire);
HardwareSerial mhSerial(1);
MHZ19 mhz19;
SPIClass spi(HSPI);
Bme68x bme;

float R0[4] = {0,0,0,0};
bool r0Calibrated[4] = {false, false, false, false};

// 高精度定时（微秒）
unsigned long lastSampleMicros = 0;
unsigned long sampleIntervalMicros = 5UL * 1000000UL; // 默认5秒，可远程修改

uint8_t sensorStatus = 0;
float lastEnvTemp = 0.0;
float lastHumidity = 0.0;
bool bmeValid = false;

// ESP-NOW 重发
int sendRetryCount = 0;
bool pendingSend = false;
uint8_t pendingData[sizeof(SensorData)];
size_t pendingLen = 0;

// 远程命令标志
volatile bool skipWarmupRequested = false;

// BME680 读取节流（避免每次采样都阻塞150ms）
unsigned long lastBmeReadTime = 0;
const unsigned long bmeReadInterval = 10000; // 每10秒读一次

// ==================== 函数声明 ====================
float voltageToPPM(float Vout_mV, int channel);
void calibrateR0();
bool initBME680();
bool readBME680TempHumid(float &temp, float &hum);
void onSent(const uint8_t *mac_addr, esp_now_send_status_t status);
void sendData(const uint8_t* data, size_t len);
void sendSensorData(SensorData &data);
void sendWarmupStatus(uint16_t remainingSec);
void onReceiveCommand(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len);
void processRemoteCommand(const char* cmdStr);
void performSampling();

// ==================== setup ====================
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  Serial.println("\n气体监测系统发送端 (高精度版，支持1秒间隔)");

  Wire.begin(I2C_SDA, I2C_SCL);

  // ADS1115
  ads.setAddr_ADS1115(ADS1115_IIC_ADDRESS0);
  ads.setGain(eGAIN_TWOTHIRDS);
  ads.setMode(eMODE_SINGLE);
  ads.setRate(eRATE_128);
  ads.setOSMode(eOSMODE_SINGLE);
  ads.init();
  delay(20);
  if (ads.checkADS1115()) {
    sensorStatus |= STATUS_ADS1115_OK;
    Serial.println("ADS1115 OK");
  } else {
    Serial.println("ADS1115 失败");
  }

  // MH-Z19C（上电延时 + 重试）
  delay(3000);
  mhSerial.begin(9600, SERIAL_8N1, MHZ_RX, MHZ_TX);
  mhz19.begin(mhSerial);
  bool mhzOk = false;
  for (int i = 0; i < 10; i++) {
    delay(500);
    int co2 = mhz19.getCO2();
    if (co2 > 0) {
      sensorStatus |= STATUS_MHZ19C_OK;
      Serial.printf("MH-Z19C OK, CO2=%d ppm\n", co2);
      mhzOk = true;
      break;
    }
    Serial.printf("MH-Z19C 尝试 %d/10\n", i+1);
  }
  if (!mhzOk) Serial.println("MH-Z19C 失败");

  // BME680
  if (initBME680()) {
    sensorStatus |= STATUS_BME680_OK;
    Serial.println("BME680 OK");
    float t, h;
    if (readBME680TempHumid(t, h)) {
      lastEnvTemp = t;
      lastHumidity = h;
      bmeValid = true;
      lastBmeReadTime = millis();
    }
  } else {
    Serial.println("BME680 失败");
  }

  // ESP-NOW
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW 初始化失败");
    while (1) delay(100);
  }
  esp_now_register_send_cb(onSent);
  esp_now_register_recv_cb(onReceiveCommand);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("添加接收端失败");
    return;
  }

  // 非阻塞预热（不影响后续定时）
  Serial.println("预热180秒，可发送 'skip warmup' 跳过");
  unsigned long warmStart = millis();
  int lastRemaining = 180;
  while (true) {
    if (skipWarmupRequested) break;
    unsigned long elapsed = (millis() - warmStart) / 1000;
    if (elapsed >= 180) break;
    int remaining = 180 - elapsed;
    if (remaining != lastRemaining) {
      lastRemaining = remaining;
      Serial.printf("预热剩余: %d 秒\n", remaining);
      if (remaining % 5 == 0 || remaining == 180) {
        sendWarmupStatus(remaining);
      }
    }
    delay(100); // 非阻塞，可响应命令
  }
  sendWarmupStatus(0);
  Serial.println("预热完成");

  // 校准 R0
  if (sensorStatus & STATUS_ADS1115_OK) {
    calibrateR0();
  } else {
    Serial.println("跳过 R0 校准");
  }

  lastSampleMicros = micros();
  Serial.printf("当前采样间隔: %d 秒\n", sampleIntervalMicros / 1000000);
  Serial.println("Odor\tHCHO\tCO\tVOC\tCO2\tCO2_Temp\tEnv_Temp\tHumidity\tStatus\tTimestamp(ms)");
}

// ==================== 主循环（精确间隔） ====================
void loop() {
  unsigned long nowMicros = micros();
  if (nowMicros - lastSampleMicros >= sampleIntervalMicros) {
    lastSampleMicros = nowMicros;
    performSampling();
  }
  yield(); // 让出CPU，保证低延迟响应
}

// ==================== 采样与发送 ====================
void performSampling() {
  unsigned long nowMs = millis();

  // ADS1115
  float ppm[4] = {0,0,0,0};
  if (sensorStatus & STATUS_ADS1115_OK) {
    float volt_mV[4] = {
      (float)ads.readVoltage(0),
      (float)ads.readVoltage(1),
      (float)ads.readVoltage(2),
      (float)ads.readVoltage(3)
    };
    for (int i = 0; i < 4; i++) {
      if (r0Calibrated[i]) {
        ppm[i] = voltageToPPM(volt_mV[i], i);
        if (ppm[i] > 5000) ppm[i] = 5000;
      }
    }
  }

  // MH-Z19C（快速重试）
  int co2 = 0, co2_temp = 0;
  if (sensorStatus & STATUS_MHZ19C_OK) {
    for (int retry = 0; retry < 2; retry++) {
      co2 = mhz19.getCO2();
      if (co2 > 0) {
        co2_temp = mhz19.getTemperature();
        break;
      }
      delay(50);
    }
    if (co2 <= 0) co2 = 0;
  }

  // BME680（每10秒读一次，避免阻塞）
  float env_temp = lastEnvTemp;
  float humidity = lastHumidity;
  if (sensorStatus & STATUS_BME680_OK) {
    unsigned long now = millis();
    if (now - lastBmeReadTime >= bmeReadInterval) {
      lastBmeReadTime = now;
      float t, h;
      if (readBME680TempHumid(t, h)) {
        env_temp = t;
        humidity = h;
        lastEnvTemp = t;
        lastHumidity = h;
        bmeValid = true;
      } else if (!bmeValid) {
        env_temp = 0.0;
        humidity = 0.0;
      } else {
        env_temp = lastEnvTemp;
        humidity = lastHumidity;
      }
    } else {
      env_temp = lastEnvTemp;
      humidity = lastHumidity;
    }
  }

  // 串口打印
  Serial.printf("%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%.2f\t%.2f\t0x%02X\t%u\n",
    ppm[0], ppm[1], ppm[2], ppm[3],
    co2, co2_temp,
    env_temp, humidity,
    sensorStatus, nowMs);

  // 发送数据包
  SensorData data;
  data.dataType = PKT_TYPE_SENSOR;
  data.odor_ppm = ppm[0];
  data.hcho_ppm = ppm[1];
  data.co_ppm = ppm[2];
  data.voc_ppm = ppm[3];
  data.co2_ppm = (uint16_t)co2;
  data.co2_temp = (int16_t)co2_temp;
  data.env_temp = env_temp;
  data.humidity = humidity;
  data.sensor_status = sensorStatus;
  data.timestamp = nowMs;
  sendSensorData(data);
}

// ==================== 传感器底层函数 ====================
bool initBME680() {
  spi.begin(BME_SCK, BME_MISO, BME_MOSI, BME_CS);
  bme.begin(BME_CS, spi);
  if (bme.checkStatus() != BME68X_OK) return false;
  bme.setTPH(BME68X_OS_8X, BME68X_OS_NONE, BME68X_OS_2X);
  bme.setFilter(BME68X_FILTER_SIZE_3);
  bme.setHeaterProf(0, 0);
  return true;
}

bool readBME680TempHumid(float &temp, float &hum) {
  bme.setOpMode(BME68X_FORCED_MODE);
  uint32_t dur = bme.getMeasDur();
  delay(dur / 1000);
  if (dur % 1000) delayMicroseconds(dur % 1000);
  if (bme.fetchData()) {
    bme68xData data;
    bme.getData(data);
    temp = data.temperature;
    hum = data.humidity;
    return true;
  }
  return false;
}

void calibrateR0() {
  Serial.println("校准 R0（洁净空气）...");
  for (int i = 0; i < 4; i++) {
    for (int retry = 0; retry < 3; retry++) {
      float Vout_mV = ads.readVoltage(i);
      float Vout = Vout_mV / 1000.0;
      if (Vout <= 0.001) {
        Serial.printf("通道%d电压异常,重试\n", i);
        delay(500);
        continue;
      }
      float Rs = (VCC - Vout) / Vout * RL[i];
      R0[i] = Rs;
      r0Calibrated[i] = true;
      Serial.printf("通道%d R0=%.2f kΩ\n", i, Rs/1000.0);
      break;
    }
  }
}

float voltageToPPM(float Vout_mV, int ch) {
  if (Vout_mV <= 0) return 0;
  float Vout = Vout_mV / 1000.0;
  if (Vout <= 0.001) return 0;
  float Rs = (VCC - Vout) / Vout * RL[ch];
  float ratio = Rs / R0[ch];
  if (ratio <= 0) return 0;
  float ppm = pow(A[ch] / ratio, 1.0 / B_const);
  return (ppm > 5000) ? 5000 : ppm;
}

// ==================== ESP-NOW 发送 ====================
void sendData(const uint8_t* data, size_t len) {
  if (pendingSend) {
    Serial.println("发送中，跳过本次");
    return;
  }
  memcpy(pendingData, data, len);
  pendingLen = len;
  pendingSend = true;
  sendRetryCount = 0;
  esp_now_send(receiverMac, pendingData, len);
}

void sendSensorData(SensorData &data) {
  sendData((uint8_t*)&data, sizeof(SensorData));
}

void sendWarmupStatus(uint16_t remainingSec) {
  WarmupStatus warmup;
  warmup.dataType = PKT_TYPE_WARMUP;
  warmup.remainingSec = remainingSec;
  warmup.timestamp = millis();
  sendData((uint8_t*)&warmup, sizeof(WarmupStatus));
  if (remainingSec == 0) Serial.println("预热完成通知已发送");
  else Serial.printf("发送预热倒计时: %d 秒\n", remainingSec);
}

void onSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (!pendingSend) return;
  if (status == ESP_NOW_SEND_SUCCESS) {
    pendingSend = false;
    // Serial.println("发送成功");
  } else {
    sendRetryCount++;
    if (sendRetryCount <= ESP_NOW_MAX_RETRY) {
      esp_now_send(receiverMac, pendingData, pendingLen);
    } else {
      pendingSend = false;
      Serial.println("发送失败，放弃");
    }
  }
}

// ==================== 远程命令 ====================
void onReceiveCommand(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  if (len == sizeof(CommandPacket)) {
    CommandPacket cmd;
    memcpy(&cmd, incomingData, sizeof(cmd));
    if (cmd.dataType == PKT_TYPE_COMMAND) {
      Serial.printf("远程命令: %s\n", cmd.command);
      processRemoteCommand(cmd.command);
    }
  }
}

void processRemoteCommand(const char* cmdStr) {
  String cmd = String(cmdStr);
  cmd.trim();
  if (cmd.startsWith("set interval")) {
    int sec = cmd.substring(13).toInt();
    if (sec >= 1 && sec <= 3600) {
      sampleIntervalMicros = sec * 1000000UL;
      Serial.printf("采样间隔已设为 %d 秒\n", sec);
    } else {
      Serial.println("无效间隔，范围1~3600");
    }
  }
  else if (cmd.equalsIgnoreCase("skip warmup")) {
    skipWarmupRequested = true;
    Serial.println("跳过预热");
  }
  else if (cmd.equalsIgnoreCase("status")) {
    Serial.printf("间隔: %d 秒\n", sampleIntervalMicros / 1000000);
    String s = "";
    if (sensorStatus & STATUS_ADS1115_OK) s += "ADS1115 ";
    if (sensorStatus & STATUS_MHZ19C_OK) s += "MHZ19C ";
    if (sensorStatus & STATUS_BME680_OK) s += "BME680 ";
    if (s.length() == 0) s = "None";
    Serial.printf("传感器: %s\n", s.c_str());
  }
  else {
    Serial.println("未知命令");
  }
}
