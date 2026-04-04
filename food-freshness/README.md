# 智能食材新鲜度监测系统 - 边缘AI嵌入式方案

## 硬件架构

| 设备 | 功能 |
|------|------|
| ESP32-S3 (发送端) | 仅负责采集多路气体传感器数据（ADS1115 + MHZ19C + BME680），通过 ESP-NOW 发送给接收端 |
| ESP32-S3 (接收端) | 运行本地AI推理（基于TFLite），融合气体数据+MaixCam图像信息，输出新鲜度判断，控制紫外线灯/三色LED，发送指令给P4控制制冷片 |
| ESP32-P4 | 接收S3指令，直接控制制冷片（可选，也可S3直驱） |
| MaixCam Pro | 图像采集，物品识别计数，校准气体数据 |
| AX630C (M5Stack) | 本地服务器，负责神经网络训练 + TTS语音播报 |

## 目录结构

```
food-freshness/
├── espnow-sender-s3.ino       # ESP32-S3 发送端（气体采集）
├── espnow-receiver-s3.ino     # ESP32-S3 接收端（AI推理 + 控制）
├── sensor_data_logger.py      # Python PC端数据采集工具，转存为CSV
└── README.md
```

## 数据采集流程

1. 发送端采集多路气体数据 → ESP-NOW → 接收端
2. 接收端格式化打印数据 → USB串口 → PC
3. `sensor_data_logger.py` 读取串口，自动解析数据块，保存为CSV文件
4. CSV格式：`Time_s, Odor, HCHO, CO, VOC, CO2, Label`，Label用于标注物品种类/新鲜度

## 编译和配置

### 发送端 (espnow-sender-s3.ino)
- 修改 `receiverMac` 为你的接收端MAC地址
- 传感器引脚已定义在开头，根据你的布线修改
- 需要安装库：DFRobot_ADS1115, MHZ19, bme68xLibrary

### 接收端 (espnow-receiver-s3.ino)
- 修改 `sensorMac` 为你的发送端MAC地址
- 制冷控制方式：`#define ENABLE_P4_CONTROL 0/1`
  - `0` = S3直接PWM控制制冷片
  - `1` = UART发送指令给ESP32-P4
- 修改对应引脚定义

## 依赖库

### Arduino/ESP-IDF
- DFRobot_ADS1115
- MHZ19
- bme68xLibrary
- ESP-NOW (ESP32核心内置)

### Python
```bash
pip install pyserial
```
（tkinter 一般Python自带）

## 使用方法

1. 烧录发送端和接收端固件
2. 打开串口监视器确认连接正常
3. 运行 `python sensor_data_logger.py`
4. 选择串口号，设置采样间隔，填写标签（比如："apple-fresh"）
5. 点击开始采集，数据自动保存到CSV

## 通信协议

### ESP-NOW 数据包（均为 packed 对齐）

| 类型 | 结构 |
|------|------|
| SENSOR | `dataType + 气体数据 + 状态 + 时间戳` |
| WARMUP | `dataType + 剩余秒 + 时间戳` |
| COMMAND | `dataType + 32字节命令 + 时间戳` |

### S3 → P4 串口指令（P4控制模式）

```c
typedef struct __attribute__((packed)) {
  uint8_t magic[2];   // 0x55 0xAA
  uint8_t cmd;        // 0=关, 1=开, 2=设置功率
  uint8_t power;      // 0-100%
  uint8_t checksum;   // magic[0] ^ magic[1] ^ cmd ^ power
} P4CoolerCmd;
```

## 后续开发路线

1. 数据采集 → CSV整理 → 在AX630C上训练神经网络
2. 模型量化转换 → `.tflite` → 部署到接收端S3
3. 集成MaixCam图像识别结果，做多模态融合
4. 对接AX630C TTS，实现语音播报结果

## 作者

ysx-lain <1875323156@qq.com>
