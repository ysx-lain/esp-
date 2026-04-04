# 智能食材新鲜度监测系统 - 边缘AI嵌入式方案

## 硬件架构

| 设备 | 功能 |
|------|------|
| ESP32-S3 (发送端) | 仅负责采集多路气体传感器数据（ADS1115 + MHZ19C + BME680），通过 ESP-NOW 发送给接收端 |
| ESP32-S3 (接收端) | 运行本地AI推理（基于TFLite Micro），融合气体数据+MaixCam图像信息，输出新鲜度判断，控制紫外线灯，发送指令给P4控制制冷片 |
| ESP32-P4 | 接收S3指令，直接控制制冷片（可选，也可S3直驱） |
| MaixCam Pro | 图像采集，物品识别计数，校准气体数据 |
| AX630C (M5Stack) / PC | 本地训练服务器，负责神经网络训练 + 可选TTS语音播报 |

## 目录结构

```
food-freshness/
├── README.md
├── espnow-sender-s3.ino       # ESP32-S3 发送端固件（气体采集）
├── espnow-receiver-s3.ino     # ESP32-S3 接收端固件（AI推理 + 控制）
├── sensor_data_logger.py      # PC端串口采集保存CSV（独立使用）
├── train_gui.py               # 🖥️ 一体化GUI训练工具（采集连接 → 训练 → 导出C头文件一步完成）
└── training/                  # 命令行训练脚本
    ├── train_classifier.py     # MLP版本（单帧输入）
    └── train_classifier_cnn.py # 1D-CNN版本（滑动窗口时间序列）
```

## 完整工作流程

```
硬件采集:
  ESP32-S3发送端 → 采气 → ESP-NOW → ESP32-S3接收端 → USB串口 → PC
                             ↓
PC端GUI:  python train_gui.py
  1. 连接串口 → 设置标签 → 开始采集 → 自动保存CSV
  2. 设置训练参数 → 点击开始训练
  3. 自动完成滑动窗口预处理 → 训练 → INT8量化 → 导出 `.h` C数组头文件
                             ↓
部署:
  把生成的 `.h` 文件复制到Arduino工程，include进去就能推理
```

## 快速开始

### 环境依赖

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn pyserial tkinter
```

### 一体化GUI使用（推荐）

```bash
python train_gui.py
```

1. 选择串口号 → 点击 **连接**
2. 填写标签（格式：`category_freshness` 例如 `apple_fresh`）
3. 点击 **开始采集**，数据自动保存到数据目录
4. 采集完不同类别，修改标签继续采集
5. 调整训练参数（窗口大小/学习率等）
6. 点击 **开始训练**，等待完成，自动输出模型头文件

### 命令行训练

```bash
cd training

# MLP版本（单帧）
python train_classifier.py --data_dir ../../sensor_data

# 1D-CNN版本（滑动窗口，推荐）
python train_classifier_cnn.py --data_dir ../../sensor_data --window_size 5 --window_step 2
```

## Arduino编译配置

### 发送端 (espnow-sender-s3.ino)
- 修改 `receiverMac` 为你的接收端MAC地址
- 传感器引脚已定义在开头，根据你的布线修改
- 需要安装库：`DFRobot_ADS1115`, `MHZ19`, `bme68xLibrary`

### 接收端 (espnow-receiver-s3.ino)
- 修改 `sensorMac` 为你的发送端MAC地址
- 制冷控制方式：`#define ENABLE_P4_CONTROL 0/1`
  - `0` = S3直接PWM控制制冷片
  - `1` = UART发送指令给ESP32-P4
- 修改对应引脚定义
- 需要安装：`TensorFlowLite for Arduino` 库

## 参数推荐

| 数据量 | window_size | window_step | dropout | lr |
|--------|-------------|-------------|---------|----|
| < 2000样本 | 5 | 2 | 0.2 | 0.001 |
| 2000-10000样本 | 7 | 3 | 0.25 | 0.0008 |
| > 10000样本 | 9 | 4 | 0.3 | 0.0005 |

如果类别超过15种，稍微增大卷积通道数即可，模型仍能保持在100KB以内。

## CSV格式要求

程序自动读取，格式：
```
Time_s,Odor,HCHO,CO,VOC,CO2,Label
120,0.35,0.12,0.5,1.2,412,apple_fresh
125,0.38,0.13,0.5,1.3,415,apple_fresh
...
```

Label 格式：`物品种类_新鲜度等级`，比如 `apple_fresh`, `apple_stale`, `banana_good` 等。

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

## 输出文件

训练完成后输出：

| 文件 | 用途 |
|------|------|
| `xxx.tflite` | INT8量化模型文件 |
| `xxx.h` | **Arduino C数组头文件**，直接放到工程 `#include` 就能用 |
| `xxx_params.npz` | 归一化参数+类别列表（Python用） |
| `xxx_confusion.png` | 混淆矩阵可视化 |
| `xxx_training.png` | 训练曲线 |

## 后续开发计划

1. ~~数据采集流程自动化~~ ✅
2. ~~一体化GUI训练~~ ✅
3. ESP32-S3 TFLite Micro推理集成代码
4. MaixCam 图像识别代码
5. 多模态融合（气体+图像）
6. AX630C TTS对接

## 作者

ysx-lain <1875323156@qq.com>
