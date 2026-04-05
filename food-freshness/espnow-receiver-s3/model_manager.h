/**
 * 模型管理 - SD卡加载/串口升级
 * 功能：
 * - 从SD卡加载训练好的模型
 * - 通过串口接收新模型，保存到SD卡
 * - 无需重新编译烧录，就能更新模型
 */

#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <Arduino.h>
#include <SD.h>
#include <SPI.h>

#if defined(ESP32) && !defined(VSPI)
#define VSPI VSPI
#endif

// 默认模型文件名
#define DEFAULT_MODEL_NAME "food_freshness.bin"
#define MAX_MODEL_SIZE  (128 * 1024)  // 最大128KB，足够我们的模型

class ModelManager {
public:
    ModelManager(int csPin = 5);
    ~ModelManager();

    // 初始化SD卡
    bool begin();

    // 检查SD卡上是否有模型
    bool hasModel(const char* filename = DEFAULT_MODEL_NAME);

    // 加载模型从SD卡到内存
    // 返回模型数据大小，0表示加载失败
    size_t loadModelFromSD(const char* filename, uint8_t* buffer, size_t maxBufferSize);

    // 通过串口接收新模型，保存到SD卡
    // 调用后会阻塞直到接收完成或超时
    // 返回true表示接收保存成功
    bool receiveAndSaveModel(Stream &stream, const char* filename = DEFAULT_MODEL_NAME, 
                             int timeoutSeconds = 60);

    // 获取最后错误信息
    const char* getLastError() { return _lastError; }

private:
    int _csPin;
    bool _sdInitialized = false;
    char _lastError[128];
    SPIClass *_spi = nullptr;
};

#endif // MODEL_MANAGER_H
