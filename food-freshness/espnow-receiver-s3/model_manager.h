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
    ModelManager(int csPin = 15) : _csPin(csPin) {
        _lastError[0] = '\0';
    }

    ~ModelManager() {
        if (_spi) {
            delete _spi;
        }
    }

    // 初始化SD卡
    bool begin() {
        // 自定义硬件SPI引脚 SCK=12, MISO=14, MOSI=13, CS=_csPin
        _spi = new SPIClass();
        _spi->begin(12, 14, 13, _csPin); // SCK, MISO, MOSI, CS
        pinMode(_csPin, OUTPUT);
        digitalWrite(_csPin, HIGH);

        if (!SD.begin(_csPin, *_spi)) {
            strncpy(_lastError, "SD card init failed", sizeof(_lastError)-1);
            return false;
        }

        _sdInitialized = true;
        return true;
    }

    // 检查SD卡上是否有模型
    bool hasModel(const char* filename = DEFAULT_MODEL_NAME) {
        if (!_sdInitialized) {
            strncpy(_lastError, "SD not initialized", sizeof(_lastError)-1);
            return false;
        }
        // 确保路径正确
        char path[64];
        if (filename[0] != '/') {
            snprintf(path, sizeof(path), "/%s", filename);
        } else {
            strncpy(path, filename, sizeof(path)-1);
        }
        File f = SD.open(path, FILE_READ);
        if (!f) {
            strncpy(_lastError, "File not found", sizeof(_lastError)-1);
            return false;
        }
        size_t size = f.size();
        f.close();
        return size > 0 && size < MAX_MODEL_SIZE;
    }

    // 获取模型大小（字节）
    size_t getModelSize(const char* filename = DEFAULT_MODEL_NAME) {
        if (!_sdInitialized) return 0;
        char path[64];
        if (filename[0] != '/') {
            snprintf(path, sizeof(path), "/%s", filename);
        } else {
            strncpy(path, filename, sizeof(path)-1);
        }
        File f = SD.open(path, FILE_READ);
        if (!f) return 0;
        size_t size = f.size();
        f.close();
        return size;
    }

    // 获取模型存储路径
    const char* getModelPath(const char* filename = DEFAULT_MODEL_NAME) {
        static char path[64];
        if (filename[0] != '/') {
            snprintf(path, sizeof(path), "/%s", filename);
        } else {
            strncpy(path, filename, sizeof(path)-1);
        }
        return path;
    }

    // 加载模型从SD卡到内存
    // 返回模型数据大小，0表示加载失败
    size_t loadModelFromSD(const char* filename, uint8_t* buffer, size_t maxBufferSize) {
        if (!_sdInitialized) {
            strncpy(_lastError, "SD not initialized", sizeof(_lastError)-1);
            return 0;
        }
        char path[64];
        if (filename[0] != '/') {
            snprintf(path, sizeof(path), "/%s", filename);
        } else {
            strncpy(path, filename, sizeof(path)-1);
        }
        File f = SD.open(path, FILE_READ);
        if (!f) {
            strncpy(_lastError, "Cannot open file for reading", sizeof(_lastError)-1);
            return 0;
        }
        size_t bytesRead = 0;
        while (f.available() && bytesRead < maxBufferSize) {
            buffer[bytesRead] = f.read();
            bytesRead++;
        }
        f.close();
        Serial.printf("Loaded model %zu bytes from SD\n", bytesRead);
        return bytesRead;
    }

    // 通过串口接收新模型，保存到SD卡
    // 调用后会阻塞直到接收完成或超时
    // 返回true表示接收保存成功
    bool receiveAndSaveModel(Stream &stream, const char* filename = DEFAULT_MODEL_NAME, 
                             int timeoutSeconds = 60) {
        if (!_sdInitialized) {
            strncpy(_lastError, "SD not initialized", sizeof(_lastError)-1);
            return false;
        }

        // 确保路径正确，ESP32 SD卡根目录需要加/
        char path[64];
        if (filename[0] != '/') {
            snprintf(path, sizeof(path), "/%s", filename);
        } else {
            strncpy(path, filename, sizeof(path)-1);
        }

        // 删除旧文件
        if (SD.exists(path)) {
            SD.remove(path);
        }

        File f = SD.open(path, FILE_WRITE);
        if (!f) {
            strncpy(_lastError, "Cannot create file for writing", sizeof(_lastError)-1);
            return false;
        }

        Serial.println("\n=== 开始接收模型数据 ===");
        Serial.println("请从电脑发送二进制模型文件...");
        Serial.printf("超时: %d 秒\n", timeoutSeconds);

        unsigned long start = millis();
        size_t bytesReceived = 0;

        while (millis() - start < (unsigned long)timeoutSeconds * 1000) {
            if (stream.available()) {
                while (stream.available() && bytesReceived < MAX_MODEL_SIZE) {
                    uint8_t b = stream.read();
                    f.write(b);
                    bytesReceived++;
                    if (bytesReceived % 1024 == 0) {
                        Serial.printf("Received %zu KB...\n", bytesReceived / 1024);
                    }
                }
                start = millis();  // 收到数据就重置超时
            }
            delay(1);
        }

        f.close();
        Serial.printf("\n=== 接收完成 ===\n");
        Serial.printf("Total: %zu bytes saved to SD: %s\n", bytesReceived, path);

        if (bytesReceived == 0) {
            strncpy(_lastError, "No data received within timeout", sizeof(_lastError)-1);
            SD.remove(path);
            return false;
        }

        if (bytesReceived >= MAX_MODEL_SIZE) {
            strncpy(_lastError, "Model too large, increase MAX_MODEL_SIZE", sizeof(_lastError)-1);
            SD.remove(path);
            return false;
        }

        return true;
    }

    // 获取最后错误信息
    const char* getLastError() { return _lastError; }

private:
    int _csPin;
    bool _sdInitialized = false;
    char _lastError[128];
    SPIClass *_spi = nullptr;
};

#endif // MODEL_MANAGER_H
