/**
 * 模型管理 - Flash分区存储模型
 * 功能：
 * - 使用自定义flash分区存储模型（比SD卡更快更稳定）
 * - 通过串口接收新模型，写入flash分区
 * - 接收完成后直接生效，无需重启ESP32
 * - 需要在 partitions.csv 中添加模型分区
 */

#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <Arduino.h>
#include <esp_partition.h>
#include <esp_flash.h>

// 默认模型最大大小 256KB足够我们的CNN模型
#define MAX_MODEL_SIZE  (256 * 1024)

class ModelManager {
public:
    ModelManager() {
        _lastError[0] = '\0';
        _partition = nullptr;
    }

    ~ModelManager() {
    }

    // 初始化 - 查找自定义model分区
    bool begin() {
        // 查找标签为"model"的分区
        // esp_partition_find_first directly returns a pointer to the partition structure
        const esp_partition_t *found = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
        if (found) {
            _partition = found;
        } else {
            // 尝试找任何足够大的数据分区
            esp_partition_iterator_t it = esp_partition_find(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, NULL);
            while (it) {
                found = esp_partition_get(it);
                if (found->size >= MAX_MODEL_SIZE) {
                    _partition = found;
                    break;
                }
                it = esp_partition_next(it);
            }
        }

        if (!_partition) {
            strncpy(_lastError, "No model partition found, check partitions.csv", sizeof(_lastError)-1);
            return false;
        }

        Serial.printf("✅ 找到模型分区: %s, 大小: %d bytes (%.1f KB)\n", 
            _partition->label, _partition->size, (float)_partition->size / 1024);
        return true;
    }

    // 检查是否有可用分区
    bool hasModelPartition() {
        return _partition != nullptr;
    }

    // 获取分区中存储的模型大小
    // 第一4字节存储实际模型大小
    size_t getModelSize() {
        if (!_partition) return 0;
        size_t size = 0;
        esp_err_t err = esp_partition_read(_partition, 0, &size, sizeof(size));
        if (err != ESP_OK) {
            return 0;
        }
        return size;
    }

    // 检查是否有模型（大小>0）
    bool hasModel() {
        return getModelSize() > 0 && getModelSize() <= MAX_MODEL_SIZE;
    }

    // 读取整个模型到buffer
    // 返回实际读取大小，0表示失败
    size_t readModel(uint8_t* buffer, size_t maxSize) {
        if (!_partition) {
            strncpy(_lastError, "No model partition", sizeof(_lastError)-1);
            return 0;
        }
        size_t storedSize = getModelSize();
        if (storedSize == 0) {
            strncpy(_lastError, "No model stored", sizeof(_lastError)-1);
            return 0;
        }
        if (storedSize > maxSize) {
            strncpy(_lastError, "Model too large for buffer", sizeof(_lastError)-1);
            return 0;
        }

        // 跳过前4字节（存储大小），读取模型数据
        esp_err_t err = esp_partition_read(_partition, sizeof(size_t), buffer, storedSize);
        if (err != ESP_OK) {
            strncpy(_lastError, esp_err_to_name(err), sizeof(_lastError)-1);
            return 0;
        }

        return storedSize;
    }

    // 通过串口接收新模型，写入flash分区
    // 返回true表示接收保存成功，成功后直接生效无需重启
    bool receiveAndWriteModel(Stream &stream, int timeoutSeconds = 60) {
        if (!_partition) {
            strncpy(_lastError, "No model partition", sizeof(_lastError)-1);
            return false;
        }

        Serial.println("\n=== 开始接收模型数据 ===");
        Serial.println("请从电脑发送二进制模型文件...");
        Serial.printf("超时: %d 秒，最大: %d bytes\n", timeoutSeconds, MAX_MODEL_SIZE);

        // 先分配临时buffer存放接收的数据
        uint8_t *tempBuffer = (uint8_t *)malloc(MAX_MODEL_SIZE);
        if (!tempBuffer) {
            strncpy(_lastError, "malloc failed for temp buffer", sizeof(_lastError)-1);
            return false;
        }

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
            strncpy(_lastError, "No data received within timeout", sizeof(_lastError)-1);
            free(tempBuffer);
            return false;
        }

        if (bytesReceived >= MAX_MODEL_SIZE) {
            strncpy(_lastError, "Model too large, increase MAX_MODEL_SIZE", sizeof(_lastError)-1);
            free(tempBuffer);
            return false;
        }

        // 写入flash: 先写大小，再写数据
        size_t size = bytesReceived;
        esp_err_t err;

        err = esp_partition_erase_range(_partition, 0, _partition->size);
        if (err != ESP_OK) {
            strncpy(_lastError, esp_err_to_name(err), sizeof(_lastError)-1);
            free(tempBuffer);
            return false;
        }
        Serial.printf("⚡ 分区擦除完成\n");

        err = esp_partition_write(_partition, 0, &size, sizeof(size));
        if (err != ESP_OK) {
            strncpy(_lastError, esp_err_to_name(err), sizeof(_lastError)-1);
            free(tempBuffer);
            return false;
        }

        err = esp_partition_write(_partition, sizeof(size), tempBuffer, bytesReceived);
        if (err != ESP_OK) {
            strncpy(_lastError, esp_err_to_name(err), sizeof(_lastError)-1);
            free(tempBuffer);
            return false;
        }

        free(tempBuffer);

        Serial.printf("✅ 模型写入flash完成: %zu bytes\n", bytesReceived);
        Serial.println("🎉 新模型已生效，无需重启！");

        return true;
    }

    // 获取最后错误信息
    const char* getLastError() { return _lastError; }

private:
    char _lastError[128];
    const esp_partition_t *_partition;
};

#endif // MODEL_MANAGER_H
