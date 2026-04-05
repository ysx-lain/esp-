/**
 * 模型管理 - Flash分区存储模型
 * 功能：
 * - 使用自定义flash分区存储模型（比SD卡更快更稳定）
 * - 通过串口接收新模型，写入flash分区
 * - 接收完成后直接生效，无需重启ESP32
 * - 需要在 partitions.csv 中添加模型分区
 * - 内置TFLite Micro推理支持
 * 适配：chirale/TensorFlowLite_ESP32 库
 */

#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include <Arduino.h>
#include <esp_partition.h>
#include <esp_flash.h>
#include <TensorFlowLite_ESP32.h>

// chirale library puts everything in global namespace
// no need for tflite namespace

// 默认模型最大大小 256KB足够我们的CNN模型
#define MAX_MODEL_SIZE  (256 * 1024)

class ModelManager {
public:
    ModelManager() {
        _lastError[0] = '\0';
        _partition = nullptr;
        _model_buffer = nullptr;
        _interpreter = nullptr;
        _error_reporter = nullptr;
        _initialized = false;
    }

    ~ModelManager() {
    }

    // 初始化 - 查找自定义model分区，加载模型
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

        // 如果分区中有模型，加载它
        if (hasModel()) {
            if (!loadModel()) {
                Serial.printf("⚠️ 加载模型失败: %s\n", getLastError());
                return false;
            }
            Serial.printf("📦 模型加载完成，大小: %zu bytes\n", getModelSize());
            _initialized = true;
        } else {
            Serial.println("⚠️ 分区中没有找到模型，使用内置模型");
            Serial.println("发送 'update model' 开始串口升级模型");
            _initialized = false;
        }

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
        esp_err_t err = esp_partition_read(_partition, 0, &size, sizeof(size_t));
        if (err != ESP_OK) {
            return 0;
        }
        return size;
    }

    // 检查是否有模型（大小>0）
    bool hasModel() {
        return getModelSize() > 0 && getModelSize() <= MAX_MODEL_SIZE;
    }

    // 加载模型从flash到内存，初始化tflite
    bool loadModel() {
        if (!_partition) {
            strncpy(_lastError, "No model partition", sizeof(_lastError)-1);
            return false;
        }
        size_t storedSize = getModelSize();
        if (storedSize == 0) {
            strncpy(_lastError, "No model stored", sizeof(_lastError)-1);
            return false;
        }
        if (storedSize > MAX_MODEL_SIZE) {
            strncpy(_lastError, "Model too large for buffer", sizeof(_lastError)-1);
            return false;
        }

        // 分配模型缓冲区 - 动态分配
        _model_buffer = (uint8_t*)heap_caps_malloc(storedSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!_model_buffer) {
            strncpy(_lastError, "malloc failed for model buffer", sizeof(_lastError)-1);
            return false;
        }

        // 跳过前4字节（存储大小），读取模型数据
        esp_err_t err = esp_partition_read(_partition, sizeof(size_t), _model_buffer, storedSize);
        if (err != ESP_OK) {
            strncpy(_lastError, esp_err_to_name(err), sizeof(_lastError)-1);
            free(_model_buffer);
            _model_buffer = nullptr;
            return false;
        }

        // 初始化TensorFlow Lite Micro
        const Model* model = GetModel(_model_buffer);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            strncpy(_lastError, "Model schema version mismatch", sizeof(_lastError)-1);
            free(_model_buffer);
            _model_buffer = nullptr;
            return false;
        }

        // 注册所有操作
        static AllOpsResolver resolver;

        // 内存分配
        constexpr size_t tensorArenaSize = 100000; // 100KB足够小模型
        _tensor_arena = (uint8_t*)heap_caps_malloc(tensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!_tensor_arena) {
            strncpy(_lastError, "malloc failed for tensor arena", sizeof(_lastError)-1);
            free(_model_buffer);
            _model_buffer = nullptr;
            return false;
        }

        // 创建错误reporter
        static MicroErrorReporter microErrorReporter;
        _error_reporter = &microErrorReporter;

        // 创建解释器
        _interpreter = new MicroInterpreter(model, resolver, _tensor_arena, tensorArenaSize, _error_reporter);
        TfLiteStatus status = _interpreter->AllocateTensors();
        if (status != kTfLiteOk) {
            strncpy(_lastError, "AllocateTensors failed", sizeof(_lastError)-1);
            free(_model_buffer);
            free(_tensor_arena);
            _model_buffer = nullptr;
            _tensor_arena = nullptr;
            _interpreter = nullptr;
            return false;
        }

        _initialized = true;
        Serial.println("✅ TFLite模型初始化完成");
        return true;
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

        // 使用静态buffer，不需要malloc，永远不会失败
        // SRAM overflow 修复: 放在flash中只读
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
            strncpy(_lastError, "No data received within timeout", sizeof(_lastError)-1);
            return false;
        }

        if (bytesReceived >= MAX_MODEL_SIZE) {
            strncpy(_lastError, "Model too large, increase MAX_MODEL_SIZE", sizeof(_lastError)-1);
            return false;
        }

        // 写入flash: 先写大小，再写数据
        size_t size = bytesReceived;
        esp_err_t err;

        err = esp_partition_erase_range(_partition, 0, _partition->size);
        if (err != ESP_OK) {
            strncpy(_lastError, esp_err_to_name(err), sizeof(_lastError)-1);
            return false;
        }
        Serial.printf("⚡ 分区擦除完成\n");

        err = esp_partition_write(_partition, 0, &size, sizeof(size_t));
        if (err != ESP_OK) {
            strncpy(_lastError, esp_err_to_name(err), sizeof(_lastError)-1);
            return false;
        }

        err = esp_partition_write(_partition, sizeof(size_t), tempBuffer, bytesReceived);
        if (err != ESP_OK) {
            strncpy(_lastError, esp_err_to_name(err), sizeof(_lastError)-1);
            return false;
        }

        Serial.printf("✅ 模型写入flash完成: %zu bytes\n", bytesReceived);

        // 如果之前有加载模型，释放内存重新加载
        if (_interpreter) {
            delete _interpreter;
            _interpreter = nullptr;
        }
        if (_model_buffer) {
            free(_model_buffer);
            _model_buffer = nullptr;
        }
        if (_tensor_arena) {
            free(_tensor_arena);
            _tensor_arena = nullptr;
        }

        // 重新加载新模型
        bool ok = loadModel();
        if (ok) {
            Serial.println("🎉 新模型已加载并生效，无需重启！");
        } else {
            Serial.printf("⚠️ 新模型写入成功，但加载失败: %s\n", getLastError());
        }

        return ok;
    }

    // TFLite API 包装
    int getInputSize() {
        if (!_interpreter || !_initialized) return 0;
        return _interpreter->inputs().size;
    }

    int getOutputSize() {
        if (!_interpreter || !_initialized) return 0;
        return _interpreter->outputs().size;
    }

    bool setInput(int index, float value) {
        if (!_interpreter || !_initialized || index >= _interpreter->inputs().size) return false;
        _interpreter->typed_input<float>(value, &index);
        return true;
    }

    float getOutput(int index) {
        if (!_interpreter || !_initialized || index >= _interpreter->outputs().size) return 0.0f;
        return _interpreter->typed_output<float>(index);
    }

    bool invoke() {
        if (!_interpreter || !_initialized) return false;
        TfLiteStatus status = _interpreter->Invoke();
        return status == kTfLiteOk;
    }

    bool isInitialized() { return _initialized; }

    // 获取最后错误信息
    const char* getLastError() { return _lastError; }

    // 获取模型存储路径（用于debug）
    const char* getModelPath() {
        if (!_partition) return "none";
        return _partition->label;
    }

private:
    char _lastError[128];
    const esp_partition_t *_partition;
    uint8_t *_model_buffer;
    uint8_t *_tensor_arena;
    MicroInterpreter *_interpreter;
    MicroErrorReporter *_error_reporter;
    bool _initialized;
};

#endif // MODEL_MANAGER_H
