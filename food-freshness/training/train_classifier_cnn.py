"""
食材新鲜度分类模型训练脚本 - 1D-CNN版本
功能：
- 自动扫描文件夹中所有CSV文件
- 滑动窗口数据增强，利用时间序列信息
- 标准化 + 自动划分数据集
- 训练轻量1D-CNN模型
- 量化导出INT8 TFLite模型
- 转换为C数组头文件，直接包含到Arduino工程

支持格式：
CSV列：Time_s, Odor, HCHO, CO, VOC, CO2, Label
Label列格式：category_freshness 例如：apple_fresh, banana_stale
"""

import os
import glob
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='训练食材新鲜度分类模型 (1D-CNN)')
    parser.add_argument('--data_dir', default='../../sensor_data', help='CSV数据文件夹路径')
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口大小（帧数）')
    parser.add_argument('--window_step', type=int, default=2, help='滑动步长')
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout比率')
    parser.add_argument('--output', default='../../food_freshness_cnn', help='输出模型名')
    return parser.parse_args()

def load_all_csv(data_dir):
    """自动加载文件夹中所有CSV文件"""
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    all_data = []
    for f in csv_files:
        print(f" 加载 {os.path.basename(f)} ...")
        df = pd.read_csv(f)
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n总共 {len(combined)} 条原始样本")
    return combined

def create_sliding_windows(X, y, window_size, step):
    """
    滑动窗口生成序列样本
    X: (n_samples, n_features)
    y: (n_samples,) 每个样本的标签
    返回: (n_windows, window_size, n_features), (n_windows,)
    """
    n_samples, n_features = X.shape
    windows = []
    labels = []
    
    for i in range(0, n_samples - window_size + 1, step):
        window = X[i:i+window_size, :]
        windows.append(window)
        # 窗口中心的标签作为整个窗口标签
        center_idx = i + window_size // 2
        labels.append(y[center_idx])
    
    return np.array(windows), np.array(labels)

def extract_features_labels(df, window_size, step):
    """提取特征和标签，创建滑动窗口"""
    # 特征：5个气体传感器通道
    feature_cols = ['Odor', 'HCHO', 'CO', 'VOC', 'CO2']
    X_raw = df[feature_cols].values
    
    # 标签：从Label列提取
    labels_raw = df['Label'].values
    
    # 去重得到类别列表
    unique_labels = sorted(list(set(labels_raw)))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y_raw = np.array([label_to_idx[l] for l in labels_raw])
    
    print(f"\n类别列表 ({len(unique_labels)} 类):")
    for l, i in label_to_idx.items():
        print(f"  [{i}] {l}")
    
    # 滑动窗口增强
    if window_size > 1:
        X_window, y_window = create_sliding_windows(X_raw, y_raw, window_size, step)
        print(f"\n滑动窗口增强: 窗口大小={window_size}, 步长={step}")
        print(f"生成 {X_window.shape[0]} 个窗口样本")
    else:
        X_window = X_raw[:, np.newaxis, :]  # (n, 1, 5)
        y_window = y_raw
    
    return X_window, y_window, unique_labels, label_to_idx

def build_model(input_shape, num_classes, dropout_rate):
    """构建轻量1D-CNN模型，适合ESP32部署"""
    # 输入形状: (window_size, n_features)
    model = Sequential([
        Conv1D(16, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv1D(32, kernel_size=2, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        Flatten(),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    return model

def plot_confusion_matrix(cm, class_names, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"混淆矩阵已保存: {output_path}")

def convert_to_tflite(model, output_path):
    """转换为INT8量化TFLite模型，适合ESP32部署"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # INT8量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # 保存模型
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"\n量化TFLite模型已保存: {output_path}, 大小: {size_kb:.1f} KB")
    
    if size_kb < 100:
        print("✅ 模型大小适合ESP32-S3部署")
    else:
        print("⚠️ 模型偏大，建议减小窗口大小或通道数")
    
    return tflite_model

def tflite_to_c_array(tflite_model, output_header_path, array_name="g_food_freshness_model"):
    """将TFLite模型转换为C数组头文件，供Arduino使用"""
    with open(output_header_path, 'w') as f:
        f.write("// Autogenerated by train_classifier_cnn.py\n")
        f.write("// DO NOT EDIT\n\n")
        f.write("#include <stddef.h>\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned int {array_name}_size = {len(tflite_model)};\n")
        f.write(f"const unsigned char {array_name}[] = {{\n  ")
        
        bytes_per_line = 16
        for i, byte in enumerate(tflite_model):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % bytes_per_line == 0:
                f.write("\n  ")
        
        if len(tflite_model) % bytes_per_line != 0:
            f.write("\n")
        
        f.write("};\n\n")
    
    print(f"C数组头文件已保存: {output_header_path}")
    print(f"在Arduino中直接: #include \"{os.path.basename(output_header_path)}\"")

def save_scaler_and_classes(scaler, class_names, output_path):
    """保存归一化参数和类别信息"""
    np.savez(output_path, 
            mean=scaler.mean_, 
            std=np.sqrt(scaler.var_),
            classes=np.array(class_names, dtype=object))
    print(f"归一化参数和类别已保存: {output_path}")

def main():
    args = parse_args()
    
    # 1. 加载数据
    print("=" * 60)
    print("食材新鲜度分类模型训练 (1D-CNN + 滑动窗口)")
    print("=" * 60)
    
    df = load_all_csv(args.data_dir)
    X_raw, y, class_names, label_to_idx = extract_features_labels(df, args.window_size, args.window_step)
    
    # 2. 数据预处理：标准化
    # 对每个特征独立标准化
    n_features = X_raw.shape[-1]
    scaler = StandardScaler()
    X_reshaped = X_raw.reshape(-1, n_features)
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X = X_scaled_reshaped.reshape(X_raw.shape)
    
    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"输入形状: {X_train.shape[1:]} (时间步 × 特征数)")
    
    # 4. 构建模型
    num_classes = len(class_names)
    model = build_model(X.shape[1:], num_classes, args.dropout)
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 5. 训练
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        f"{args.output}_best.h5", 
        monitor='val_accuracy', 
        save_best_only=True,
        verbose=1
    )
    
    print("\n开始训练...")
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # 6. 评估
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集准确率: {test_acc:.4f}")
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(cm, class_names, f"{args.output}_confusion.png")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output}_training.png", dpi=150)
    
    # 7. 导出TFLite模型
    print("\n" + "=" * 60)
    print("导出INT8量化TFLite模型")
    print("=" * 60)
    tflite_model = convert_to_tflite(model, f"{args.output}.tflite")
    
    # 8. 转换为C数组头文件
    tflite_bytes = list(tflite_model)
    tflite_to_c_array(tflite_bytes, f"{args.output}.h", "g_food_freshness_model")
    
    # 9. 保存归一化参数
    save_scaler_and_classes(scaler, class_names, f"{args.output}_params")
    
    print("\n✅ 训练完成！输出文件：")
    print(f"  {args.output}.tflite        - 量化模型文件")
    print(f"  {args.output}.h            - Arduino C数组头文件，直接包含使用")
    print(f"  {args.output}_params.npz  - 归一化参数和类别")
    print(f"  {args.output}_confusion.png - 混淆矩阵")
    print(f"  {args.output}_training.png - 训练曲线")
    print("\n使用方法：将 {args.output}.h 放到你的Arduino工程目录，"
          "然后在代码中包含它即可调用模型")

if __name__ == '__main__':
    main()
