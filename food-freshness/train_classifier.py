"""
食材新鲜度分类模型训练脚本
功能：
- 自动扫描文件夹中所有CSV文件
- 从文件名或标签列读取类别标签
- 训练MLP神经网络分类
- 导出量化后的TFLite模型，可直接部署到ESP32-S3

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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='训练食材新鲜度分类模型')
    parser.add_argument('--data_dir', default='sensor_data', help='CSV数据文件夹路径')
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout比率')
    parser.add_argument('--output', default='food_freshness_model', help='输出模型名')
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
    print(f"\n总共 {len(combined)} 条样本")
    return combined

def extract_features_labels(df):
    """提取特征和标签"""
    # 特征：5个气体传感器通道
    feature_cols = ['Odor', 'HCHO', 'CO', 'VOC', 'CO2']
    X = df[feature_cols].values
    
    # 标签：从Label列提取
    labels = df['Label'].values
    
    # 去重得到类别列表
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels])
    
    print(f"\n类别列表 ({len(unique_labels)} 类):")
    for l, i in label_to_idx.items():
        print(f"  [{i}] {l}")
    
    return X, y, unique_labels, label_to_idx

def build_model(input_dim, num_classes, dropout_rate):
    """构建轻量MLP模型，适合ESP32部署"""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),
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

def convert_to_tflite(model, scaler, class_names, output_path):
    """转换为INT8量化TFLite模型，适合ESP32部署"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # INT8量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 这里可以用代表性数据集进一步优化，简化跳过也可
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # 保存模型
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"\n量化TFLite模型已保存: {output_path}, 大小: {size_kb:.1f} KB")
    print("此大小适合ESP32-S3部署")
    
    # 同时保存类别标签和缩放参数，方便部署时使用
    info_path = output_path.replace('.tflite', '_info.txt')
    with open(info_path, 'w') as f:
        f.write("class_names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")
        f.write(f"\nscaler_mean: {list(scaler.mean_)}\n")
        f.write(f"scaler_std: {list(np.sqrt(scaler.var_))}\n")
    print(f"模型信息已保存: {info_path}")

def main():
    args = parse_args()
    
    # 1. 加载数据
    print("=" * 60)
    print("食材新鲜度分类模型训练")
    print("=" * 60)
    
    df = load_all_csv(args.data_dir)
    X, y, class_names, label_to_idx = extract_features_labels(df)
    
    # 2. 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 3. 构建模型
    num_classes = len(class_names)
    model = build_model(X_scaled.shape[1], num_classes, args.dropout)
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. 训练
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
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
    
    # 5. 评估
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
    
    # 6. 导出TFLite模型
    print("\n" + "=" * 60)
    print("导出INT8量化TFLite模型")
    print("=" * 60)
    convert_to_tflite(model, scaler, class_names, f"{args.output}.tflite")
    
    # 保存scaler参数供C++使用
    np.savez(f"{args.output}_scaler.npz", 
            mean=scaler.mean_, 
            std=np.sqrt(scaler.var_),
            classes=np.array(class_names, dtype=object))
    print(f"缩放参数已保存: {args.output}_scaler.npz")
    
    print("\n训练完成！")
    print(f"最终模型: {args.output}.tflite")
    print(f"可以直接将此文件烧录到ESP32-S3使用TFLite Micro推理")

if __name__ == '__main__':
    main()
