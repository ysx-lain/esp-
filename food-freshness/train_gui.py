#!/usr/bin/env python3
"""
智能食材新鲜度检测 - 一体化训练GUI
功能：
1. 连接串口实时查看传感器数据
2. 点击开始采集数据，自动保存带标签CSV
3. 一键训练，自动滑动窗口预处理 → 训练 → 量化 → 导出C头文件
4. 显示训练曲线和混淆矩阵
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import glob
import pandas as pd
import numpy as np
import serial
import serial.tools.list_ports
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 训练模块导入
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

class TrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("食材新鲜度检测 - 一体化训练工具")
        self.root.geometry("1200x800")

        # 变量
        self.ser = None
        self.is_logging = False
        self.thread = None
        self.csv_file = None
        self.csv_writer = None
        
        self.data_dir = tk.StringVar(value="sensor_data")
        self.label_text = tk.StringVar(value="apple_fresh")
        self.window_size = tk.IntVar(value=5)
        self.window_step = tk.IntVar(value=2)
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=32)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.dropout = tk.DoubleVar(value=0.2)
        self.output_name = tk.StringVar(value="food_freshness")

        self.create_widgets()
        self.update_com_ports()
        self.root.after(5000, self.periodic_refresh)

    def create_widgets(self):
        # 左侧面板 - 设置
        left = ttk.Frame(self.root, padding=10, width=350)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # 串口设置
        frame = ttk.LabelFrame(left, text="串口连接", padding=5)
        frame.pack(fill=tk.X, pady=(0,10))

        ttk.Label(frame, text="串口号:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.port_combo = ttk.Combobox(frame, width=15)
        self.port_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Button(frame, text="刷新", command=self.update_com_ports).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="波特率:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.baud_combo = ttk.Combobox(frame, values=["9600","115200","230400"], 
                                      textvariable=tk.StringVar(value="115200"), width=10)
        self.baud_combo.grid(row=1, column=1, padx=5, sticky=tk.W)

        self.connect_btn = ttk.Button(frame, text="连接", command=self.connect_serial)
        self.connect_btn.grid(row=2, column=0, columnspan=2, pady=5)

        # 数据采集
        frame = ttk.LabelFrame(left, text="数据采集", padding=5)
        frame.pack(fill=tk.X, pady=(0,10))

        ttk.Label(frame, text="保存目录:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.data_dir, width=20).grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Button(frame, text="浏览", command=self.select_dir).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="当前标签:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.label_text, width=20).grid(row=1, column=1, columnspan=2, padx=5, sticky=tk.W)
        ttk.Label(frame, text="格式: category_freshness").grid(row=2, column=1, columnspan=2, sticky=tk.W)

        self.start_log_btn = ttk.Button(frame, text="开始采集", command=self.start_logging, state=tk.DISABLED)
        self.start_log_btn.grid(row=3, column=0, columnspan=2, pady=5)
        self.stop_log_btn = ttk.Button(frame, text="停止采集", command=self.stop_logging, state=tk.DISABLED)
        self.stop_log_btn.grid(row=3, column=2, pady=5)

        self.status_label = ttk.Label(frame, text="状态: 未连接", foreground="gray")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=2)

        # 训练参数
        frame = ttk.LabelFrame(left, text="训练参数", padding=5)
        frame.pack(fill=tk.X, pady=(0,10))

        ttk.Label(frame, text="滑动窗口大小:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.window_size, width=8).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(frame, text="滑动步长:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.window_step, width=8).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(frame, text="训练轮数:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=10, to=300, textvariable=self.epochs, width=8).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(frame, text="批次大小:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=8, to=128, textvariable=self.batch_size, width=8).grid(row=3, column=1, sticky=tk.W)

        ttk.Label(frame, text="学习率:").grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=0.0001, to=0.01, textvariable=self.learning_rate, width=8, increment=0.0001).grid(row=4, column=1, sticky=tk.W)

        ttk.Label(frame, text="Dropout:").grid(row=5, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=0.1, to=0.5, textvariable=self.dropout, width=8, increment=0.05).grid(row=5, column=1, sticky=tk.W)

        ttk.Label(frame, text="输出模型名:").grid(row=6, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.output_name, width=15).grid(row=6, column=1, padx=5, sticky=tk.W)

        self.train_btn = tk.Button(frame, text="开始训练", command=self.start_training, bg="#4CAF50", fg="white")
        self.train_btn.grid(row=7, column=0, columnspan=3, pady=10)

        # 右侧面板 - 输出和图表
        right = ttk.Frame(self.root, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        # 日志页
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="日志")
        self.log_text = tk.Text(log_frame, height=20, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 图表页
        plot_frame = ttk.Frame(notebook)
        notebook.add(plot_frame, text="训练曲线")
        self.plot_figure = Figure(figsize=(8, 4), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 混淆矩阵页
        cm_frame = ttk.Frame(notebook)
        notebook.add(cm_frame, text="混淆矩阵")
        self.cm_figure = Figure(figsize=(8, 4), dpi=100)
        self.cm_canvas = FigureCanvasTkAgg(self.cm_figure, master=cm_frame)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------ 串口和采集 ------------------
    def update_com_ports(self):
        ports = serial.tools.list_ports.comports()
        port_list = [p.device for p in ports]
        self.port_combo['values'] = port_list
        if port_list and not self.port_combo.get():
            self.port_combo.set(port_list[0])
        self.log(f"刷新串口，找到 {len(port_list)} 个设备")

    def periodic_refresh(self):
        self.update_com_ports()
        self.root.after(5000, self.periodic_refresh)

    def select_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.data_dir.set(d)

    def connect_serial(self):
        port = self.port_combo.get()
        baud = int(self.baud_combo.get())
        if not port:
            messagebox.showerror("错误", "请选择串口号")
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.connect_btn.config(text="已连接", state=tk.DISABLED)
            self.start_log_btn.config(state=tk.NORMAL)
            self.update_status("已连接", "green")
            self.log(f"成功连接 {port} @ {baud} baud", "success")
        except Exception as e:
            messagebox.showerror("错误", f"连接失败: {str(e)}")
            self.log(f"连接失败: {str(e)}", "error")

    def update_status(self, text, color="black"):
        self.status_label.config(text=f"状态: {text}", foreground=color)

    def log(self, msg, level="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        tag = ""
        if level == "error":
            tag = "❌ "
            self.log_text.tag_config("error", foreground="red")
        elif level == "success":
            tag = "✅ "
            self.log_text.tag_config("success", foreground="green")
        elif level == "data":
            tag = "📟 "
            self.log_text.tag_config("data", foreground="blue")
        self.log_text.insert(tk.END, f"[{ts}] {tag}{msg}\n", level)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_logging(self):
        if self.is_logging:
            return
        if not self.ser or not self.ser.is_open:
            messagebox.showerror("错误", "串口未连接")
            return
        label = self.label_text.get().strip()
        if not label:
            messagebox.showerror("错误", "请填写标签 (如: apple_fresh)")
            return
        
        os.makedirs(self.data_dir.get(), exist_ok=True)
        start_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.data_dir.get(), f'{label}_{start_str}.csv')
        
        try:
            self.csv_file = open(filename, 'w', newline='', encoding='utf-8-sig')
            self.csv_writer = csv.writer(self.csv_file)
            header = ['Time_s', 'Odor', 'HCHO', 'CO', 'VOC', 'CO2', 'Label']
            self.csv_writer.writerow(header)
            self.csv_file.flush()
        except Exception as e:
            messagebox.showerror("错误", f"创建文件失败: {str(e)}")
            return
        
        self.is_logging = True
        self.start_log_btn.config(state=tk.DISABLED)
        self.stop_log_btn.config(state=tk.NORMAL)
        self.update_status(f"采集中 - {label}", "green")
        self.log(f"开始采集，标签: {label}, 文件: {filename}", "success")
        
        self.thread = threading.Thread(target=self.reader_thread, daemon=True)
        self.thread.start()

    def stop_logging(self):
        if not self.is_logging:
            return
        self.is_logging = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        self.start_log_btn.config(state=tk.NORMAL)
        self.stop_log_btn.config(state=tk.DISABLED)
        self.update_status("已停止", "gray")
        self.log("采集已停止", "success")

    def reader_thread(self):
        block_lines = []
        inside_block = False
        while self.is_logging and self.ser and self.ser.is_open:
            try:
                raw = self.ser.readline().decode('utf-8', errors='ignore').rstrip('\r\n')
                if not raw:
                    continue
                self.root.after(0, lambda r=raw: self.log(f"RAW: {r}", "data"))

                if raw.startswith("========== 接收到传感器数据 =========="):
                    block_lines = []
                    inside_block = True
                    continue
                elif raw.startswith("======================================") and inside_block:
                    record = self.parse_block(block_lines)
                    if record:
                        self.root.after(0, lambda r=record: self.write_record(r))
                    inside_block = False
                    block_lines = []
                    continue
                elif inside_block:
                    block_lines.append(raw)

            except Exception as e:
                self.root.after(0, lambda e=e: self.log(f"串口异常: {str(e)}", "error"))
                break

    def parse_block(self, lines):
        import re
        def extract(pattern):
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    return match.group(1).strip()
            return None

        odor = extract(r"Odor:\s+([\d\.]+)\s+ppm")
        hcho = extract(r"HCHO:\s+([\d\.]+)\s+ppm")
        co = extract(r"CO:\s+([\d\.]+)\s+ppm")
        voc = extract(r"VOC:\s+([\d\.]+)\s+ppm")
        co2 = extract(r"CO2:\s+(\d+)\s+ppm")
        timestamp = extract(r"发送端时间戳:\s+(\d+)\s+ms")

        if None in [odor, hcho, co, voc, co2, timestamp]:
            return None

        try:
            return [
                int(timestamp)//1000,
                float(odor),
                float(hcho),
                float(co),
                float(voc),
                int(co2),
                self.label_text.get().strip()
            ]
        except:
            return None

    def write_record(self, record):
        # record: [time_s, odor, hcho, co, voc, co2, label]
        self.csv_writer.writerow(record)
        self.csv_file.flush()

    # ------------------ 训练 ------------------
    def create_sliding_windows(self, X, y, window_size, step):
        n_samples, n_features = X.shape
        windows = []
        labels = []
        for i in range(0, n_samples - window_size + 1, step):
            window = X[i:i+window_size, :]
            windows.append(window)
            center_idx = i + window_size // 2
            labels.append(y[center_idx])
        return np.array(windows), np.array(labels)

    def build_model(self, input_shape, num_classes, dropout_rate):
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

    def convert_to_tflite(self, model, output_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        return tflite_model

    def tflite_to_c_array(self, tflite_model, output_header_path, array_name="g_food_freshness_model"):
        with open(output_header_path, 'w') as f:
            f.write("// Autogenerated by train_gui.py\n")
            f.write("// DO NOT EDIT - generated from training GUI\n\n")
            f.write("#include <stddef.h>\n")
            f.write("#include <stdint.h>\n\n")
            f.write(f"const unsigned int {array_name}_size = {len(tflite_model)};\n");
            f.write(f"const unsigned char {array_name}[] = {{\n  ");
            bytes_per_line = 16;
            for i, byte in enumerate(tflite_model):
                f.write(f"0x{byte:02x}, ");
                if (i + 1) % bytes_per_line == 0:
                    f.write("\n  ");
            if len(tflite_model) % bytes_per_line != 0:
                f.write("\n");
            f.write("};\n\n");
        return output_header_path

    def start_training(self):
        if not os.path.exists(self.data_dir.get()):
            messagebox.showerror("错误", "数据目录不存在");
            return;
        csv_files = glob.glob(os.path.join(self.data_dir.get(), '*.csv'));
        if len(csv_files) == 0:
            messagebox.showerror("错误", "数据目录中没有找到CSV文件，请先采集数据");
            return;

        self.train_btn.config(state=tk.DISABLED);
        self.log("\n" + "="*60, "info");
        self.log("开始训练...", "info");
        self.log(f"数据目录: {self.data_dir.get()}", "info");
        self.log(f"找到 {len(csv_files)} 个CSV文件", "info");

        threading.Thread(target=self.training_thread, daemon=True).start();

    def training_thread(self):
        try:
            # 加载所有CSV
            all_dfs = [];
            for f in glob.glob(os.path.join(self.data_dir.get(), '*.csv')):
                self.root.after(0, lambda f=f: self.log(f"  加载 {os.path.basename(f)}", "info"));
                df = pd.read_csv(f);
                all_dfs.append(df);
            df = pd.concat(all_dfs, ignore_index=True);
            self.root.after(0, lambda: self.log(f"\n总共 {len(df)} 条原始样本", "info"));

            # 提取特征标签
            feature_cols = ['Odor', 'HCHO', 'CO', 'VOC', 'CO2'];
            X_raw = df[feature_cols].values;
            labels_raw = df['Label'].values;
            unique_labels = sorted(list(set(labels_raw)));
            label_to_idx = {l:i for i,l in enumerate(unique_labels)};
            y_raw = np.array([label_to_idx[l] for l in labels_raw]);

            self.root.after(0, lambda: self.log(f"\n类别列表 ({len(unique_labels)} 类):", "info"));
            for l,i in label_to_idx.items():
                self.root.after(0, lambda l=l,i=i: self.log(f"  [{i}] {l}", "info"));

            # 滑动窗口
            ws = self.window_size.get();
            step = self.window_step.get();
            if ws > 1:
                X_window, y_window = self.create_sliding_windows(X_raw, y_raw, ws, step);
                self.root.after(0, lambda: self.log(f"\n滑动窗口: 大小={ws}, 步长={step}, 生成 {X_window.shape[0]} 样本", "info"));
            else:
                X_window = X_raw[:, np.newaxis, :];
                y_window = y_raw;

            # 标准化
            n_features = X_window.shape[-1];
            scaler = StandardScaler();
            X_reshaped = X_window.reshape(-1, n_features);
            X_scaled_reshaped = scaler.fit_transform(X_reshaped);
            X = X_scaled_reshaped.reshape(X_window.shape);

            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_window, test_size=0.2, random_state=42, stratify=y_window
            );
            self.root.after(0, lambda: self.log(f"\n训练集: {X_train.shape[0]} 样本", "info"));
            self.root.after(0, lambda: self.log(f"测试集: {X_test.shape[0]} 样本", "info"));

            # 构建模型
            num_classes = len(unique_labels);
            model = self.build_model(X.shape[1:], num_classes, self.dropout.get());
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate.get()),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            );

            # 训练
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True);
            self.root.after(0, lambda: self.log("\n开始训练...", "info"));

            history = model.fit(
                X_train, y_train,
                batch_size=self.batch_size.get(),
                epochs=self.epochs.get(),
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0
            );

            # 评估
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0);
            self.root.after(0, lambda: self.log(f"\n训练完成，测试准确率: {test_acc:.4f}", "success"));

            y_pred = model.predict(X_test, verbose=0);
            y_pred_classes = np.argmax(y_pred, axis=1);
            report = classification_report(y_test, y_pred_classes, target_names=unique_labels);
            self.root.after(0, lambda: self.log("\n分类报告:\n" + report, "info"));

            # 绘制训练曲线
            self.root.after(0, lambda: self.plot_training(history));

            # 绘制混淆矩阵
            cm = confusion_matrix(y_test, y_pred_classes);
            self.root.after(0, lambda: self.plot_confusion(cm, unique_labels));

            # 导出
            output_base = self.output_name.get();
            tflite_path = f"{output_base}.tflite";
            header_path = f"{output_base}.h";

            self.root.after(0, lambda: self.log(f"\n导出TFLite模型: {tflite_path}", "info"));
            tflite_model = self.convert_to_tflite(model, tflite_path);
            size_kb = len(tflite_model) / 1024;
            self.root.after(0, lambda: self.log(f"模型大小: {size_kb:.1f} KB", "info"));

            self.root.after(0, lambda: self.log(f"导出C头文件: {header_path}", "info"));
            self.tflite_to_c_array(list(tflite_model), header_path);

            # 保存参数
            np.savez(f"{output_base}_params.npz", 
                    mean=scaler.mean_, 
                    std=np.sqrt(scaler.var_),
                    classes=np.array(unique_labels, dtype=object));

            self.root.after(0, lambda: self.log(f"\n✅ 全部完成！输出文件:", "success"));
            self.root.after(0, lambda: self.log(f"  {tflite_path} - INT8量化模型", "success"));
            self.root.after(0, lambda: self.log(f"  {header_path} - Arduino C头文件，直接include使用", "success"));
            self.root.after(0, lambda: self.log(f"  {output_base}_params.npz - 参数信息", "success"));

        except Exception as e:
            self.root.after(0, lambda: self.log(f"训练出错: {str(e)}", "error"));
            import traceback
            self.root.after(0, lambda: self.log(traceback.format_exc(), "error"));
        finally:
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL));

    def plot_training(self, history):
        self.plot_figure.clear();
        ax1 = self.plot_figure.add_subplot(121);
        ax1.plot(history.history['accuracy'], label='train');
        ax1.plot(history.history['val_accuracy'], label='val');
        ax1.set_title('Accuracy');
        ax1.legend();
        ax2 = self.plot_figure.add_subplot(122);
        ax2.plot(history.history['loss'], label='train');
        ax2.plot(history.history['val_loss'], label='val');
        ax2.set_title('Loss');
        ax2.legend();
        self.plot_figure.tight_layout();
        self.plot_canvas.draw();

    def plot_confusion(self, cm, class_names):
        self.cm_figure.clear();
        ax = self.cm_figure.add_subplot(111);
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues);
        self.cm_figure.colorbar(im, ax=ax);
        ax.set_xticks(np.arange(len(class_names)));
        ax.set_yticks(np.arange(len(class_names)));
        ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor");
        ax.set_yticklabels(class_names);
        ax.set_ylabel('True Label');
        ax.set_xlabel('Predicted Label');

        # 标注数字
        thresh = cm.max() / 2.;
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black");
        self.cm_figure.tight_layout();
        self.cm_canvas.draw();

if __name__ == "__main__":
    import csv
    root = tk.Tk()
    app = TrainGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_logging)
    root.mainloop()
