#!/usr/bin/env python3
"""
智能食材新鲜度检测 - 一体化训练GUI
功能：
1. 连接串口实时查看传感器数据，支持断开重连
2. 点击开始采集数据，自动保存带标签CSV（每个标签单独文件）
3. 一键训练，自动滑动窗口预处理 → 训练 → 量化 → 导出C头文件
4. 显示训练曲线和混淆矩阵
5. 自动同步采集频率到ESP32
6. 支持定时采集，从第一组数据开始计时，到点自动停止
7. 移除自动刷新串口，避免爆发式采集问题
8. 训练完成后直接发送模型文件到ESP32 SD卡升级
9. 支持指定输出目录保存模型文件
10.一键查询ESP端模型信息（大小、版本等）
11.持续读取串口输出，ESP端消息实时显示在日志
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
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
        self.root.geometry("1280x800")

        # 变量
        self.ser = None
        self.is_logging = False
        self.thread = None
        self.csv_file = None
        self.csv_writer = None
        self.auto_stop_timer_thread = None
        self.first_block_received = False
        self.timer_start_time = 0
        self.auto_stop_minutes = 10
        
        self.data_dir = tk.StringVar(value="sensor_data")
        self.label_text = tk.StringVar(value="apple_fresh")
        self.sample_interval = tk.IntVar(value=5)      # 采集频率（秒）
        self.window_size = tk.IntVar(value=5)        # 滑动窗口大小（帧数）
        self.window_step = tk.IntVar(value=2)          # 滑动步长（帧数）
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=32)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.dropout = tk.DoubleVar(value=0.2)
        self.output_dir = tk.StringVar(value="./")
        self.output_name = tk.StringVar(value="food_freshness")
        self.auto_stop_minutes_var = tk.DoubleVar(value=10)

        # 缓存多行数据块
        self.block_lines = []
        self.inside_block = False

        # 防抖定时器
        self.interval_change_timer = None

        self.create_widgets()
        self.update_com_ports()
        # 移除自动定时刷新串口 → 彻底解决爆发式采集问题
        # 移除实时监听频率变化 → 避免频繁发送命令导致爆发读取
        # 只有开始采集时才发送一次频率命令

    def create_widgets(self):
        # 左侧面板 - 设置
        left = tk.Frame(self.root, width=380, padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # 串口设置
        frame = tk.LabelFrame(left, text="串口连接", padx=5, pady=5)
        frame.pack(fill=tk.X, pady=(0,10))

        ttk.Label(frame, text="串口号:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.port_combo = ttk.Combobox(frame, width=15)
        self.port_combo.grid(row=0, column=1, padx=5)
        tk.Button(frame, text="刷新", command=self.update_com_ports, bg="#2196F3", fg="white").grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="波特率:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.baud_combo = ttk.Combobox(frame, values=["9600","115200","230400"], 
                                      textvariable=tk.StringVar(value="115200"), width=10)
        self.baud_combo.grid(row=1, column=1, padx=5, sticky=tk.W)

        self.connect_btn = tk.Button(frame, text="连接", command=self.connect_serial, bg="#2196F3", fg="white")
        self.connect_btn.grid(row=2, column=0, columnspan=2, pady=5)
        self.disconnect_btn = tk.Button(frame, text="断开", command=self.disconnect_serial, bg="#f44336", fg="white", state=tk.DISABLED)
        self.disconnect_btn.grid(row=2, column=2, pady=5)

        # 数据采集
        frame = tk.LabelFrame(left, text="数据采集", padx=5, pady=5)
        frame.pack(fill=tk.X, pady=(0,10))

        ttk.Label(frame, text="保存目录:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.data_dir, width=20).grid(row=0, column=1, padx=5, sticky=tk.W)
        tk.Button(frame, text="浏览", command=self.select_dir, bg="#2196F3", fg="white").grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="当前标签:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.label_text, width=20).grid(row=1, column=1, columnspan=2, padx=5, sticky=tk.W)
        ttk.Label(frame, text="格式: cat_fresh").grid(row=2, column=1, columnspan=2, sticky=tk.W)

        ttk.Label(frame, text="间隔(秒):").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=2, to=60, textvariable=self.sample_interval, width=6).grid(row=3, column=1, sticky=tk.W)
        ttk.Label(frame, text="开始同步").grid(row=3, column=2, padx=3, sticky=tk.W);

        ttk.Label(frame, text="定时(分):").grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=0, to=120, textvariable=self.auto_stop_minutes_var, width=6).grid(row=4, column=1, sticky=tk.W)
        ttk.Label(frame, text="0=不停止").grid(row=4, column=2, padx=3, sticky=tk.W);

        self.start_log_btn = tk.Button(frame, text="开始采集", command=self.start_logging, state=tk.DISABLED, bg="#4CAF50", fg="white")
        self.start_log_btn.grid(row=5, column=0, columnspan=2, pady=5)
        self.stop_log_btn = tk.Button(frame, text="停止采集", command=self.stop_logging, state=tk.DISABLED, bg="#f44336", fg="white")
        self.stop_log_btn.grid(row=5, column=2, pady=5)

        self.status_label = tk.Label(frame, text="状态: 未连接", foreground="gray")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=2)

        self.timer_label = tk.Label(frame, text="", foreground="#d32f2f", font=("bold", 12))
        self.timer_label.grid(row=7, column=0, columnspan=3, pady=2)

        # 训练参数
        frame = tk.LabelFrame(left, text="训练参数", padx=5, pady=5)
        frame.pack(fill=tk.X, pady=(0,10))

        ttk.Label(frame, text="窗口大小:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.window_size, width=6).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(frame, text="帧").grid(row=0, column=2, padx=2, sticky=tk.W);

        ttk.Label(frame, text="步长:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.window_step, width=6).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(frame, text="帧").grid(row=1, column=2, padx=2, sticky=tk.W);

        ttk.Label(frame, text="轮数:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=10, to=300, textvariable=self.epochs, width=6).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(frame, text="批次:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=8, to=128, textvariable=self.batch_size, width=6).grid(row=3, column=1, sticky=tk.W)

        ttk.Label(frame, text="LR:").grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=0.0001, to=0.01, textvariable=self.learning_rate, width=6, increment=0.0001).grid(row=4, column=1, sticky=tk.W)

        ttk.Label(frame, text="Drop:").grid(row=5, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(frame, from_=0.1, to=0.5, textvariable=self.dropout, width=6).grid(row=5, column=1, sticky=tk.W)

        ttk.Label(frame, text="输出目录:").grid(row=6, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.output_dir, width=12).grid(row=6, column=1, padx=5, sticky=tk.W)
        tk.Button(frame, text="浏览", command=self.select_output_dir, bg="#2196F3", fg="white").grid(row=6, column=2, pady=2, padx=2)

        ttk.Label(frame, text="输出模型名:").grid(row=7, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.output_name, width=15).grid(row=7, column=1, columnspan=2, padx=5, sticky=tk.W)

        self.train_btn = tk.Button(frame, text="开始训练", command=self.start_training, bg="#4CAF50", fg="white")
        self.train_btn.grid(row=8, column=0, columnspan=2, pady=10)
        self.send_model_btn = tk.Button(frame, text="发送模型到ESP", command=self.select_and_send_model, bg="#FF9800", fg="white")
        self.send_model_btn.grid(row=8, column=2, pady=10)

        # 模型查询按钮
        self.check_model_btn = tk.Button(frame, text="查看模型信息", command=self.check_model_info, bg="#9C27B0", fg="white")
        self.check_model_btn.grid(row=9, column=0, columnspan=3, pady=5)

        # 右侧面板 - 输出和图表
        right = tk.Frame(self.root, padx=10, pady=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)

        # 日志页
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="日志")
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 实时数据页
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="实时数据")
        self.data_text = scrolledtext.ScrolledText(data_frame, height=20, font=("Consolas", 9))
        self.data_text.pack(fill=tk.BOTH, expand=True)

        # 训练曲线页
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

    def select_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.data_dir.set(d)

    def select_output_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir.set(d)

    def send_command_to_sensor(self, cmd):
        if not self.ser or not self.ser.is_open:
            self.log(f"串口未打开，无法发送命令: {cmd}", "error")
            return False
        try:
            self.ser.write((cmd + "\n").encode())
            self.log(f"已发送命令: {cmd}", "success")
            return True
        except Exception as e:
            self.log(f"发送命令失败: {e}", "error")
            return False

    def sync_interval(self):
        interval = self.sample_interval.get()
        if interval < 2:
            self.log(f"采集间隔 {interval} 秒过小，强制设为2秒", "warning")
            interval = 2
            self.sample_interval.set(interval)
        cmd = f"set interval {interval}"
        if self.is_logging and self.ser and self.ser.is_open:
            self.send_command_to_sensor(cmd)
        else:
            self.log(f"未在采集中，间隔同步命令暂不发送: {cmd}", "info")



    def connect_serial(self):
        port = self.port_combo.get()
        baud = int(self.baud_combo.get())
        if not port:
            messagebox.showerror("错误", "请选择串口号")
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.connect_btn.config(text="已连接", state=tk.DISABLED, bg="#cccccc")
            self.disconnect_btn.config(state=tk.NORMAL, bg="#f44336")
            self.start_log_btn.config(state=tk.NORMAL)
            self.update_status("已连接", "green")
            self.log(f"成功连接 {port} @ {baud} baud", "success")
            self.log("采集频率将在点击开始采集时自动同步，避免频繁发送", "info")
            # 启动串口读取线程，实时显示ESP输出
            self.serial_reader_thread = threading.Thread(target=self.serial_reader_loop, daemon=True)
            self.serial_reader_thread.start()
        except Exception as e:
            messagebox.showerror("错误", f"连接失败: {str(e)}")
            self.log(f"连接失败: {str(e)}", "error")

    def disconnect_serial(self):
        if self.is_logging:
            self.stop_logging()
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.connect_btn.config(text="连接", state=tk.NORMAL, bg="#2196F3")
        self.disconnect_btn.config(state=tk.DISABLED, bg="#cccccc")
        self.start_log_btn.config(state=tk.DISABLED)
        self.stop_log_btn.config(state=tk.DISABLED)
        self.update_status("未连接", "gray")
        self.log("已断开串口连接", "success")

    def serial_reader_loop(self):
        """持续读取串口输出，显示到日志"""
        while self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').rstrip('\r\n')
                    if line:
                        self.root.after(0, lambda l=line: self.log(f"ESP: {l}", "info"))
                else:
                    time.sleep(0.01)
            except Exception as e:
                if self.ser and self.ser.is_open:
                    self.root.after(0, lambda e=e: self.log(f"串口读取错误: {e}", "error"))
                break

    def select_and_send_model(self):
        """选择模型文件，发送到ESP32 SD卡升级"""
        if not self.ser or not self.ser.is_open:
            messagebox.showerror("错误", "串口未连接，请先连接")
            return
        filename = filedialog.askopenfilename(
            title="选择模型文件发送 (.tflite / .bin)",
            filetypes=[("Binary files", "*.tflite *.bin"), ("All files", "*.*")]
        )
        if not filename:
            return
        try:
            file_size = os.path.getsize(filename)
            if file_size > 128 * 1024:
                messagebox.showerror("错误", "模型文件太大，最大支持128KB")
                return
            # 发送命令
            self.ser.write(b"update model\n")
            self.log(f"🔔 已发送 'update model' 命令，开始发送 {os.path.basename(filename)} ({file_size} bytes)...", "info")
            # 读取并发送文件
            with open(filename, 'rb') as f:
                data = f.read()
                self.ser.write(data)
            self.log(f"✅ 发送完成，总共 {len(data)} bytes", "success")
            self.log("⌛ 等待ESP接收完成，完成后会显示结果，重启ESP加载新模型", "info")
        except Exception as e:
            messagebox.showerror("错误", f"发送失败: {str(e)}")
            self.log(f"发送失败: {str(e)}", "error")

    def check_model_info(self):
        """发送命令查询ESP端模型信息"""
        if not self.ser or not self.ser.is_open:
            messagebox.showerror("错误", "串口未连接，请先连接")
            return
        self.log("\n🔍 查询模型信息...", "info")
        self.send_command_to_sensor("info model")

    def update_status(self, text, color="black"):
        self.status_label.config(text=f"状态: {text}", foreground=color)

    def log(self, msg, level="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        if level == "error":
            self.log_text.insert(tk.END, f"[{ts}] ❌ {msg}\n", "error")
            self.log_text.tag_config("error", foreground="red")
        elif level == "success":
            self.log_text.insert(tk.END, f"[{ts}] ✅ {msg}\n", "success")
            self.log_text.tag_config("success", foreground="green")
        elif level == "data":
            self.log_text.insert(tk.END, f"[{ts}] 📟 {msg}\n", "data")
            self.log_text.tag_config("data", foreground="blue")
        elif level == "warning":
            self.log_text.insert(tk.END, f"[{ts}] ⚠️ {msg}\n", "warning")
            self.log_text.tag_config("warning", foreground="orange")
        else:
            self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def append_display(self, record):
        # record: [time_s, odor, hcho, co, voc, co2]
        line = f"Time:{record[0]:8d}s | O:{record[1]:5.2f} | H:{record[2]:5.2f} | C:{record[3]:5.2f} | V:{record[4]:5.2f} | CO2:{record[5]:4d}\n"
        self.data_text.insert(tk.END, line)
        if int(self.data_text.index('end-1c').split('.')[0]) > 21:
            self.data_text.delete('1.0', '2.0')
        self.data_text.see(tk.END)
        self.root.update_idletasks()

    def check_auto_stop(self):
        """自动停止定时器 - 从第一包接收后开始计时"""
        while self.is_logging and self.auto_stop_minutes > 0:
            if not self.is_logging:
                break  # 已经手动停止采集，直接退出
            elapsed_min = (time.time() - self.timer_start_time) / 60.0
            remaining = self.auto_stop_minutes - elapsed_min
            if remaining <= 0:
                self.log(f"⏰ 定时采集到达设定时间 ({self.auto_stop_minutes} 分钟)，自动停止", "success")
                self.root.after(0, self.stop_logging)
                break
            else:
                m = int(remaining)
                s = int((remaining - m) * 60)
                self.root.after(0, lambda: self.timer_label.config(text=f"剩余时间: {m}:{s:02d}"))
            time.sleep(1)
        if self.is_logging:
            self.timer_label.config(text="")

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
        
        self.auto_stop_minutes = self.auto_stop_minutes_var.get()
        os.makedirs(self.data_dir.get(), exist_ok=True)
        start_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.data_dir.get(), f'{label}_{start_str}.csv')
        
        try:
            import csv
            self.csv_file = open(filename, 'w', newline='', encoding='utf-8-sig')
            self.csv_writer = csv.writer(self.csv_file)
            header = ['Time_s', 'Odor', 'HCHO', 'CO', 'VOC', 'CO2', 'Label']
            self.csv_writer.writerow(header)
            self.csv_file.flush()
        except Exception as e:
            messagebox.showerror("错误", f"创建文件失败: {str(e)}")
            return
        
        # 开始采集时，只发送一次频率命令，避免频繁发送导致爆发读取
        interval = self.sample_interval.get()
        if interval < 2:
            interval = 2
            self.sample_interval.set(interval)
        cmd = f"set interval {interval}"
        self.send_command_to_sensor(cmd)
        self.log(f"同步采集频率: {interval} 秒 → {cmd}", "success")

        # 开始采集前清空串口缓冲区，避免残留旧数据一次性爆发读出
        self.ser.reset_input_buffer()
        self.block_lines = []
        self.inside_block = False

        self.is_logging = True
        self.first_block_received = False
        self.start_log_btn.config(state=tk.DISABLED)
        self.stop_log_btn.config(state=tk.NORMAL)
        self.update_status(f"采集中 - {label}", "green")
        self.log(f"开始采集，标签: {label}, 文件: {filename}（每个标签单独CSV）", "success")
        self.log("已清空串口输入缓冲区，避免残留旧数据", "info")
        if self.auto_stop_minutes > 0:
            self.log(f"⏰ 自动停止设置: {self.auto_stop_minutes} 分钟，第一组数据到达后开始计时", "info")
        
        self.thread = threading.Thread(target=self.reader_thread, daemon=True)
        self.thread.start()

        if self.auto_stop_minutes > 0:
            self.auto_stop_timer_thread = threading.Thread(target=self.check_auto_stop, daemon=True)
            self.auto_stop_timer_thread.start()

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
        self.timer_label.config(text="")
        self.log("采集已停止", "success")

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
        self.log(f"记录样本: Time={record[0]}s, O={record[1]}, H={record[2]}, C={record[3]}, V={record[4]}, CO2={record[5]}", "data")
        self.append_display(record[:6])

    def reader_thread(self):
        while self.is_logging and self.ser and self.ser.is_open:
            try:
                raw = self.ser.readline().decode('utf-8', errors='ignore').rstrip('\r\n')
                if not raw:
                    continue
                self.root.after(0, lambda r=raw: self.log(f"RAW: {r}", "data"))

                if raw.startswith("========== 接收到传感器数据 =========="):
                    self.block_lines = []
                    self.inside_block = True
                    if not self.first_block_received:
                        self.root.after(0, lambda: self.log("✅ 收到第一组数据，开始计时", "success"))
                        self.first_block_received = True
                        self.timer_start_time = time.time()
                    continue
                elif raw.startswith("======================================") and self.inside_block:
                    record = self.parse_block(self.block_lines)
                    if record and self.csv_writer:
                        self.root.after(0, lambda r=record: self.write_record(r))
                    self.inside_block = False
                    self.block_lines = []
                    continue
                elif self.inside_block:
                    self.block_lines.append(raw)

            except serial.SerialException as e:
                self.root.after(0, lambda e=e: self.log(f"串口异常: {e}", "error"))
                break
            except Exception as e:
                self.root.after(0, lambda e=e: self.log(f"未知错误: {e}", "error"))
                break

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
        seq_len = input_shape[0]
        model = Sequential()
        model.add(Conv1D(16, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        if seq_len > 5:
            model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))
        
        model.add(Conv1D(32, kernel_size=2, activation='relu'))
        model.add(BatchNormalization())
        if seq_len > 10:
            model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))
        
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))
        return model

    def convert_to_tflite(self, model, output_path, X_train):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 需要为INT8量化提供代表性数据集
        def representative_dataset_gen():
            # 取前100个样本做代表性校准
            n_cal = min(100, X_train.shape[0])
            for i in range(n_cal):
                yield [X_train[np.newaxis, i].astype(np.float32)]
        
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        return tflite_model

    def tflite_to_c_array(self, tflite_model, output_header_path, array_name="g_food_freshness_model"):
        with open(output_header_path, 'w') as f:
            f.write("// Autogenerated by train_gui.py\n");
            f.write("// DO NOT EDIT - generated from training GUI\n\n");
            f.write("#include <stddef.h>\n");
            f.write("#include <stdint.h>\n\n");
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
            messagebox.showerror("错误", "数据目录不存在")
            return
        csv_files = glob.glob(os.path.join(self.data_dir.get(), '*.csv'))
        if len(csv_files) == 0:
            messagebox.showerror("错误", "数据目录中没有找到CSV文件，请先采集数据")
            return

        self.train_btn.config(state=tk.DISABLED)
        self.log("\n" + "="*60, "info")
        self.log("开始训练...", "info")
        self.log(f"数据目录: {self.data_dir.get()}", "info")
        self.log(f"找到 {len(csv_files)} 个CSV文件", "info")

        threading.Thread(target=self.training_thread, daemon=True).start()

    def training_thread(self):
        try:
            # 加载所有CSV
            all_dfs = []
            for f in glob.glob(os.path.join(self.data_dir.get(), '*.csv')):
                self.root.after(0, lambda f=f: self.log(f"  加载 {os.path.basename(f)}", "info"))
                df = pd.read_csv(f)
                all_dfs.append(df)
            df = pd.concat(all_dfs, ignore_index=True)
            self.root.after(0, lambda: self.log(f"\n总共 {len(df)} 条原始样本", "info"))

            # 提取特征标签
            feature_cols = ['Odor', 'HCHO', 'CO', 'VOC', 'CO2']
            X_raw = df[feature_cols].values
            labels_raw = df['Label'].values
            unique_labels = sorted(list(set(labels_raw)))
            label_to_idx = {l:i for i,l in enumerate(unique_labels)}
            y_raw = np.array([label_to_idx[l] for l in labels_raw])

            self.root.after(0, lambda: self.log(f"\n类别列表 ({len(unique_labels)} 类):", "info"))
            for l,i in label_to_idx.items():
                self.root.after(0, lambda l=l,i=i: self.log(f"  [{i}] {l}", "info"))

            # 滑动窗口
            ws = self.window_size.get()
            step = self.window_step.get()
            if ws > 1:
                X_window, y_window = self.create_sliding_windows(X_raw, y_raw, ws, step)
                self.root.after(0, lambda: self.log(f"\n滑动窗口: 大小={ws}, 步长={step}, 生成 {X_window.shape[0]} 样本", "info"))
            else:
                X_window = X_raw[:, np.newaxis, :]
                y_window = y_raw

            # 标准化
            n_features = X_window.shape[-1]
            scaler = StandardScaler()
            X_reshaped = X_window.reshape(-1, n_features)
            X_scaled_reshaped = scaler.fit_transform(X_reshaped)
            X = X_scaled_reshaped.reshape(X_window.shape)

            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_window, test_size=0.2, random_state=42, stratify=y_window
            )
            self.root.after(0, lambda: self.log(f"\n训练集: {X_train.shape[0]} 样本", "info"))
            self.root.after(0, lambda: self.log(f"测试集: {X_test.shape[0]} 样本", "info"))

            # 构建模型
            num_classes = len(unique_labels)
            model = self.build_model(X.shape[1:], num_classes, self.dropout.get())
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate.get()),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # 训练
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            self.root.after(0, lambda: self.log("\n开始训练...", "info"))

            history = model.fit(
                X_train, y_train,
                batch_size=self.batch_size.get(),
                epochs=self.epochs.get(),
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0
            )

            # 评估
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            self.root.after(0, lambda: self.log(f"\n训练完成，测试准确率: {test_acc:.4f}", "success"))

            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            report = classification_report(y_test, y_pred_classes, target_names=unique_labels)
            self.root.after(0, lambda: self.log("\n分类报告:\n" + report, "info"))

            # 绘制训练曲线
            self.root.after(0, lambda: self.plot_training(history))

            # 绘制混淆矩阵
            cm = confusion_matrix(y_test, y_pred_classes)
            self.root.after(0, lambda: self.plot_confusion(cm, unique_labels))

            # 导出
            output_base = os.path.join(self.output_dir.get(), self.output_name.get())
            tflite_path = f"{output_base}.tflite"
            header_path = f"{output_base}.h"
            params_path = f"{output_base}_params.npz"

            # 确保输出目录存在
            os.makedirs(self.output_dir.get(), exist_ok=True)

            self.root.after(0, lambda: self.log(f"\n导出TFLite模型: {tflite_path}", "info"))
            tflite_model = self.convert_to_tflite(model, tflite_path, X_train)
            size_kb = len(tflite_model) / 1024
            self.root.after(0, lambda: self.log(f"模型大小: {size_kb:.1f} KB", "info"))

            self.root.after(0, lambda: self.log(f"导出C头文件: {header_path}", "info"))
            self.tflite_to_c_array(list(tflite_model), header_path)

            # 保存参数
            np.savez(params_path, 
                    mean=scaler.mean_, 
                    std=np.sqrt(scaler.var_),
                    classes=np.array(unique_labels, dtype=object))

            self.root.after(0, lambda: self.log(f"\n✅ 全部完成！输出文件:", "success"))
            self.root.after(0, lambda: self.log(f"  {tflite_path} - INT8量化模型", "success"))
            self.root.after(0, lambda: self.log(f"  {header_path} - Arduino C头文件，直接include使用", "success"))
            self.root.after(0, lambda: self.log(f"  {params_path} - 参数信息", "success"))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"训练出错: {str(e)}", "error"))
            import traceback
            self.root.after(0, lambda: self.log(traceback.format_exc(), "error"))
        finally:
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))

    def plot_training(self, history):
        self.plot_figure.clear()
        ax1 = self.plot_figure.add_subplot(121)
        ax1.plot(history.history['accuracy'], label='train')
        ax1.plot(history.history['val_accuracy'], label='val')
        ax1.set_title('Accuracy')
        ax1.legend()
        ax2 = self.plot_figure.add_subplot(122)
        ax2.plot(history.history['loss'], label='train')
        ax2.plot(history.history['val_loss'], label='val')
        ax2.set_title('Loss')
        ax2.legend()
        self.plot_figure.tight_layout()
        self.plot_canvas.draw()

    def plot_confusion(self, cm, class_names):
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)
        import matplotlib.pyplot as plt
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        self.cm_figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(class_names)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        # 标注数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        self.cm_figure.tight_layout()
        self.cm_canvas.draw()

if __name__ == "__main__":
    import csv
    root = tk.Tk()
    app = TrainGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()