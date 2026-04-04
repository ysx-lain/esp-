import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import serial
import serial.tools.list_ports
import threading
import time
import csv
import os
import re
from datetime import datetime

class SensorLogger:
 def __init__(self, root):
  self.root = root
  self.root.title("气体监测数据采集器 - 自动同步采样间隔")
  self.root.geometry("1000x850")

  # 变量
  self.serial_port = tk.StringVar()
  self.baudrate = tk.StringVar(value="115200")
  self.output_dir = tk.StringVar(value="sensor_data")
  self.file_interval = tk.IntVar(value=30) # CSV 文件切换间隔（分钟）
  self.record_interval = tk.IntVar(value=30) # 记录间隔（秒），同时用于同步传感器
  self.label_text = tk.StringVar(value="") # 自定义标签
  self.is_logging = False
  self.ser = None
  self.thread = None
  self.csv_file = None
  self.csv_writer = None
  self.file_start_time = None
  self.auto_refresh_enabled = True

  # 缓存多行数据块
  self.block_lines = []
  self.inside_block = False

  # 用于记录间隔变化时的防抖定时器
  self.interval_change_timer = None

  self.create_widgets()
  self.update_com_ports()
  self.root.after(2000, self.periodic_refresh)

  # 监听 record_interval 的变化
  self.record_interval.trace('w', self.on_record_interval_changed)

 # ------------------ 界面 ------------------
 def create_widgets(self):
  main = ttk.Frame(self.root, padding="10")
  main.pack(fill=tk.BOTH, expand=True)

  cfg = ttk.LabelFrame(main, text="串口与存储设置", padding="5")
  cfg.pack(fill=tk.X, pady=(0,10))

  ttk.Label(cfg, text="串口号:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
  self.port_combo = ttk.Combobox(cfg, textvariable=self.serial_port, width=15)
  self.port_combo.grid(row=0, column=1, padx=5)
  ttk.Button(cfg, text="刷新", command=self.update_com_ports).grid(row=0, column=2, padx=5)

  ttk.Label(cfg, text="波特率:").grid(row=0, column=3, padx=5, pady=2)
  baud_combo = ttk.Combobox(cfg, textvariable=self.baudrate, values=["9600","115200","230400"], width=8)
  baud_combo.grid(row=0, column=4, padx=5)

  ttk.Label(cfg, text="保存目录:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
  ttk.Entry(cfg, textvariable=self.output_dir, width=40).grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5)
  ttk.Button(cfg, text="浏览", command=self.select_dir).grid(row=1, column=4, padx=5)

  ttk.Label(cfg, text="文件切换间隔(分钟):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
  ttk.Spinbox(cfg, from_=1, to=1440, textvariable=self.file_interval, width=8).grid(row=2, column=1, sticky=tk.W)
  ttk.Label(cfg, text="(每30分钟生成一个新CSV文件)").grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=10)

  ttk.Label(cfg, text="记录间隔(秒):").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
  # 最小间隔改为 2 秒
  self.interval_spinbox = ttk.Spinbox(cfg, from_=2, to=3600, textvariable=self.record_interval, width=8)
  self.interval_spinbox.grid(row=3, column=1, sticky=tk.W)
  ttk.Label(cfg, text="(同步传感器采集间隔，最小2秒，每收到一条数据即保存)").grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=10)

  ttk.Label(cfg, text="标签(Label):").grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
  ttk.Entry(cfg, textvariable=self.label_text, width=30).grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5)
  ttk.Label(cfg, text="(添加到CSV最后一列)").grid(row=4, column=3, sticky=tk.W, padx=10)

  btn_frame = ttk.Frame(cfg)
  btn_frame.grid(row=5, column=0, columnspan=5, pady=10)
  self.start_btn = ttk.Button(btn_frame, text="开始采集", command=self.start_logging)
  self.start_btn.pack(side=tk.LEFT, padx=5)
  self.stop_btn = ttk.Button(btn_frame, text="停止采集", command=self.stop_logging, state=tk.DISABLED)
  self.stop_btn.pack(side=tk.LEFT, padx=5)
  self.status_label = ttk.Label(btn_frame, text="状态: 未开始", foreground="gray")
  self.status_label.pack(side=tk.LEFT, padx=20)

  disp_frame = ttk.LabelFrame(main, text="实时数据 (最新20条)", padding="5")
  disp_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
  self.data_text = scrolledtext.ScrolledText(disp_frame, height=12, font=("Consolas", 9))
  self.data_text.pack(fill=tk.BOTH, expand=True)

  log_frame = ttk.LabelFrame(main, text="运行日志 & 原始数据", padding="5")
  log_frame.pack(fill=tk.BOTH, expand=True)
  self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=("Consolas", 9))
  self.log_text.pack(fill=tk.BOTH, expand=True)

 # ------------------ 辅助函数 ------------------
 def update_com_ports(self):
  ports = serial.tools.list_ports.comports()
  port_list = [p.device for p in ports]
  self.port_combo['values'] = port_list
  if port_list and not self.serial_port.get():
   self.serial_port.set(port_list[0])
  self.log(f"串口列表: {', '.join(port_list) if port_list else '无'}", level="info")

 def periodic_refresh(self):
  if self.auto_refresh_enabled:
   self.update_com_ports()
  self.root.after(5000, self.periodic_refresh)

 def select_dir(self):
  d = filedialog.askdirectory()
  if d:
   self.output_dir.set(d)

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
  else:
   self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
  self.log_text.see(tk.END)
  self.root.update_idletasks()

 def update_status(self, text, color="black"):
  self.status_label.config(text=f"状态: {text}", foreground=color)

 # 实时显示：去掉温度和湿度
 def append_display(self, record):
  # record 包含9个元素: [占位, odor, hcho, co, voc, co2, env_temp, hum, timestamp_ms]
  time_sec = record[8] // 1000
  line = (f"Time:{time_sec:10d}s | O:{record[1]:5.2f} | H:{record[2]:5.2f} | "
    f"C:{record[3]:5.2f} | V:{record[4]:5.2f} | CO2:{record[5]:4d}\n")
  self.data_text.insert(tk.END, line)
  if int(self.data_text.index('end-1c').split('.')[0]) > 21:
   self.data_text.delete('1.0', '2.0')
  self.data_text.see(tk.END)

 # ------------------ 发送命令到接收端（进而控制发送端） ------------------
 def send_command_to_sensor(self, cmd):
  """通过串口向接收端发送命令，接收端会通过ESP-NOW转发给发送端"""
  if not self.ser or not self.ser.is_open:
   self.log(f"串口未打开，无法发送命令: {cmd}", level="error")
   return False
  try:
   self.ser.write((cmd + "\n").encode())
   self.log(f"已发送命令: {cmd}", level="success")
   return True
  except Exception as e:
   self.log(f"发送命令失败: {e}", level="error")
   return False

 # 同步采样间隔，最小2秒
 def sync_interval(self):
  """同步采样间隔：向发送端发送 set interval 命令"""
  interval = self.record_interval.get()
  if interval < 2:
   self.log(f"记录间隔 {interval} 秒过小，强制设为2秒", level="warning")
   interval = 2
   self.record_interval.set(interval)
  cmd = f"set interval {interval}"
  if self.is_logging and self.ser and self.ser.is_open:
   self.send_command_to_sensor(cmd)
  else:
   self.log(f"未在采集中，间隔同步命令暂不发送: {cmd}", level="info")

 def on_record_interval_changed(self, *args):
  """当记录间隔数值被修改时调用（防抖处理）"""
  if not self.is_logging:
   return
  if self.interval_change_timer:
   self.root.after_cancel(self.interval_change_timer)
  self.interval_change_timer = self.root.after(1000, self.sync_interval)

 # ------------------ CSV 管理 ------------------
 def create_csv(self):
  if self.csv_file:
   self.csv_file.close()
  os.makedirs(self.output_dir.get(), exist_ok=True)
  start_str = datetime.now().strftime('%Y%m%d_%H%M%S')
  filename = os.path.join(self.output_dir.get(), f'sensor_data_{start_str}.csv')
  self.csv_file = open(filename, 'w', newline='', encoding='utf-8-sig')
  self.csv_writer = csv.writer(self.csv_file)
  # 表头：去掉温度湿度
  header = ['Time_s', 'Odor', 'HCHO', 'CO', 'VOC', 'CO2', 'Label']
  self.csv_writer.writerow(header)
  self.csv_file.flush()
  self.file_start_time = time.time()
  self.log(f"创建CSV文件: {filename}", "success")

 def rotate_csv(self):
  if self.is_logging and self.file_start_time:
   if time.time() - self.file_start_time >= self.file_interval.get() * 60:
    self.log("达到切换时间，创建新文件...")
    self.create_csv()

 # 写入CSV：每次收到数据立即写入，无时间间隔过滤
 def try_write_record(self, record):
  label = self.label_text.get().strip()
  time_sec = record[8] // 1000
  # record[1:6] 对应 odor, hcho, co, voc, co2
  row = [time_sec] + record[1:6] + [label]
  self.csv_writer.writerow(row)
  self.csv_file.flush()
  self.log(f"✅ 写入CSV: Time={time_sec}s, CO2={record[5]}ppm", level="success")
  self.append_display(record)

 # ------------------ 多行数据块解析 ------------------
 def parse_block(self, lines):
  text = "\n".join(lines)
  def extract(pattern):
   match = re.search(pattern, text)
   return match.group(1).strip() if match else None

  odor = extract(r"Odor:\s+([\d\.]+)\s+ppm")
  hcho = extract(r"HCHO:\s+([\d\.]+)\s+ppm")
  co = extract(r"CO:\s+([\d\.]+)\s+ppm")
  voc = extract(r"VOC:\s+([\d\.]+)\s+ppm")
  co2 = extract(r"CO2:\s+(\d+)\s+ppm")
  env_temp = extract(r"环境温度:\s+([\d\.]+)\s+°C")
  humidity = extract(r"湿度:\s+([\d\.]+)\s+%")
  timestamp = extract(r"发送端时间戳:\s+(\d+)\s+ms")

  if None in [odor, hcho, co, voc, co2, env_temp, humidity, timestamp]:
   self.log(f"解析失败：缺少字段", level="error")
   return None

  try:
   record = [
    0,
    float(odor),
    float(hcho),
    float(co),
    float(voc),
    int(co2),
    float(env_temp),
    float(humidity),
    int(timestamp)
   ]
   return record
  except (ValueError, TypeError) as e:
   self.log(f"数值转换错误: {e}", level="error")
   return None

 # ------------------ 串口读取线程 ------------------
 def reader_thread(self):
  while self.is_logging and self.ser and self.ser.is_open:
   try:
    raw = self.ser.readline().decode('utf-8', errors='ignore').rstrip('\r\n')
    if not raw:
     continue
    self.root.after(0, lambda r=raw: self.log(f"RAW: {r}", level="data"))

    if raw.startswith("========== 接收到传感器数据 =========="):
     self.block_lines = []
     self.inside_block = True
     continue
    elif raw.startswith("======================================") and self.inside_block:
     record = self.parse_block(self.block_lines)
     if record:
      self.root.after(0, lambda: self.try_write_record(record))
     self.inside_block = False
     self.block_lines = []
     continue
    elif self.inside_block:
     self.block_lines.append(raw)

    self.root.after(0, self.rotate_csv)
   except serial.SerialException as e:
    self.root.after(0, lambda e=e: self.log(f"串口异常: {e}", "error"))
    break
   except Exception as e:
    self.root.after(0, lambda e=e: self.log(f"未知错误: {e}", "error"))
    break

 # ------------------ 开始/停止 ------------------
 def start_logging(self):
  if self.is_logging:
   return
  port = self.serial_port.get()
  if not port:
   messagebox.showerror("错误", "请选择串口号")
   return
  try:
   baud = int(self.baudrate.get())
  except ValueError:
   messagebox.showerror("错误", "波特率必须是数字")
   return

  try:
   self.ser = serial.Serial(port, baud, timeout=1)
  except Exception as e:
   messagebox.showerror("打开串口失败", str(e))
   return

  self.create_csv()

  # 同步采样间隔，确保最小2秒
  interval = self.record_interval.get()
  if interval < 2:
   interval = 2
   self.record_interval.set(interval)
  cmd = f"set interval {interval}"
  self.send_command_to_sensor(cmd)
  time.sleep(0.5)

  self.is_logging = True
  self.auto_refresh_enabled = False
  self.thread = threading.Thread(target=self.reader_thread, daemon=True)
  self.thread.start()

  self.start_btn.config(state=tk.DISABLED)
  self.stop_btn.config(state=tk.NORMAL)
  self.update_status(f"采集中 - {port}", "green")
  self.log(f"开始采集，传感器上报间隔={interval}秒（最小2秒），每收到一条数据立即保存", "success")

 def stop_logging(self):
  if not self.is_logging:
   return
  self.is_logging = False
  if self.ser and self.ser.is_open:
   self.ser.close()
  if self.csv_file:
   self.csv_file.close()
   self.csv_file = None
   self.csv_writer = None
  self.auto_refresh_enabled = True
  self.update_status("已停止", "gray")
  self.start_btn.config(state=tk.NORMAL)
  self.stop_btn.config(state=tk.DISABLED)
  self.log("采集已停止", "success")

 def on_closing(self):
  self.stop_logging()
  self.root.destroy()

if __name__ == "__main__":
 root = tk.Tk()
 app = SensorLogger(root)
 root.protocol("WM_DELETE_WINDOW", app.on_closing)
 root.mainloop()
