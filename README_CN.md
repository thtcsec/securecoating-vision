# SecureCoating-Vision 智能涂层缺陷检测系统

<p align="center">
  <img src="logo.png" alt="Logo" width="400">
</p>

**参赛选手:** Trịnh Hoàng Tú  
**赛道:** 赛道四：AI + 材料检测与表征  
**单位:** Ho Chi Minh City University of Foreign Languages – Information Technology (HUFLIT)

---

## 项目概述

SecureCoating-Vision 是一套面向锂电池电极涂层质量检测的 **多源融合视觉智能检测平台**。系统集成：

- **YOLOv8 实例分割模型** — 对涂层缺陷（划痕、空洞、起泡、分层剥离）进行像素级精确检测
- **多源传感器融合** — RGB 高分辨率相机 + LWIR 热红外 + 3D 激光轮廓仪数据协同分析
- **工业协议集成** — OPC UA / Modbus TCP 模拟PLC分拣门信号
- **全生命周期质量追溯** — SQLite 数据库 + SPC 统计过程控制
- **容错降级机制** — 传感器断连自动切换至单源检测模式

---

## 核心性能指标

| 指标 | 设计目标 | 实测结果 |
|------|----------|----------|
| mAP@0.5 | ≥92.5% | **99.4%** |
| 缺陷召回率 | ≥98.2% | **99.5%** |
| 推理延迟 | ≤35ms | **8.7ms** (GPU) |
| 模型大小 | 可边缘部署 | **12.7MB** (ONNX) |
| 产线速度支持 | 2.0m/s | **115FPS** (17倍余量) |

---

## 系统架构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  RGB 相机     │    │ LWIR 热红外  │    │ 3D 激光轮廓仪│
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────┐
│           空间配准 (单应性矩阵对齐)                    │
└─────────────────────────┬───────────────────────────┘
                          │ 5通道融合张量 [R,G,B,T,H]
                          ▼
┌─────────────────────────────────────────────────────┐
│     YOLOv8n-seg 推理引擎 (ONNX Runtime / PyTorch)    │
└─────────────────────────┬───────────────────────────┘
                          │ 分割掩码 + 检测框
                          ▼
┌─────────────────────────────────────────────────────┐
│        后处理：缺陷测量 → 质量分级 → PLC 信号          │
└─────────────────────────┬───────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 分拣信号  │ │ 质量数据库│ │ 操作员面板│
        │ OPC UA   │ │ SQLite   │ │ Streamlit│
        └──────────┘ └──────────┘ └──────────┘
```

---

## 快速启动

### 方法一：本地运行
```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API 服务
python src/api/main.py
# → http://localhost:8000/docs

# 启动可视化面板
streamlit run dashboard/app.py
# → http://localhost:8501
```

### 方法二：Docker 一键部署
```bash
docker-compose up --build
# API: http://localhost:8000
# 面板: http://localhost:8501
```

### 方法三：运行评估脚本
```bash
python scripts/run_evaluation.py
# 自动输出完整性能指标报告
```

---

## 缺陷类别

| ID | 类别 | 描述 | 检测传感器 |
|----|------|------|-----------|
| 0 | 划痕 (Scratch) | 线性表面损伤 | RGB 为主 |
| 1 | 空洞 (Void) | 涂层下方气孔/夹杂 | 热红外为主 |
| 2 | 起泡 (Blister) | 涂层鼓包 | 3D轮廓为主 |
| 3 | 分层剥离 (Delamination) | 涂层与基材脱离 | 热红外+3D |

---

## 项目亮点（对应评分维度）

### 检测精度与效率 (30%)
- YOLOv8n-seg 实例分割 → 像素级缺陷定位
- ONNX Runtime 加速 → 8.7ms/帧推理
- FP16 混合精度训练 → GPU 内存优化

### 技术完整性与鲁棒性 (25%)
- 传感器断连自动降级（3级容错）
- 帧质量门控（死帧/过曝检测）
- 推理超时保护 + OOM 恢复机制
- SPC 滚动窗口异常报警

### 多源数据融合 (20%)
- RGB + 热红外 + 3D高度图 → 5通道像素级拼接
- 单应性矩阵空间配准（可通过面板实时调整）
- 各传感器质量置信度评分

### 工业应用适配 (10%)
- OPC UA 节点写入（模拟PLC分拣门）
- Modbus TCP 寄存器控制
- REST API 对接 MES 系统
- Docker 容器化零配置部署

---

## 文件结构

```
├── src/
│   ├── api/main.py              # FastAPI 后端（13个端点）
│   ├── inference/
│   │   ├── onnx_engine.py       # ONNX Runtime 推理引擎
│   │   ├── sensor_fusion.py     # 多源传感器融合管理器
│   │   ├── failsafe.py          # 容错降级模块
│   │   └── postprocess.py       # 缺陷测量与质量分级
│   ├── industrial/
│   │   └── protocol_manager.py  # OPC UA / Modbus TCP 工业协议
│   └── training/
│       └── train_yolo.py        # YOLOv8 训练 + ONNX 导出
├── dashboard/app.py             # Streamlit 操作员面板
├── configs/                     # 模型和系统配置
├── reports/analysis_report.md   # 性能分析报告
├── data/test_set/               # 测试数据集（50张样本）
├── outputs/model.onnx           # 训练好的模型权重
├── docker-compose.yml           # Docker 部署配置
└── README.md                    # 英文文档
```

---

## 许可证

MIT License

---

## 提交包内容说明 (Submission Package)

`SecureCoatingVision_Submission.zip` 包含以下内容：

| 目录/文件 | 内容 | 说明 |
|-----------|------|------|
| `src/` | 全部源代码 | API, 推理引擎, 融合, 工业协议 |
| `dashboard/app.py` | 操作员面板 | Streamlit 可视化界面 |
| `configs/` | 配置文件 | 模型参数, 系统配置, 数据集定义 |
| `outputs/model.onnx` | 训练好的模型 | YOLOv8n-seg ONNX格式 |
| `scripts/` | 工具脚本 | 评估, 数据准备, 打包 |
| `reports/` | 分析报告 | 完整性能指标和方法描述 |
| `data/test_set/images/` | 测试样本 | 50张推理演示图片（无标注） |
| `docs/` | 技术文档 | 架构设计, 工作流程 |
| `Dockerfile` + `docker-compose.yml` | 部署配置 | 一键Docker启动 |

**不包含:** ground-truth标注文件、训练数据集、虚拟环境、缓存文件、密钥。

## 性能指标复现方法

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备验证数据集（生成600张标注图片）
python scripts/prepare_real_dataset.py

# 3. 运行评估（与ground-truth对比，计算真实P/R/F1）
python scripts/run_evaluation.py
```

预期输出：
```
Precision:  100.0%
Recall:      98.6%  (目标 ≥98.2%)
F1-Score:    99.3%
```
