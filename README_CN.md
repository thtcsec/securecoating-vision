# SecureCoating-Vision 智能涂层缺陷多源融合检测系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](src/api/main.py)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-orange.svg)](dashboard/app.py)
[![Battery Standard](https://img.shields.io/badge/Standard-T%2FCIAPS%200006--2020-brightgreen.svg)](src/inference/electrode_metrology.py)

<p align="center">
  <img src="logo.png" alt="Logo" width="450">
</p>

**团队编号 (Team ID):** 71  
**参赛选手 (Team Leader):** Trịnh Hoàng Tú  
**学术导师 (Academic Supervisor):** Kris Singh — CEO, SRII；清华大学访问教授；澳大利亚纽卡斯尔大学实践兼职教授；曾任 IBM / AMD / Intel / National Semiconductor 高管  
**参赛赛道 (Track):** 赛道四：AI + 材料检测与表征 (AI + Materials Testing and Characterization)  
**参赛单位 (Affiliation):** Ho Chi Minh City University of Foreign Languages – Information Technology (HUFLIT)  
**当前状态 (Status):** **全国总决赛入围 (FINALS)** — 终审答辩：2026年8月下旬（6分钟陈述 + 2分钟问答）

---

## ⚡ 项目核心概述与工业全流程架构

**SecureCoating-Vision** 是一套专为 **锂电池极片高速涂布产线 ($v = 1.8 - 2.5\text{ m/s}$)** 设计的 **多源融合视觉智能检测与闭环质量追溯平台**。

与仅停留在单张静态图片推断的传统学术Demo不同，**SecureCoating-Vision** 实现了工业级 **七工位全流程在线闭环质检架构**：

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                      SecureCoating-Vision: 7工位工业在线质检全流程流水线                                         |
+-------------------------------------------------------------------------------------------------------------------------------+
| 工位 1: 连续卷料卷径与增量正交编码器同步 (v=1.8 m/s, X米沿卷长 / Y毫米跨宽物理坐标映射)                                            |
| 工位 2: 多源物理传感采集 (光学明场/暗场散射, 动态行波热扩散反演, 3D共聚焦激光测厚)                                                  |
| 工位 3: 亚像素单应性空间矩阵对齐与5通道张量拼接 [R, G, B, T_diff, H_topo]                                                       |
| 工位 4: 高通量边缘 AI TensorRT / ONNX 实例分割推理引擎 (FP16精度, 门控多尺度特征金字塔)                                            |
| 工位 5: 融入物理机理的电池极片定量表征 (3D体积积分, 辊压后微短路风险指数, T/CIAPS 0006 / QC/T 743 行业标准审计)                   |
| 工位 6: 多级质量判定与亚10毫秒硬件气动分拣门信号联动 (Modbus TCP / OPC UA)                                                       |
| 工位 7: AI 闭环根因诊断与上游涂布设备参数反调建议 (涂布缝隙 \Delta h, 烘箱温度 \Delta T, 泵转速 \Delta Q)                             |
+-------------------------------------------------------------------------------------------------------------------------------+
```

---

## 🏆 决赛核心技术创新点 (Finals Innovations)

### 1. 连续卷料运动控制与正交编码器同步 (`src/industrial/web_synchronizer.py`)
- **正交 A/B 编码器亚脉冲分数累积：** 彻底消除高速长卷运行数公里后的累积脉冲离散化漂移。
- **卷料全生命周期状态机：** 完整实现 `IDLE`, `LOADED`, `RUNNING`, `PAUSED`, `ROLL_CHANGE`, `COMPLETED`, `FAULT` 状态流转。
- **1,200米大卷数字孪生缺陷图谱：** 精准记录每一个缺陷的绝对物理坐标 $(X_m, Y_{mm})$ 及分切条道分布（Lane 1-4）。

### 2. 融入电池电化学机理的定量物理表征 (`src/inference/electrode_metrology.py`)
- **隔膜微短路危险指数 ($0.0 - 1.0$)：** 引入辊压压缩率模型，评估极片硬颗粒突起高度与隔膜安全厚度极限 ($14\,\mu\mathrm{m}$) 的比例，防止电芯卷绕与辊压时刺穿隔膜引发内部微短路。
- **面密度波动分析 ($\mathrm{mg/cm^2}$)：** 评估局部活性物质涂布量偏差，防止充放电局部极化与析锂。
- **行业标准自动审计：** 内嵌 **T/CIAPS 0006-2020（锂离子电池电极片通用技术规范）**、**QC/T 743**、**GB 38031-2025** 动力蓄电池安全要求条款。

### 3. AI 闭环设备根因诊断与工艺反调 (`src/traceability/root_cause_engine.py`)
- **狭缝涂布头 / 刮刀：** 纵向划痕/条纹 $\to$ 触发涂布头超声波清洗并驱动刮刀伺服侧移 ($\Delta y = +2.5\text{ mm}$)。
- **双行星浆料搅拌脱泡机：** 周期性微孔/空洞 $\to$ 建议提高真空脱泡度 ($\Delta P = -8.0\text{ kPa}$)。
- **多段式气浮烘箱：** 表面结皮起泡/脱落 $\to$ 建议一区温度缓降 ($\Delta T_1 = -4.0^\circ\mathrm{C}$) 并开大排风排湿阀门 ($+8\%$)。

### 4. Keyence/Cognex 工业 SCADA 控制台 (`dashboard/app.py`)
- **交互式 3D 缺陷微观形貌：** 基于 Plotly 3D Surface 渲染微米级表面形貌、断层切片与体积位移。
- **1,200米连续瀑布流大卷图谱：** 实时展示全卷缺陷密度，支持点击任意坐标即时回放多传感器图像。
- **工业协议与硬件遥测：** 实时 Modbus TCP 保持寄存器十六进制表 (40001-40016) 与 OPC UA 命名空间节点树。
- **SHA-256 防篡改数字质量证书：** 一键生成附带加密防篡改签名的极片大卷出厂检验报告。

---

## 📊 选用材料数据集与基准

1. **CoatingVision Benchmark (2026):** 锂电池电极涂层缺陷检测专业数据集。
2. **NEU Surface Defect Database (东北大学):** 连续带材表面缺陷基准（包含裂纹、夹杂、麻坑、划痕）。
3. **Severstal Strip Steel Defect Dataset (Kaggle):** 高速带钢表面分割。
4. **GC10-DET 金属箔材缺陷数据集:** 连续轧制表面质量分析。

---

## 🚀 快速启动与复现

### 1. 启动工业 SCADA 控制台
```bash
streamlit run dashboard/app.py
```
浏览器打开：`http://localhost:8501`

### 2. 启动 REST API 后端服务
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Swagger 接口文档：`http://localhost:8000/docs`

### 3. 运行自动化单元测试 (100% 通过)
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 4. 生成无泄露竞赛提交包
```bash
python scripts/build_submission.py
```
生成安全校验通过的 ZIP 提交文件：`SecureCoatingVision_Submission.zip`。
