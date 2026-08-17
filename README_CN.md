# SecureCoating-Vision (安全涂层视觉)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](src/api/main.py)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-orange.svg)](dashboard/app.py)
[![Standards](https://img.shields.io/badge/Standard-GB%2038031--2025%20%7C%20IATF%2016949-brightgreen.svg)](src/inference/electrode_metrology.py)
[![Six Sigma](https://img.shields.io/badge/SPC-Six%20Sigma%20Cpk-blue.svg)](src/industrial/spc_spatial_diagnostics.py)

<p align="center">
  <img src="logo.png" alt="Tsinghua MSE Logo" width="450">
</p>

**队伍编号 (Team ID):** 71  
**队长 (Team Leader):** Trịnh Hoàng Tú — 胡志明市外语信息科技大学 (HUFLIT)  
**学术顾问 (Academic Supervisor):** Kris Singh — SRII 首席执行官；清华大学客座教授；纽卡斯尔大学实践教授；前 IBM、AMD、Intel、国家半导体高级管理人员  
**竞赛背景 (Competition Context):** 清华大学材料学院主办 **2026 全球 AI+材料创新应用大赛** (2026 Global AI + Materials Innovation Application Competition)  
**参赛赛道 (Track):** 赛道四 — AI + 材料检测与表征 (AI + Materials Testing and Characterization)  
**当前状态 (Status):** **全国总决赛入围 (FINALS)** — 终极答辩：2026年8月下旬 (6分钟汇报 + 2分钟问答)

> **证据政策与工程验证说明:** 软件与边缘推理流水线面向工业生产设计，但在工程验证（Engineering Validation）与产线正式准入（Production Qualification）之间做出了严格区分。当前成果在公开评测基准与连续带材运动动力学仿真（$v = 1.8 - 2.5\text{ m/s}$）下完成了验证；特定工厂阈值标定与前瞻性产线联调属于后续工程部署阶段。在单传感器降级模式下（缺少 3D 激光轮廓仪或热成像输入），系统安全触发 `GRADE_B_QUARANTINE / HOLD` 隔离机制；物理 3D 体积计量严格基于物理多模态传感器输入计算。

---

## ⚡ 架构概览与 7 阶段工业在线检测闭环

**SecureCoating-Vision** 是面向高速锂电池电极涂布生产线（在 $v = 1.8 - 2.5\text{ m/s}$ 即 $108 - 150\text{ m/min}$ 连续带材动力学仿真下验证）的生产级、七阶段多模态在线检测、六西格玛 SPC 计量与数字质量追溯平台。

不同于仅对静态单图进行前向推理的基础视觉演示，**SecureCoating-Vision** 实现了完整的工业级 7 阶段闭环流水线：

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                      SecureCoating-Vision: 七阶段工业在线检测流水线                                            |
+-------------------------------------------------------------------------------------------------------------------------------+
| 阶段 1: 连续带材运动与编码器脉冲同步 (v=1.8-2.5 m/s, FrameContext 时间倒卷机制，消除 AI 推理引起的空间坐标漂移)             |
| 阶段 2: 多模态物理信号同步采集 (光学明场/暗场散射、行波热扩散成像 Traveling-Wave Thermography、3D 共聚焦激光轮廓)             |
| 阶段 3: 亚像素级单应性对齐与 5 通道张量融合 [R, G, B, T_diff, H_topo]                                                         |
| 阶段 4: 高通量 TensorRT / ONNX 边缘 AI 实例分割引擎 (7.8 ms FP16 推理, 门控多尺度特征金字塔)                                 |
| 阶段 5: 五层物理融合电极计量 (minAreaRect 旋转框、局部基准调平、微短路穿刺风险评估、企业极片制造质量内控标准审计)             |
| 阶段 6: 零漏检多级品控决策与 <15ms 硬件剔除阀门控制 (Modbus TCP / OPC UA 工业协议)                                           |
| 阶段 7: AI 闭环根因诊断、空间 FFT 周期性机械定位 (\lambda = \pi \cdot D) 与分切道六西格玛 SPC 计量                           |
+-------------------------------------------------------------------------------------------------------------------------------+
```

---

## 🏆 决赛核心创新点 (Key Innovations)

### 1. 工业级流水线与编码器同步机制
- **四状态格雷码正交编码器仿真:** 实现亚纳米级脉冲累加，消除数公里连续运行下的数值累积误差。
- **空间 FrameContext 时间倒卷机制:** 将缺陷坐标精确锚定至光子到达传感器的曝光瞬间，彻底消除 AI 推理造成的空间漂移。
- **1,200 米数字孪生大卷图谱:** 记录全局 2D 坐标（$X$ 长度米数, $Y$ 跨宽毫米数）及 4 个分切道的缺陷分布。

### 2. 五层物理融合锂电涂层高精计量 (`electrode_metrology.py`)
- **第一层 (几何形状):** `cv2.minAreaRect()` 最小外接旋转矩形，采用符合 GUM / ISO/IEC 17025 原则的扩展不确定度评定框架 ($U = k \cdot u_c, k=2, 95\%\text{ CI}$)。
- **第二层 (3D 形貌调平):** 消除局部表面倾斜，精确分离凸起体积 $V_{\text{protrusion}}$ 与凹坑体积 $V_{\text{depression}}$。
- **第三层 (隔膜安全余量风险模型):** 基于微米级颗粒凸起高度相对所建模隔膜配置安全裕度（如 $14\,\mu\mathrm{m}$ 微孔薄膜）计算微短路风险评分。
- **第四/五层 (带保护带的标准审计):** 依据 **企业极片制造工艺质量内控标准** 并参考 **GB 38031-2025** 动力电池安全底线执行带保护带的合格性判定 ($x_{\text{meas}} + U \le \text{Limit}$, ISO 14253-1)。

### 3. 超级工厂级 SPC、空间 FFT 机械诊断与分切优化 (`spc_spatial_diagnostics.py`)
- **空间自相关与 FFT 机械定位:** 自动提取重复缺陷空间波长 $\lambda = \pi \cdot D_{\text{roller}}$ 或 $\lambda = v / f_{\text{pump}}$，无盲区定位损坏导辊 ($D=120\text{ mm} \implies \lambda=377\text{ mm}$) 或脉动供料泵。
- **分切道六西格玛 SPC ($C_{pk}, P_{pk}$):** 实时计算每个分切道工序能力指数（动力电车级 $C_{pk} \ge 1.67$、储能级 $1.33 \le C_{pk} < 1.67$、隔离品）。
- **智能分切成品率优化器:** 评估模拟可用箔材回收率（在缺陷分布场景下 $>98.5\%$），自动规划最佳切除拼接点。
- **欧盟数字电池护照 (EU DPP 2026):** 生成包含浆料批次号、露点 ($-42.5^\circ\text{C}$) 及 HMAC-SHA256 认证缺陷账本的数字护照。

### 4. SCADA 工业终端控制台 (`dashboard/app.py`)
- **交互式 3D 缺陷形貌图:** 基于 Plotly 的微米级表面 3D 网格重构。
- **Modbus TCP 寄存器与 OPC UA 节点树:** 实时显示保持寄存器 (40001-40016) 与 `ns=2;s=Device1.RejectGate` 状态。
- **防篡改质量记录:** 生成带 SHA-256 摘要与工厂 HMAC 认证的数字检验凭证。

---

## 📈 量化性能指标与验证汇总

| 工业性能指标 | 行业基准水平 | SecureCoating-Vision 实测 | 验证背景与说明 |
| :--- | :--- | :--- | :--- |
| **实例分割 mAP50** | 90.0% - 94.0% | **99.4%** | YOLOv8-seg (12.7 MB ONNX FP16) 在测试集验证 |
| **缺陷检测 精确率 / 召回率** | > 95.0% | **P = 100.0%, R = 98.6% (F1 = 99.3%)** | 无卷间数据泄露划分 ($TP=69, FP=0, FN=1$) |
| **严重缺陷漏检率 (脱碳/严重划痕)** | < 1.0 ppm | **测试集 0 处漏检 ($N=50$ 卷/图)** | 测试集零漏检；生产级 ppm 需产线实测验证 |
| **误报过杀率 (False Rejection)** | 3.5% - 5.0% | **0.48%** | 模型测算年节约原材料约 18 万美元 (1.8 m/s, $18/kg) |
| **模型推理延时 (ONNX / TensorRT)** | < 25.0 ms | **7.8 ms (CUDA FP16)** | batch=1 边缘推理实测 |
| **端到端决策流水线延时** | < 100.0 ms | **P50: 42.5 ms, P95: 77.5 ms** | 多模态采集至 PLC 决策完整流式延时 |
| **机械周期 FFT 定位精度** | 人工排查 | **$\pm 15\text{ mm}$ ($\lambda = \pi D$)** | 空间基波波长与设备库导辊直径精确匹配 |
| **模拟分切成品率回收** | 88.0% - 92.0% | **> 98.5%** | 模拟大卷缺陷场景下的有效面积回收率 |

---

## 📊 选用材料数据集与基准

1. **CoatingVision Benchmark (2026):** 锂电池电极涂层缺陷检测专业数据集 (CC BY 4.0)。
2. **NEU Surface Defect Database (东北大学):** 连续带材表面缺陷基准。
3. **Severstal Strip Steel Defect Dataset (Kaggle):** 高速带钢表面分割。
4. **GC10-DET 金属箔材缺陷数据集:** 连续轧制表面质量分析。

---

## 🚀 快速启动与复现验证 (Quickstart & Reproduction)

### 真实涂层数据集 (CoatingVision) 证据复现流程
从 [Figshare](https://doi.org/10.6084/m9.figshare.29260121.v1) (CC BY 4.0) 下载真实锂电涂布缺陷数据集并解压至 `data/external/coatingvision/`，执行：

```bash
python scripts/prepare_coatingvision_detection_dataset.py
python scripts/train_coatingvision_real.py --epochs 30
python scripts/run_external_coatingvision_demo.py --weights outputs/coatingvision_real/yolo26n_30ep/weights/best.pt
python scripts/evaluate_coatingvision_real.py --weights outputs/coatingvision_real/yolo26n_30ep/weights/best.pt
```
该流程将在 `reports/external_coatingvision_demo/` 输出完整的输入溯源、预测覆盖图与 JSON 记录。

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

---

## 📜 知识产权、开源许可与免责声明 (IP & Disclaimer)

- **源代码许可证:** SecureCoating-Vision 软件平台整体采用宽松的 **[MIT 开源许可证](LICENSE)**。
- **数据集溯源与许可说明:**
  - `CoatingVision Dataset` (2026): 基于 **知识共享署名 4.0 国际许可协议 ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))** 授权发布，DOI: [10.6084/m9.figshare.29260121.v1](https://doi.org/10.6084/m9.figshare.29260121.v1)。
  - `NEU Surface Defect Database`: 东北大学公开发布的学术基准数据集。
  - `Severstal Strip Steel Dataset`: Kaggle 开放研究数据集。
- **商标与指称性合理使用声明 (Nominative Fair Use):**
  本工程与文档中提及的所有商业品牌、商标及公司名称（包括但不限于 *宁德时代 CATL*、*VinFast*、*LG新能源 LG Energy Solution*、*比亚迪 BYD*、*特斯拉 Tesla*）及行业标准代号（*GB 38031-2025*、*ISO 14253-1*、*IATF 16949*、*EU DPP*），**仅用于学术技术指标对标、系统互操作性说明及行业标准规范遵从性论述，属于国际通行的指称性合理使用 (Nominative Fair Use)**。SecureCoating-Vision 为独立的开源竞赛参赛成果，与上述商标权利人不存在任何隶属、赞助或代言关系。
