# SecureCoating-Vision（安全涂层视觉）

<p align="center">
  <img src="logo.png" alt="SecureCoating-Vision" width="450">
</p>

[![Track 4 Finalist](https://img.shields.io/badge/清华大学材料学院2026-赛道四全国总决赛入围-C8102E.svg)](README_CN.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English README](README.md)

**队伍 71** · 队长 Trịnh Hoàng Tú — 胡志明市外语信息科技大学（HUFLIT）<br>
**学术顾问：** Kris Singh 教授 — SRII / 清华大学客座教授<br>
**竞赛：** 2026 全球 AI+材料创新应用大赛 · 赛道四（AI + 材料检测与表征）

注册英文名称、版本和口号以 [README.md](README.md) 与
[`configs/project_identity.yaml`](configs/project_identity.yaml) 为唯一来源。

SecureCoating-Vision 是一个面向卷对卷涂布场景的**证据感知、失效闭锁工业视觉原型**。
它把真实光学图像、公开标签、独立 AI 检测、可选多模态输入、卷材追溯和 PLC
安全门控连接为一条可审计流程。当前仓库不是经产线认证的生产系统，也不宣称
“零漏检”、ppm、六西格玛、安全等级急停或工厂节拍达标。

<!-- TEST_MANIFEST:START -->
当前软件验证快照为 271 项测试通过、0 项跳过、0 项失败（提交 `b9e33bd8763d`，工作树干净，源码差异 `none`，Python 3.11.9，耗时 198.79 秒）。权威记录为 [reports/test_manifest.json](reports/test_manifest.json)。该记录仅证明当前软件测试结果，不代表工厂性能、真实 PLC 行为、安全等级急停认证或生产资质。
<!-- TEST_MANIFEST:END -->

## 评委可复现的本地演示

默认 Docker Compose 使用保守 CPU 路径：

```powershell
docker compose up --build -d
```

打开 `http://127.0.0.1:8501`。Dashboard 的运行时卡片会显示**实际启用**的配置，
例如 `EDGE · ONNX CPU · FP32 · 512px`。历史记录按同一采集帧展示：
原始 JPEG → 公开像素掩膜 → 公开标签热图 → 独立 AI 检测叠加层。

NVIDIA 主机已安装 NVIDIA Container Toolkit 时可运行：

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

只有实时卡片明确显示 TensorRT 时，演示讲解才可以称其为 TensorRT 路径；若本机缺少
TensorRT 动态库，系统会记录原因并回退到 YOLO CUDA FP16。详见
[自适应推理配置](docs/hardware_profiles.md)。

## 自适应硬件策略

| 配置 | 目标硬件 | 首选运行时 | 典型输入 | 设计意图 |
|---|---|---|---:|---|
| `edge` | 低功耗 CPU / 工控机 | ONNX CPU FP32 | 512 px | 稳定、低资源占用 |
| `balanced` | 中档 GPU | TensorRT/CUDA，失败则安全回退 | 640 px | 精度与速度平衡 |
| `performance` | 高性能 GPU | TensorRT/CUDA FP16 | 768 px | 展示更高吞吐能力 |
| `auto` | 任意主机 | 根据可用 provider 自动选择 | 自动 | 默认推荐 |

健康接口和 Dashboard 同时公开请求配置、实际配置、provider、实际精度、图像尺寸与回退原因，
避免把配置意图误报为真实运行状态。

## 数据集与证据等级

| 数据源 | 用途 | 必须说明的边界 |
|---|---|---|
| Argonne CoatingVision | 真实光学图像、公开掩膜和多标签类别 | 当前检测报告按图像隔离，不等于工厂卷批独立测试 |
| LIBAD 官方本地挂载 | 真实卷对卷电极制造的对齐 VIS + X-rayL 验证输入 | 本地适配器不是作者的 DINOv3/DA-Core，不可与论文指标直接比较 |
| 合成配对涂层数据 | 开发、故障注入、接口和安全契约测试 | 永远不作为真实产线性能 |
| 热成像 / 轮廓仪适配器 | 配准与失效闭锁接口验证 | 未连接并标定工厂仪器时属于仿真或注入输入 |

Git 不收录 4.84 GB LIBAD 官方数据。可用 `scripts/download_libad.py` 在本地挂载，
再运行 `scripts/record_libad_official_manifest.py` 固化文件树哈希。已跟踪的
`reports/libad/libad_benchmark.json` 是确定性 `protocol_fixture`；官方输入上的本地
numpy 结果位于 `reports/libad/official_local_adapter.json`，始终保持
`comparable_to_paper: false`。详见 [LIBAD 验证扩展](docs/libad_validation_extension.md)。

## 安全决策契约

- 只有已加载训练模型产生的 `OPTIMAL` 结果，才可能进入自动 PASS/REJECT 决策。
- 模型未加载、推理异常/超时、传感器缺失、标定未验证、数据库故障、PLC 通信失败或联锁锁存，一律输出 `HOLD_REQUIRED`。
- 软件 E-stop 只是请求与状态机，不是安全等级硬件急停。
- 模拟 OPC UA/Modbus 写入不等于厂商 PLC ACK，也不能替代硬件在环验证。
- AI 检测不会覆盖安全门；真实传感器、PLC 就绪和有效标定缺一不可。
- 公共标签、合成数据和模型预测分别存档，不能混合为一个没有证据来源的指标。

## 配置与运行

安装锁定依赖并以显式本地演示模式启动：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$env:SECURECOATING_ENV = "development"
$env:SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO = "true"
$env:SECURECOATING_ENABLE_SENSOR_SIMULATION = "true"
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

该模式只允许在可信开发工作站使用，不得暴露到公网。Dashboard 是 API 客户端，
不直接拥有 PLC 控制权；Compose 固定一个 API worker，因为 PLC、推理和活动卷材状态是有状态资源。

生产模式至少需要：

```powershell
$env:SECURECOATING_ENV = "production"
$env:SECURECOATING_API_KEY = "<至少 32 字节的随机密钥>"
$env:SECURECOATING_FACTORY_SECRET = "<至少 32 字节的随机签名密钥>"
$env:SECURECOATING_ALLOWED_ORIGINS = "https://operator.example.com"
$env:SECURECOATING_TRUSTED_HOSTS = "operator.example.com,api"
```

生产启动会拒绝缺失/过短密钥、通配 CORS、通配可信主机、格式错误的 origin，以及非回环地址的
HTTP origin。操作员控制还要求 `SECURECOATING_OPERATOR_CREDENTIALS` 中存放哈希 token；
禁止把明文 token 写入仓库。

## 已加入的应用与容器加固

- API 对成功、认证失败和请求体超限响应统一增加 `no-store`、禁止 framing、禁止 MIME sniffing、严格 CSP、Permissions Policy 和跨域资源策略。
- HTTPS 请求才返回 HSTS；TLS 终止仍由反向代理或平台负责。
- 上传限制字节数、像素数、允许的图像格式，并限制并发检测，防止资源耗尽。
- Compose 使用非 root 用户、只读根文件系统、删除 Linux capabilities、`no-new-privileges`、PID/文件描述符上限、受限 `tmpfs`、优雅停机和日志轮换。
- 服务端口默认只绑定回环地址；真实部署仍需网络分区、TLS、镜像签名、SBOM/漏洞扫描、备份以及 PLC/HIL 验证。
- CI 使用只读仓库权限、不保留 checkout 凭证，并验证 CPU 与 NVIDIA 两套 Compose 配置。

## 完整验证

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/record_test_manifest.py
.venv\Scripts\python.exe -m compileall -q src dashboard scripts tests
.venv\Scripts\python.exe -m pip check
python -m ruff check .
docker compose config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
```

LIBAD 官方挂载验证：

```powershell
.venv\Scripts\python.exe scripts/record_libad_official_manifest.py
.venv\Scripts\python.exe scripts/run_libad_benchmark.py --require-official --out reports/libad/official_local_adapter.json
```

## 演示时应该怎么说

可以说：系统支持 CPU/GPU 自适应运行；当前运行时由实时卡片证明；真实图像、公开标签、AI 结果
和安全门控分别可追溯；缺少物理证据时系统主动 HOLD。

不要说：已经通过产线认证、软件 E-stop 是安全等级急停、模拟 PLC 是真实 PLC、合成数据代表真实
精度，或本地 LIBAD 适配结果等价于论文方法。

分阶段实现、证据与发布门槛见 [实施状态](docs/implementation_status.md)。许可协议见 [MIT License](LICENSE)。
