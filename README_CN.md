# SecureCoating-Vision（安全涂层视觉）

本项目是用于涂层缺陷分割、失效闭锁决策、卷材同步仿真、PLC 协议集成和质量追溯研究的**工业计算机视觉原型**，目前不是经过产线认证的生产系统。

仓库中没有工厂标定证书、PLC 硬件在环验证、独立卷批测试集或安全机构批准。因此：

- 不得把软件 E-stop 当作安全等级急停；
- 不得把模拟 OPC UA/Modbus 写入当作 PLC ACK；
- 不得把当前开发评测结果宣传为独立测试集、零漏检、ppm、六西格玛或标准认证结果；
- 未训练模型、推理超时/异常、传感器缺失、标定未验证、数据库故障、PLC 通信失败和联锁锁存均必须输出 `HOLD`。

当前 `data/evaluation` 与训练验证集存在文件哈希重叠，因此其中的报告只能作为开发调试记录，不能作为泛化性能证明。英文 [README](README.md) 记录了真实的安全边界、配置、测试、训练和评测要求。

## 本地测试模式

```powershell
$env:SECURECOATING_ENV = "development"
$env:SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO = "true"
$env:SECURECOATING_ENABLE_SENSOR_SIMULATION = "true"
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

该模式仅限可信开发工作站，不得对外网开放。Dashboard 是本地只读仿真视图，不拥有真实 PLC 控制权。

## 实施状态与升级计划

分阶段功能清单、验证证据、发布门槛和后续计划见 [docs/implementation_status.md](docs/implementation_status.md)。文档明确区分代码测试、本地仿真、硬件在环和工厂验证，避免把未验证内容当作已完成能力。

## 验证命令

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src dashboard scripts tests
.venv\Scripts\python.exe -m pip check
docker compose config
```

生产模式必须配置 API 密钥、证书签名密钥、真实传感器标定、OPC UA `SignAndEncrypt` 证书和受控 Modbus 网关，并完成厂商 PLC/HIL 与功能安全验证。
