# YLJ / Luoyang 通用 Parquet 训练与评测

## 预测时距规则

- Luoyang 配置固定为 5 分钟颗粒度、未来 240 分钟，共 48 步：`t+5, ..., t+240`。
- YLJ 的总步数不写死，由 `240 / dataset.forecast_step_minutes` 推导。当前配置为 15 分钟，共 16 步：`t+15, ..., t+240`。
- 两个数据集的正式指标都通过导出记录的 `horizon_minutes` 选择 15 和 240 分钟，不使用数组下标。
- 配置步长必须能实际产生 t+15 和 t+240；真实 Parquet 的连续时间间隔必须与 `sampling_interval_minutes` 一致，否则数据集会报告期望和实际间隔并停止。目标不会插值、后向填充或伪造。

## 训练前必须补齐的配置

YLJ 的 [配置文件](../configs/ylj_parquet.yaml) 使用：

- `/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet`；
- 功率单位 MW，`power_scale`、`rated_power` 和 NRMSE/NMAE 分母均为 `468.0`；
- `observe_ghi-onsite` 为现场 GHI；`estimated_DNI-onsite`、`estimated_DHI-onsite` 为估算 DNI/DHI，已配置为辅助监督并在 metadata 中标明来源；
- 源 PWAT 列 `PWAT-NWP_observe` 的单位为毫米，乘以 `0.1` 转换为 REST2 使用的厘米；
- `observe_power` 的夜间负值按显式 `dataset.target_floor: 0.0` 截至物理下限；缺失功率不会插值或后向填充，缺少 issue 时刻或任一预测目标的样本会被拒绝。

首次运行前用 `scripts/inspect_ylj_parquet.py` 验证真实 schema、颗粒度、DNI/DHI 覆盖与 PWAT 分位数。检查不通过时脚本返回非零退出码；不会修改 Parquet，也不会启动训练。

Luoyang 的 [配置文件](../configs/luoyang_parquet.yaml) 已使用新文件、图片字段、48629.73 容量及经纬度 `34.700/112.285`、海拔 `220 m`、时区 `Asia/Shanghai`。`msl-NWP_forecast` 会根据海拔换算为站点气压。现场 GHI 与估算 DNI/DHI 均为每行五个分钟值及其时间戳；适配器按时间戳精确匹配每个主时间轴时刻，不使用固定列表下标，也不插值或后向填充。

Luoyang 可用 `scripts/inspect_luoyang_parquet.py` 复核 list 长度、有效值、内部时间戳偏移和字段覆盖率。

## 8 卡 V100S DDP

脚本根据 `WORLD_SIZE` 初始化 DDP；当前两个 Parquet 配置均使用 FP32，避免未经归一化的 Pa 量级 REST2 输入在 FP16 中溢出。train/validation 使用 `DistributedSampler`，每个 epoch 调用 `set_epoch`，只由 rank 0 写 checkpoint 和 resolved config。checkpoint 包含字段顺序、训练集归一化、容量、站点信息、数据 schema 和物理量来源。

CPU/单卡 dry-run 可使用 `--smoke`；2 进程 CPU/gloo 自动检查位于 `scripts/ddp_cpu_smoke.py`。完整运行命令只在根 `README.md` 维护。

## 分布式推理与输出

YLJ/Luoyang 只通过 `--config` 切换，不维护两套脚本。分布式预测会跨 rank 汇总，按 `issue_time,target_time` 去重并排序，只由 rank 0 原子写文件：

- `outputs/.../point_predictions.csv`：固定五列 `issue_time,target_time,horizon_minutes,y_true,y_pred`，数值均为原始功率单位；
- `outputs/.../official_test_metrics.json`：分别包含 t+15 和 t+240 的 overall、hard-delta、方向准确率、NRMSE/NMAE，以及 PCD 闭合、负辐照度和夜间非零诊断。

正式评测缺少 `evaluation.nrmse_denominator` 会明确停止；不会用测试集最大值替代容量。

## 图像和 DNI/DHI

YLJ 默认关闭图像，不创建图像编码器。Luoyang 从 `[issue_time-75min, issue_time]` 读取相对 `images.root` 的异步图片，排序去重，超过 16 张均匀抽取，不足部分补零并用 mask 标记。未来图片、重复图片时间戳和路径/时间列表长度不一致都会报错；图片文件缺失则保留零 mask，不影响 serial/physics 权重。

Luoyang 的 `GHI-onsite`、`estimated_DNI-onsite`、`estimated_DHI-onsite` 已启用辅助监督；list 中当前主时间轴时刻缺值时，其对应 target mask 为 0。整个 batch 没有有效辐照度标签时，辅助损失为同设备有限零标量。
