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
- `DNI_observe`、`DHI_observe` 已配置为辅助监督；
- 源 PWAT 单位为毫米，`PWAT_observe * 0.1` 转换为 REST2 使用的厘米。

首次在服务器运行前，使用只读检查脚本验证真实 schema、15 分钟颗粒度、DNI/DHI 覆盖与 PWAT 分位数：

```bash
python scripts/inspect_ylj_parquet.py \
  --parquet /data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet \
  --pwat-unit mm \
  --output-json outputs/ylj_parquet_inspection.json
```

检查不通过时脚本返回非零退出码；不会修改 Parquet，也不会启动训练。

Luoyang 的 [配置文件](../configs/luoyang_parquet.yaml) 已包含 Parquet、图片路径和 48629.73 容量，但仍需填写经验证的 `site.latitude`、`site.longitude`、`site.altitude_m` 与 `site.timezone`。缺失经纬度或时区时，配置静态解析可以通过，首次构建太阳几何样本会明确失败，不会借用其他站点参数。缺少海拔时不会把 MSL 静默当成站点气压，而使用 `physics_defaults.pressure_pa` 并在 checkpoint metadata 中记录来源为 default。

## 8 卡 V100S DDP

在 Linux 服务器仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/train_parquet.py \
  --config configs/ylj_parquet.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/train_parquet.py \
  --config configs/luoyang_parquet.yaml
```

脚本根据 `WORLD_SIZE` 初始化 DDP；V100 使用 FP16 autocast 与 GradScaler，不启用 BF16。train/validation 使用 `DistributedSampler`，每个 epoch 调用 `set_epoch`，只由 rank 0 写 checkpoint 和 resolved config。checkpoint 包含字段顺序、训练集归一化、容量、站点信息、数据 schema 和物理量来源。

CPU 单进程 dry-run 可使用 `--smoke`。自动化的 2 进程 CPU/gloo 前向、反向和参数更新检查为：

```bash
python scripts/ddp_cpu_smoke.py
```

## 分布式推理与输出

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/evaluate_parquet.py \
  --config configs/luoyang_parquet.yaml \
  --checkpoint outputs/luoyang_parquet/checkpoint_last.pt \
  --output-dir outputs/luoyang_official_test
```

也可仅通过 `--config configs/ylj_parquet.yaml` 切换到 YLJ，不维护第二套脚本。分布式预测会跨 rank 汇总，按 `issue_time,target_time` 去重并排序，只由 rank 0 原子写文件：

- `outputs/.../point_predictions.csv`：固定五列 `issue_time,target_time,horizon_minutes,y_true,y_pred`，数值均为原始功率单位；
- `outputs/.../official_test_metrics.json`：分别包含 t+15 和 t+240 的 overall、hard-delta、方向准确率、NRMSE/NMAE，以及 PCD 闭合、负辐照度和夜间非零诊断。

正式评测缺少 `evaluation.nrmse_denominator` 会明确停止；不会用测试集最大值替代容量。

## 图像和 DNI/DHI

YLJ 默认关闭图像，不创建图像编码器。Luoyang 从 `[issue_time-75min, issue_time]` 读取相对 `images.root` 的异步图片，排序去重，超过 16 张均匀抽取，不足部分补零并用 mask 标记。未来图片、重复图片时间戳和路径/时间列表长度不一致都会报错；图片文件缺失则保留零 mask，不影响 serial/physics 权重。

DNI/DHI 配置为 null 时，其 target mask 为 0；整个 batch 没有有效辐照度标签时辅助损失为同设备有限零标量。以后只需在配置的 `irradiance_columns` 中填写列名即可启用相应监督。
