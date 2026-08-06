# REST2–PCD融合模型：YLJ/Luoyang双Parquet迁移任务

## 1. 工作目录与任务目标

工作目录：

```text
F:\多模态时序大模型\华为深圳测试\REST2_physics_fusion_Aligned_Model
```

目标是在不改变REST2、PCD、现有MoE、功率头及融合数学主干的前提下，完成：

1. 用一个通用、配置驱动的Parquet数据接口支持YLJ和Luoyang；
2. YLJ和Luoyang分别使用独立配置文件，字段、路径、维度、时间范围、物理默认值、
   归一化、容量和指标阈值均不能散落硬编码；
3. 两个数据集都预测到未来4小时，但输出步数由各自时间颗粒度决定：Luoyang为
   5分钟48步，YLJ不固定48步；
4. 导出与PVMMoE洛阳实验一致的逐点预测文件和指标JSON；
5. 图像模态由配置开关控制，并支持部分缺失、全部缺失和异步时间戳；
6. 支持8张Tesla V100S 32GB的DDP训练和分布式推理；
7. 只运行CPU轻量测试、形状测试和DDP smoke设计，不运行完整训练；
8. 最终只提交并推送到远程新分支 `why`，不得修改或推送原分支。

预测目标始终是功率。DNI/DHI后续会加入数据集，本次必须先把可选字段、掩码和
辅助损失接口写好；当前缺失时不能阻止纯功率训练。

## 2. Git现状与强制交付规则

当前仓库：

```text
origin: https://github.com/xinhuaHan/REST2_physics_fusion.git
current branch: Aligned_Model
current commit: 095899d
```

当前工作树已经存在与本任务相关、尚未提交的可选图像模态修改，包括：

```text
.gitignore
configs/base.yaml
docs/LUOYANG_IMAGE_ADAPTER_ZH.md
src/pv_physics_moe/config.py
src/pv_physics_moe/data/dataset.py
src/pv_physics_moe/models/encoders.py
src/pv_physics_moe/models/fusion.py
src/pv_physics_moe/models/integrated.py
tests/test_integrated_model.py
tests/test_station_server_pipeline.py
agent.md
```

开始工作前必须执行 `git status --short --branch` 和 `git diff`，逐项审查并保留这些
相关改动。不得用reset、checkout覆盖、stash丢弃或重新克隆来绕过现有工作树。

在继续编码前切换到新分支：

```bash
git fetch origin
git ls-remote --heads origin why
git switch -c why
```

如果本地或远程已经存在 `why`，不要强制覆盖、删除或force push；先检查该分支是否
属于本任务，再安全切换或报告冲突。任何提交必须发生在 `why`，不得在
`Aligned_Model`、默认分支或其他分支提交。

用户已经明确授权：为本任务创建 `why` 分支、提交本任务代码并推送
`origin/why`。没有授权创建PR、合并PR、修改其他分支、删除分支或force push。

提交前必须：

- 检查完整的staged/unstaged diff；
- 只用 `git add -- <明确文件列表>` 暂存本任务文件；
- 禁止使用 `git add .`、`git add -A`、`git add --all`；
- 不提交Parquet、图片、权重、checkpoint、日志、输出结果、cache、虚拟环境、
  `__pycache__`、密钥或服务器隐私文件；
- 提交后推送 `git push -u origin why`；
- 验证本地HEAD与 `origin/why` 完全一致；
- 最终报告远程URL、分支、提交哈希、修改文件、测试结果和运行命令。

## 3. 不允许修改的核心算法

本任务不是重新设计算法。以下内容只能为适配张量维度做必要的外围连接，不能改变
数学含义、网络顺序、损失定义或核心结构：

- `Rest2FeatureBuilder`和REST2清空计算；
- `SolarPhysicsStack`、PCD可行域投影和
  `GHI = cos(zenith) * DNI + DHI`约束；
- 当前 `PVMMoEDecoder`、路由专家、共享专家、Top-k和负载均衡；
- Serial Encoder、Physics Encoder的核心计算；
- 物理感知跨模态融合的现有主干；
- 原始辐照度修正头、PCD上下文头和功率回归头；
- 任务损失、MoE均衡损失和raw-correction损失的含义。

允许修改的范围：配置系统、数据适配器、字段映射、缺失掩码、可选图像编码器的
外围适配、输入输出维度、训练/推理脚本、DDP运行、checkpoint元数据、指标、结果
导出、测试及文档。

现有图像实现参考了PVMMoE的“单帧CNN＋时间编码＋异步时间偏移”思路，但输出为
本融合层所需的 `[B,F,D]`。先审查现有实现；不要直接复制PVMMoE复杂的世界模型、
Frame-AC或Stage-I/Stage-II算法，也不要因此替换当前融合模型。

## 4. 通用Parquet接口

新增一个配置驱动的通用数据集类，例如：

```text
src/pv_physics_moe/data/parquet_dataset.py
```

不得为两个数据集复制两套窗口、归一化和输出逻辑。通用实现读取两个独立YAML的
字段映射，返回统一batch契约：

```text
serial                  [B, history_points, serial_input_dim]
serial_valid_mask       [B, history_points]
physics                 [B, H, 26]
physics_raw             [B, H, 26]
physics_valid_mask      [B, H]
future_zenith           [B, H]
target                  [B, H, 1]
target_valid_mask       [B, H]
irradiance_target       [B, H, 3]        # 字段存在时有效
irradiance_target_mask  [B, H, 3]        # 当前缺DNI/DHI时显式为0
images                  [B, F, 3, IH, IW] # 仅启用图像时
image_valid_mask        [B, F]
image_time_offsets      [B, F]
issue_time              长度B
target_times            [B, H]
current_power           [B, 1]
```

要求：

1. 时间戳按升序排列、去重并验证可解析；
2. `H = forecast_horizon_minutes / forecast_step_minutes`，目标窗口从一个数据集自己的
   `forecast_step_minutes`递增到240分钟；Luoyang是 `t+5,...,t+240`，YLJ步数不写死；
3. 所有目标必须来自真实目标时间戳，禁止线性插值、后向填充或使用未来数据伪造；
4. 数据实际颗粒度不满足配置时给出包含期望值和实际值的错误；
5. 只隔离未来目标，允许测试样本使用测试起点之前的历史输入；
6. 划分边界依据 `issue_time` 和最后目标时间，不能让未来目标跨出允许区间；
7. 所有归一化/缺失统计只用训练集拟合，并保存在checkpoint或metadata；
8. 功率可在模型内部归一化，但导出和最终指标必须恢复真实功率单位；
9. 连续气象字段采用配置指定的因果填充策略和mask，不能用测试统计；
10. 字段维度必须根据配置列表校验，不能写死9、13、15或22。

## 5. 两个独立配置文件

创建：

```text
configs/ylj_parquet.yaml
configs/luoyang_parquet.yaml
```

公共配置至少包含：

```yaml
dataset:
  name: ...
  parquet_file: ...
  timestamp_column: ...
  target_column: ...
  serial_columns: [...]
  irradiance_columns:
    ghi: ...
    dni: null
    dhi: null
  sampling_interval_minutes: null
  history_points: 16
  forecast_step_minutes: null
  forecast_steps: null  # 加载时由240/forecast_step_minutes校验或推导
  forecast_horizon_minutes: 240

site:
  latitude: null
  longitude: null
  altitude_m: null
  timezone: null

physics_defaults:
  pressure_pa: 101325.0
  pwv_cm: 1.5
  aod700: 0.08
  precip: 0.0

images:
  enabled: false
  root: null
  paths_column: null
  timestamps_column: null
  max_frames: 16
  channels: 3
  size: 64
  sampling_strategy: uniform
  reject_future: true

normalization:
  power_scale: null
  rated_power: null

evaluation:
  horizons_minutes: [15, 240]
  nrmse_denominator: null
  hard_delta_fraction: 0.10
  direction_min_delta_fraction: 0.02

runtime:
  distributed: true
  gpu_devices: "0,1,2,3,4,5,6,7"
  precision: fp16
  num_workers: 4
  pin_memory: true
```

配置加载器必须允许暂时未知的站点参数为 `null`。配置静态检查可以通过，但真正
构建需要太阳几何的样本前必须报出清楚的缺参信息；绝不能静默套用另一站点的经纬度。
如果用户后来补充站点参数或DNI/DHI，只改配置字段即可启用。

### 5.1 YLJ字段

YLJ没有图像，默认 `images.enabled: false`。已知字段：

```text
timestamp
observe_power
GHI_observe
TEMP_observe
WS_observe
WD_observe
PREC_observe
PWAT_observe
SDWE_observe
GHI_forecast_1day
TEMP_forecast_1day
WS_forecast_1day
WD_forecast_1day
PREC_forecast_1day
PWAT_forecast_1day
SDWE_forecast_1day
GHI_forecast_4hour
TEMP_forecast_4hour
WS_forecast_4hour
WD_forecast_4hour
PREC_forecast_4hour
PWAT_forecast_4hour
SDWE_forecast_4hour
```

YLJ配置先写入全部22个非时间字段作为可配置候选时序通道，实际
`serial_columns`顺序必须明确且自动决定 `serial_input_dim`。不得继续保留训练脚本
中“YLJ必须固定13维”的断言。

当前 `configs/server_station.yaml` 中已有YLJ实验信息，可作为迁移参考：2024年用于
训练/内部验证，2025年为测试；历史16点；原始站点配置为经度100.641、纬度
29.919、海拔4300m、时区Asia/Shanghai。不要删除旧配置。新YLJ Parquet配置中的
这些值应集中在 `site` 节，若无法确认则允许置空并给出TODO，不要写入代码。

YLJ Parquet路径和功率容量目前未知，配置中使用明确的空值/TODO。真实训练和最终
NRMSE/NMAE前必须要求用户补齐，不能假装默认容量是真实容量。

YLJ不固定为5分钟或48步。当前旧站点算法采用15分钟颗粒度，因此新配置可以先写：

```text
sampling_interval_minutes = 15
forecast_step_minutes = 15
forecast_horizon_minutes = 240
forecast_steps = 16
evaluation_horizons_minutes = [15, 240]
```

但适配器必须检查真实Parquet时间间隔；以后确认YLJ颗粒度不同，只改配置并由
`240 / forecast_step_minutes`重新得到步数。无论总步数多少，正式指标都按
`horizon_minutes`选择15分钟和240分钟，禁止用固定数组下标猜测。

### 5.2 Luoyang字段

已知路径：

```text
parquet_file: /data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1.parquet
image_root: /home/ma-user/work/john/IMAGE/Luoyang/Luoyang_EKO_ASI_resized_64
```

已知字段：

```text
timestamp
final_power
asi_path
asi_path_timestamps
GHI_mean_observe
msl_observe
t2m_observe
u10_observe
v10_observe
u100_observe
v100_observe
GHI_mean_forecast
msl_forecast
t2m_forecast
u10_forecast
v10_forecast
u100_forecast
v100_forecast
```

默认使用此前PVMMoE正式实验的9个时序通道：

```text
final_power
GHI_mean_observe
GHI_mean_forecast
msl_forecast
t2m_forecast
u10_forecast
v10_forecast
u100_forecast
v100_forecast
```

图片默认启用，字段为 `asi_path/asi_path_timestamps`。图片使用真实异步时间戳，
从 `[issue_time-75min, issue_time]` 收集，排序去重，禁止未来图片；最多16张，超过
时uniform抽取，不足时补零并由mask标识。图像路径必须相对于 `images.root`解析。

Luoyang容量参数统一为：

```text
power_scale = rated_power = nrmse_denominator = 48629.73
```

Luoyang保持5分钟颗粒度、未来4小时48步：

```text
sampling_interval_minutes = 5
forecast_step_minutes = 5
forecast_horizon_minutes = 240
forecast_steps = 48
evaluation_horizons_minutes = [15, 240]
```

时间划分：

```text
development: [2026-04-05 00:00:00, 2026-05-11 00:00:00)
test:        [2026-05-11 00:00:00, 2026-06-12 00:00:00)
validation: development内最后15%的有效issue_time
```

## 6. REST2物理参数和DNI/DHI渐进启用

通用适配器必须由timestamp、站点信息和原始气象构建模型现有的26维REST2输入，
而不是改动REST2核心。

- `mu0`、未来天顶角和 `dni_extra`：由目标时间及站点经纬度/时区计算；
- `input_ghi/previous_ghi`：来自因果历史GHI；
- Luoyang的 `msl` 是海平面气压，不能不加说明地直接当站点气压；有海拔时换算，
  否则使用配置默认并记录quality flag；
- YLJ的PWAT只有确认单位后才能转成 `pwv_cm`，换算比例写入配置；
- 当前缺少的PWV、AOD700、降水或气压可从 `physics_defaults` 读取；
- 每个物理量都返回来源/可用mask或写入metadata，区分observed、derived、default；
- DNI/DHI字段暂时为null时，`irradiance_target_mask`对应位置为0，辅助辐照度损失只
  在有效分量上计算；不得把缺失DNI/DHI当0参与白天损失；
- 以后配置写入GHI/DNI/DHI字段后，不需要修改模型代码即可自动启用完整辅助监督。

如果整个batch没有有效辐照度监督，irradiance loss必须是设备一致的有限零标量。

## 7. 图像开关与缺失模态

当前工作树已经加入 `SkyImageEncoder` 和 `model.use_image`。新Agent必须先验证并完善
而不是重复实现。

要求：

- YLJ配置关闭图片时，不创建和调用图像编码器，原serial+physics路径有效；
- Luoyang开启图片时，融合模态为serial+physics+image；
- 部分缺图、整条样本无图、整个batch无图均必须输出有限值；
- 全缺图不能稀释serial/physics融合权重或产生伪图像信息；
- `image_valid_mask`和真实分钟偏移必须传到编码器；
- 图片晚于issue_time必须拒绝；
- 不要为适配图片更换当前融合主干或MoE核心。

## 8. 数据集自适应步数输出和PVMMoE指标

推理必须输出长表CSV，每个有效 `issue_time` 恰好H行，H由当前数据集配置决定，
按issue_time、target_time排序：

```text
issue_time,target_time,horizon_minutes,y_true,y_pred
```

所有值必须是原始功率单位，不能输出归一化功率。`target_time`依次为：

```text
issue_time + forecast_step_minutes, ..., issue_time + 240min
```

Luoyang每个issue_time输出48行；YLJ若为15分钟颗粒度则输出16行。以后YLJ颗粒度
变化时行数随配置变化，但最后一个目标必须始终是240分钟。

如果hard-delta评估需要当前功率，可以在内部表或附加文件保存 `current_power`，但
上述五列和顺序必须保持稳定。分布式推理需要跨rank汇总、去重和排序，仅rank 0
原子写文件。

输出 `official_test_metrics.json`。至少分别统计15分钟和240分钟，按照导出记录的
`horizon_minutes`选择，不允许硬编码为第3步、第48步或其他固定下标：

- `overall`: count、MSE、RMSE、MAE、NRMSE、NMAE；
- `hard_delta`: count/fraction、MSE、RMSE、MAE、NRMSE、NMAE、direction_accuracy；
- 可额外输出全部H个时距逐步指标，但不得替代15和240分钟入口。

指标使用真实单位：

```text
MSE   = mean((y_pred-y_true)^2)
RMSE  = sqrt(MSE)
MAE   = mean(abs(y_pred-y_true))
NRMSE = RMSE / evaluation.nrmse_denominator
NMAE  = MAE  / evaluation.nrmse_denominator
hard  = abs(y_true-current_power) >= hard_delta_fraction * denominator
directional = abs(y_true-current_power) >= direction_min_delta_fraction * denominator
```

不得使用测试集 `y_max` 作为分母。每个数据集的容量、阈值、单位都来自各自配置。
配置缺少分母时允许数据检查和训练dry-run，但正式评测必须明确失败并提示补参。

同时保留PCD诊断：最大闭合误差、负GHI/DNI/DHI计数、夜间非零计数。DNI/DHI真实
标签存在后，按mask输出对应RMSE。

## 9. 8卡V100S训练

使用PyTorch DDP/`torchrun`，不能仅设置 `CUDA_VISIBLE_DEVICES` 后仍单进程训练，
也不要使用旧式 `nn.DataParallel` 作为正式8卡方案。

要求：

- `DistributedSampler`用于train/validation/test；
- 每epoch调用 `sampler.set_epoch(epoch)`；
- 仅rank 0写checkpoint、配置、日志、CSV和JSON；
- 验证指标在全rank正确聚合；
- 预测按issue_time去重，不能因sampler补齐产生重复；
- V100使用FP16 autocast和GradScaler，不启用BF16；
- checkpoint保存resolved config、字段顺序、归一化、容量、站点信息和数据schema；
- 支持单GPU和CPU smoke，不能让DDP改动模型数学结果。

提供类似命令并写入运行文档：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/train_parquet.py \
  --config configs/ylj_parquet.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/train_parquet.py \
  --config configs/luoyang_parquet.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/evaluate_parquet.py \
  --config configs/luoyang_parquet.yaml --checkpoint <checkpoint>
```

命令中的脚本名以最终实现为准，但YLJ/Luoyang只能通过 `--config` 切换，禁止分别
维护两套训练脚本。

## 10. 测试与验收

必须编写并运行轻量测试，不读取用户真实服务器数据、不执行完整训练：

1. 两个配置均可解析，字段顺序决定维度；
2. 合成15分钟YLJ Parquet产生 `[B,16,C]`、`[B,16,26]`、`[B,16,1]`；
3. 合成Luoyang Parquet产生图像、mask和异步偏移；
4. 图像关闭、部分缺失、全部缺失均前向有限；
5. 缺少DNI/DHI时mask正确且loss有限，补充字段后自动启用；
6. 严格拒绝未来图像、重复时间戳、路径/时间列表长度不一致；
7. Luoyang目标严格为48个五分钟点；YLJ目标数量由4小时和配置颗粒度推导，
   15分钟示例严格为16点；
8. 时间颗粒度不匹配时明确失败，不静默插值；
9. 训练统计不读取验证/测试数据；
10. Luoyang样本导出48行，15分钟YLJ样本导出16行，时间戳、排序和真实单位正确；
11. 手算验证15/240分钟overall、hard_delta、NRMSE/NMAE固定容量分母；
12. PCD闭合关系和夜间约束继续通过；
13. 单进程CPU smoke前向/反向通过；
14. DDP至少提供可自动执行的2进程CPU/gloo smoke测试或等价测试，不启动8卡训练；
15. 原有测试全部通过。

任何依赖真实路径的测试必须skip并给出原因，不能伪造“真实数据已验证”。

## 11. 最终交付

完成代码、测试和文档后：

1. 再次确认当前分支为 `why`；
2. 检查 `git status`、完整diff和新文件；
3. 显式暂存本任务文件；
4. 创建清晰提交，例如：

```text
Add configurable YLJ and Luoyang parquet forecasting pipeline
```

5. 推送且仅推送 `origin/why`；
6. 用read-only命令确认远程 `why` 的提交哈希；
7. 最终报告：修改摘要、两个配置仍需用户补充的参数、测试结果、训练/推理命令、
   结果文件位置、远程URL、分支和提交哈希。

如果网络或认证失败，不要反复盲目重试或改动其他远程；保留本地 `why` 提交并准确
报告失败命令和错误。不得声称未验证的上传已经成功。
