# PVMMOE + REST2 + PCD 五时距光伏预测模型：完整运行指南

本工程将三个模型拼接为一个可独立运行的双模态预测系统：

```text
历史功率/辐照度/天气序列 ── Serial Transformer ──┐
                                                  ├─ 双向跨模态融合
五个目标时距的 REST2 物理向量 ─ Physics Encoder ─┘
                                                  │
                                         PVMMOE Top-k MoE
                                                  │
                                  ┌───────────────┴───────────────┐
                                  │                               │
                              功率预测头                 PCD 物理约束解码器
                                  │                               │
                         功率 [B,5,1]            GHI/DNI/DHI [B,5,3]
```

当前任务一次预测五个时距：`15 min、30 min、60 min、4 h、1 d`。PCD 保证：

```text
GHI = max(cos(zenith), 0) × DNI + DHI
GHI >= 0, DNI >= 0, DHI >= 0
夜间 GHI = DNI = DHI = 0
```

---

## 1. 当前站点配置

正式配置位于 `configs/server_station.yaml`，已写入：

| 参数 | 当前值 |
|---|---:|
| 经度 | 100.641°E |
| 纬度 | 29.919°N |
| 海拔 | 4300 m |
| 时区 | Asia/Shanghai |
| 原始采样间隔 | 15 min |
| 历史窗口 | 16 步，即过去 4 h |
| 预测时距 | 15、30、60、240、1440 min |
| PWAT 单位 | cm，与 Folsom/REST2 一致 |
| 夜间负功率 | 截断为 0 |
| 气压 | 根据海拔自动计算 |

标准气压计算公式为：

```text
P(h) = 101325 × (1 - 2.25577×10⁻⁵ × h)^5.25588
P(4300 m) = 59268.17 Pa ≈ 59.27 kPa
```

配置中的 `default_pressure_pa: null` 表示自动换算。如果以后取得站点实测气压，可直接填写 Pa 值覆盖估算值。

---

## 2. 运行环境

最低要求：

- Python 3.9 或更高版本；
- 正式 GPU 训练建议使用支持 CUDA 的 PyTorch；
- CPU 可以完成预处理、测试和小规模训练；
- 原始数据推荐使用 CSV，也支持 `.xlsx`。

### 2.1 Windows PowerShell

打开 PowerShell，进入项目：

```powershell
Set-Location -LiteralPath "C:\Users\ADMIN\Desktop\模型整体框架\PVMMOE_REST2_PCD"
python --version
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

验证 Python 和 PyTorch：

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果输出 `cuda=False`，请先在配置中改为：

```yaml
training:
  device: cpu
  num_workers: 0
```

### 2.2 Linux 服务器

```bash
cd /path/to/PVMMOE_REST2_PCD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果服务器需要特定 CUDA 版本，先按照该服务器 CUDA 环境安装匹配的 PyTorch，再执行 `pip install -r requirements.txt`。

---

## 3. 准备五个原始数据文件

### 3.1 默认目录结构

最省事的方法是把五个文件放到：

```text
PVMMOE_REST2_PCD/
└─ data/
   └─ raw/
      └─ station/
         ├─ site_power_ghi.csv
         ├─ site_irradiance.csv
         ├─ forecast_4h.csv
         ├─ forecast_1d.csv
         └─ site_weather.csv
```

也可以使用 `.xlsx`，但需要同步修改 `configs/server_station.yaml` 中对应文件名。

> 注意：目前 `C:\Users\ADMIN\Desktop\dataset` 中检测到的是 `Folsom_irradiance.csv`、`Folsom_weather.csv` 和 2014—2016 图像目录，不是上述站点五表。不能直接把该目录作为五表输入，除非其中补齐五个站点文件并在 YAML 中配置正确文件名。

### 3.2 五个文件的字段要求

#### 文件 1：功率和 GHI

默认文件名：`site_power_ghi.csv`

```text
dtime, observe_power, observe_ghi
```

#### 文件 2：三分量辐照度

默认文件名：`site_irradiance.csv`

```text
dtime, observe_ghi, observe_dni, observe_dhi
```

#### 文件 3：4 小时天气预报

默认文件名：`forecast_4h.csv`

```text
dtime, GHI, TEMP, WS, WD, PREC, PWAT, SDWE, interval
```

建议同时包含：

```text
report_date, report_time, update_time, govern_flag
```

#### 文件 4：1 天天气预报

默认文件名：`forecast_1d.csv`，字段与 4 小时预报相同。

#### 文件 5：站点实测天气

默认文件名：`site_weather.csv`

```text
dtime, GHI, TEMP, WS, WD, PREC, PWAT, SDWE
```

### 3.3 字段含义和单位

| 字段 | 含义 | 约定 |
|---|---|---|
| `dtime` | 实测时间或预报有效时间 | 当地时间，时区为 Asia/Shanghai |
| `observe_power` | 光伏功率 | 保持原数据单位，模型输出使用相同单位 |
| `GHI/DNI/DHI` | 辐照度 | W/m² |
| `TEMP` | 温度 | 训练和推理数据必须保持同一单位 |
| `WS` | 风速 | 训练和推理保持一致 |
| `WD` | 风向 | 角度，0～360° |
| `PREC` | 降水 | 训练和推理保持一致 |
| `PWAT` | 可降水量 | **cm** |
| `SDWE` | 积雪水当量 | 训练和推理保持一致 |
| `interval` | 预报提前量 | **分钟** |
| `report_date/time` | 预报发布时间 | `report_time` 支持 `HH_MM` 或 `HH:MM` |

所有实测时间轴应满足：

- 15 min 间隔；
- 时间升序；
- 没有重复时间戳；
- 五张表的有效时间范围尽量重叠；
- 至少覆盖“4 h 历史 + 1 d 未来”，正式训练建议使用数月或更长数据。

### 3.4 不复制数据，直接指定外部目录

Windows PowerShell：

```powershell
$env:STATION_DATA_ROOT = "D:\pv_data\station"
$env:STATION_PROCESSED_DIR = "D:\pv_data\processed"
```

Linux：

```bash
export STATION_DATA_ROOT=/data/pv_station/raw
export STATION_PROCESSED_DIR=/data/pv_station/processed
```

也可以在命令中使用 `--data-root` 和 `--output` 临时覆盖路径。

---

## 4. 检查配置

打开 `configs/server_station.yaml`，至少确认以下项目：

```yaml
data:
  longitude: 100.641
  latitude: 29.919
  altitude_m: 4300.0
  timezone: Asia/Shanghai
  sample_minutes: 15
  history_length: 16
  forecast_lead_minutes: [15, 30, 60, 240, 1440]
  power_target_floor: 0.0
  default_pressure_pa: null

model:
  forecast_horizon: 5

training:
  device: cuda
  output_dir: outputs/station
```

必须满足：

```text
model.forecast_horizon = forecast_lead_minutes 的数量 = 5
```

如果原文件名或列名不同，只修改 YAML，不需要改 Python 代码。

---

## 5. 第一步：执行五表结构检查

在项目根目录运行：

```powershell
python scripts\check_station_tables.py --config configs\server_station.yaml
```

Linux 写法：

```bash
python scripts/check_station_tables.py --config configs/server_station.yaml
```

正常结果中应出现：

```json
{"ok": true}
```

如果使用临时数据目录：

```powershell
python scripts\check_station_tables.py --config configs\server_station.yaml --data-root "D:\pv_data\station"
```

此步骤只读取表头，不修改原始数据。

---

## 6. 第二步：生成训练数据

```powershell
python scripts\prepare_station_dataset.py --config configs\server_station.yaml
```

或者显式指定输入输出：

```powershell
python scripts\prepare_station_dataset.py `
  --config configs\server_station.yaml `
  --data-root "D:\pv_data\station" `
  --output "D:\pv_data\processed"
```

预处理完成后会打印 `metadata.json` 内容，其中需要重点确认：

```text
forecast_lead_minutes = [15, 30, 60, 240, 1440]
forecast_horizon = 5
station.pressure_pa ≈ 59268.17
split_samples.train/val/test 均大于 0
negative_power_targets_clipped 为实际截断数量
```

默认输出目录为：

```text
data/processed/station/
├─ metadata.json
├─ normalization.json
├─ timeline/
│  ├─ serial.npy
│  ├─ physics.npy
│  ├─ physics_raw.npy
│  ├─ target_power.npy
│  ├─ target_irradiance.npy
│  ├─ future_zenith.npy
│  └─ timestamps.npy
└─ splits/
   ├─ train_end_indices.npy
   ├─ val_end_indices.npy
   └─ test_end_indices.npy
```

预处理后的单样本张量形状：

| 张量 | 形状 |
|---|---|
| 历史序列 `serial` | `[16,13]` |
| REST2 物理模态 `physics` | `[5,26]` |
| 未归一化物理量 `physics_raw` | `[5,26]` |
| 五时距天顶角 `future_zenith` | `[5]` |
| 功率标签 `target` | `[5,1]` |
| 辐照度标签 `irradiance_target` | `[5,3]` |

### 防止预报信息泄漏

对于任意预测时距 `h`，预报记录必须满足：

```text
forecast_issue_time <= target_valid_time - h = 预测起点
```

- 15/30/60/240 min 以 4 h 预报为主；
- 1440 min 以 1 d 预报为主；
- 不使用预测起点之后发布的天气预报；
- 不使用目标时刻的未来实测天气作为输入；
- 数据按时间顺序划分，归一化只使用训练段。

---

## 7. 第三步：运行测试

### 7.1 模型核心测试

```powershell
python scripts\run_smoke_tests.py
```

预期最后显示：

```text
4 smoke tests passed
```

### 7.2 五表、五时距端到端测试

```powershell
python scripts\run_station_smoke_test.py
```

该测试自动生成同结构临时数据，并验证：

- 五表预处理；
- `[B,16,13]` 与 `[B,5,26]` 对齐；
- 双模态融合；
- MoE 前向和反向传播；
- 五时距功率与辐照度输出；
- PCD 物理闭合；
- checkpoint 保存。

预期最后显示：

```text
PASS five-table/five-horizon preprocessing, two-modal fusion, backward training and PCD closure
```

---

## 8. 第四步：用真实预处理数据进行训练冒烟测试

先执行完第 6 节的真实数据预处理，再运行：

```powershell
python scripts\train_station.py --config configs\server_station.yaml --smoke
```

该命令：

- 只读取前 4 个训练样本；
- 自动缩小模型；
- 使用 CPU 训练 1 个 epoch；
- 验证真实处理数据能否完成前向、反向和保存。

输出权重：

```text
outputs/station/checkpoint_smoke.pt
```

如果真实数据冒烟测试失败，不要直接开始正式训练，应先处理错误信息。

---

## 9. 第五步：正式训练

确认 `training.device` 与服务器环境一致后运行：

```powershell
python scripts\train_station.py --config configs\server_station.yaml
```

默认正式训练参数：

```yaml
batch_size: 16
epochs: 50
learning_rate: 0.0003
weight_decay: 0.00001
num_workers: 4
device: cuda
```

训练日志示例：

```text
epoch=1 loss=... power=... irradiance=...
```

最终输出：

```text
outputs/station/checkpoint_last.pt
outputs/station/config_resolved.json
```

其中 checkpoint 同时保存：

- 模型权重；
- 完整解析后配置；
- 训练数据归一化参数。

### 显存不足时

优先把：

```yaml
training:
  batch_size: 16
```

改成 `8`、`4` 或 `2`。仍然不足时再减小 `model.hidden_size`、`moe_layers` 或 `num_experts`。

---

## 10. 第六步：评估测试集

```powershell
python scripts\evaluate_station.py `
  --config configs\server_station.yaml `
  --checkpoint outputs\station\checkpoint_last.pt `
  --split test
```

Linux：

```bash
python scripts/evaluate_station.py \
  --config configs/server_station.yaml \
  --checkpoint outputs/station/checkpoint_last.pt \
  --split test
```

也可以把 `--split` 改成 `train` 或 `val`。

评估结果包括：

- 五个预测时距各自的功率 RMSE；
- 五个预测时距各自的功率 MAE；
- 五个时距各自的 GHI/DNI/DHI RMSE；
- 五时距宏平均功率指标；
- PCD 最大物理闭合误差。

示意结构：

```json
{
  "lead_minutes": [15, 30, 60, 240, 1440],
  "per_horizon": {
    "15": {"power_rmse": 0.0, "power_mae": 0.0, "irradiance_rmse": [0.0, 0.0, 0.0]},
    "1440": {"power_rmse": 0.0, "power_mae": 0.0, "irradiance_rmse": [0.0, 0.0, 0.0]}
  },
  "pcd_max_closure_error_wm2": 0.0
}
```

数字仅表示输出格式，不代表真实精度。

---

## 11. 一套可复制的 Windows 完整命令

确保五个站点文件已经放入 `data\raw\station` 后：

```powershell
Set-Location -LiteralPath "C:\Users\ADMIN\Desktop\模型整体框架\PVMMOE_REST2_PCD"

python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts\check_station_tables.py --config configs\server_station.yaml
python scripts\prepare_station_dataset.py --config configs\server_station.yaml
python scripts\run_smoke_tests.py
python scripts\run_station_smoke_test.py
python scripts\train_station.py --config configs\server_station.yaml --smoke
python scripts\train_station.py --config configs\server_station.yaml
python scripts\evaluate_station.py --config configs\server_station.yaml --checkpoint outputs\station\checkpoint_last.pt --split test
```

---

## 12. 常见问题

### 12.1 `station input does not exist`

原因：数据目录或 YAML 文件名不正确。

处理：检查 `data.root_dir`、环境变量 `STATION_DATA_ROOT` 和五个 `*_file` 配置。

### 12.2 `missing columns`

原因：实际列名与 YAML 不一致。

处理：优先修改 `configs/server_station.yaml` 中列名映射，不要直接改模型代码。

### 12.3 `duplicate timestamps`

原因：实测表中同一 `dtime` 出现多行。

处理：根据业务规则先去重或聚合；不要让程序任意选择重复记录。

### 12.4 `only N valid samples`

常见原因：

- 数据不足以形成 4 h 历史和 1 d 未来标签；
- 时间轴不是连续 15 min；
- 五张表时间范围不重叠；
- 天气预报发布时间晚于预测起点，被防泄漏规则过滤；
- 关键列包含大量空值。

### 12.5 CUDA 不可用

先运行：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

临时使用 CPU 时，把 YAML 中 `device` 改为 `cpu`，`num_workers` 改为 `0`。

### 12.6 Windows DataLoader 报错或进程卡住

把：

```yaml
training:
  num_workers: 0
```

正式 Linux 服务器再根据 CPU 核数调大。

### 12.7 训练损失出现 NaN

检查：

- `metadata.json` 中三个数据集样本数是否正常；
- PWAT 是否确实为 cm；
- 辐照度是否为 W/m²；
- 气象列是否存在异常极值；
- 学习率是否过大。

### 12.8 PCD 闭合误差

模型结构保证：

```text
GHI = cos(zenith) × DNI + DHI
```

浮点计算可能产生极小数值误差，但不应出现明显偏差。可通过 `evaluate_station.py` 的 `pcd_max_closure_error_wm2` 检查。

---

## 13. 推荐执行顺序

```text
确认五表和单位
      ↓
check_station_tables.py
      ↓
prepare_station_dataset.py
      ↓
run_smoke_tests.py
      ↓
run_station_smoke_test.py
      ↓
train_station.py --smoke（真实数据）
      ↓
train_station.py（正式训练）
      ↓
evaluate_station.py（分时距评估）
```

模型架构和汇报说明见 `MODEL_ARCHITECTURE_REPORT_ZH.md`，服务器数据对齐细节见 `SERVER_RUN_ZH.md`。

---

## 14. YLJ Parquet 正式训练命令（Linux 8 卡）

本节使用 `configs/ylj_parquet.yaml`，对应真实数据文件：

```text
/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet
```

该配置固定预测未来 4 小时：15 分钟颗粒度时输出 16 步（`t+15` 至 `t+240`），正式指标只统计 `horizon_minutes=15` 和 `240`，NRMSE/NMAE 分母为 468 MW。执行正式训练前依次运行以下命令。

```bash
# 0. 进入仓库并取得包含 YLJ 配置的 why 分支
cd /path/to/REST2_physics_fusion_Aligned_Model
git fetch origin
git switch why
git pull --ff-only origin why

# 1. 只读检查真实 Parquet：schema、15 分钟颗粒度、功率、DNI/DHI 和 PWAT 单位
python scripts/inspect_ylj_parquet.py \
  --parquet /data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format-with_DNI_DHI.parquet \
  --pwat-unit mm \
  --output-json outputs/ylj_parquet_inspection.json

# 2. 单卡真实数据 smoke：仅 1 epoch、最多 4 个样本；成功后再启动正式训练
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_parquet.py \
  --config configs/ylj_parquet.yaml \
  --smoke \
  --output-dir outputs/ylj_parquet_smoke

# 3. 正式 8 卡 V100S DDP 训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/train_parquet.py \
  --config configs/ylj_parquet.yaml \
  --output-dir outputs/ylj_parquet

# 4. 正式测试集分布式评测和结果导出
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/evaluate_parquet.py \
  --config configs/ylj_parquet.yaml \
  --checkpoint outputs/ylj_parquet/checkpoint_last.pt \
  --output-dir outputs/ylj_official_test \
  --split test
```

第 1 步必须以退出码 0 完成；该脚本不会修改 Parquet。第 2 步成功后会生成 `outputs/ylj_parquet_smoke/checkpoint_smoke.pt`。正式训练完成后生成 `outputs/ylj_parquet/checkpoint_last.pt`；评测目录包含 `point_predictions.csv` 和 `official_test_metrics.json`。负功率会按配置截断到 0 MW，缺失功率不会被插值或后向填充。
