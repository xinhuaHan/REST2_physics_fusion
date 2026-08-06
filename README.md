# PVMMOE–REST2–PCD 五时距光伏功率预测

这是一个可独立运行的双模态光伏预测工程。模型保留 PVMMOE 的跨模态融合和混合专家主干，将 REST2 生成的物理向量作为第二模态，并使用 PCD 在输出端强制满足辐照度物理闭合。

## 1. 模型结构

```text
历史功率/辐照度/天气序列 ─ Serial Transformer ─┐
                                                ├─ 双向跨模态注意力
五时距 REST2 物理向量 ───── Physics Encoder ───┘
                                                │
                                       PVMMOE Top-k MoE
                                                │
                            ┌───────────────────┴───────────────────┐
                            │                                       │
                        功率预测头                         PCD 物理约束解码器
                            │                                       │
                   功率 [B,5,1]                    GHI/DNI/DHI [B,5,3]
```

PCD 保证：

```text
GHI = max(cos(zenith), 0) × DNI + DHI
GHI >= 0, DNI >= 0, DHI >= 0
夜间 GHI = DNI = DHI = 0
```

## 2. 当前实验设置

| 项目 | 设置 |
|---|---|
| 站点 | 100.641°E，29.919°N |
| 海拔 | 4300 m |
| 时区 | Asia/Shanghai |
| 标准气压 | 59,268.17 Pa，按海拔自动换算 |
| 历史窗口 | 16 × 15 min，即过去 4 h |
| 预测时距 | 15、30、60、240、1440 min |
| PWAT | cm，与 Folsom/REST2 约定一致 |
| 夜间负功率 | 截断为 0 |

年份严格隔离：

- 2024 年前 85%：训练集；
- 2024 年后 15%：内部验证集；
- 训练/验证之间隔离 1440 min，避免 1 d 标签跨越边界；
- 2025 年：锁定为甲方测试集；
- 归一化和损失尺度只使用 2024 年训练子集拟合；
- 2025 年不参与训练、归一化、调参或模型选择。

配置位于 `configs/server_station.yaml`：

```yaml
experiment:
  fixed_year_split: true
  training_years: [2024]
  holdout_years: [2025]
  train_fraction_within_training_years: 0.85
```

## 3. 仓库目录

```text
PVMMOE_REST2_PCD/
├─ .gitignore
├─ .gitattributes
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ configs/
│  └─ server_station.yaml
├─ src/
│  └─ pv_physics_moe/
│     ├─ config.py
│     ├─ data/
│     ├─ models/
│     ├─ physics/
│     └─ training/
├─ scripts/
│  ├─ check_station_tables.py
│  ├─ prepare_station_dataset.py
│  ├─ prepare_station_dataset_all_years.py
│  ├─ train_station.py
│  ├─ train_station_core.py
│  ├─ evaluate_station.py
│  ├─ run_smoke_tests.py
│  └─ run_station_smoke_test.py
├─ tests/
│  ├─ test_integrated_model.py
│  └─ test_station_server_pipeline.py
├─ data/raw/station/.gitkeep
├─ outputs/.gitkeep
├─ EXPERIMENT_2024_2025_ZH.md
├─ MODEL_ARCHITECTURE_REPORT_ZH.md
└─ RUN_GUIDE_ZH.md
```

## 4. 安装环境

要求 Python 3.9 或更高版本。

### Windows PowerShell

```powershell
git clone https://github.com/你的用户名/你的仓库名.git
Set-Location -LiteralPath ".\你的仓库名"

python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux 服务器

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available())"
```

若没有 CUDA，把 `configs/server_station.yaml` 改为：

```yaml
training:
  device: cpu
  num_workers: 0
```

## 5. 准备数据

真实数据、2025 年甲方评测数据和预处理数组不包含在 GitHub 仓库中。

默认需要将五个文件放入：

```text
data/raw/station/
├─ site_power_ghi.csv
├─ site_irradiance.csv
├─ forecast_4h.csv
├─ forecast_1d.csv
└─ site_weather.csv
```

支持 CSV 和 `.xlsx`。如果使用其他文件名，请修改 `configs/server_station.yaml`。

### 五表字段

| 文件 | 必需字段 |
|---|---|
| `site_power_ghi.csv` | `dtime, observe_power, observe_ghi` |
| `site_irradiance.csv` | `dtime, observe_ghi, observe_dni, observe_dhi` |
| `forecast_4h.csv` | `dtime, GHI, TEMP, WS, WD, PREC, PWAT, SDWE, interval` |
| `forecast_1d.csv` | 同 4 h 预报 |
| `site_weather.csv` | `dtime, GHI, TEMP, WS, WD, PREC, PWAT, SDWE` |

预报文件建议额外包含：

```text
report_date, report_time, update_time, govern_flag
```

关键单位：

- `GHI/DNI/DHI`：W/m²；
- `PWAT`：cm；
- `WD`：0～360°；
- `interval`：分钟；
- `dtime`：Asia/Shanghai 当地时间；
- `observe_power`：保持原始功率单位，模型输出使用相同单位。

也可以通过环境变量指定仓库外的数据路径：

Windows：

```powershell
$env:STATION_DATA_ROOT = "D:\pv_data\station"
$env:STATION_PROCESSED_DIR = "D:\pv_data\processed"
```

Linux：

```bash
export STATION_DATA_ROOT=/data/pv_station/raw
export STATION_PROCESSED_DIR=/data/pv_station/processed
```

## 6. 完整运行流程

### 6.1 检查五个文件

```bash
python scripts/check_station_tables.py --config configs/server_station.yaml
```

正常结果包含：

```json
{"ok": true}
```

### 6.2 预处理和年份隔离

```bash
python scripts/prepare_station_dataset.py --config configs/server_station.yaml
```

该命令自动完成：

1. 五张表按 15 min 时间轴对齐；
2. 生成五个预测时距的 REST2 物理向量；
3. 对每个时距执行预报发布时间防泄漏检查；
4. 将 2024 划分为训练/验证；
5. 将 2025 锁定为测试集；
6. 只用 2024 训练子集重新计算归一化参数。

已有处理结果时，只重新划分年份：

```bash
python scripts/prepare_station_dataset.py --config configs/server_station.yaml --resplit-only
```

处理后检查 `data/processed/station/metadata.json`：

```text
training_years = [2024]
holdout_years = [2025]
normalization_fit_years = [2024]
client_holdout_locked = true
train_validation_embargo_minutes = 1440
```

### 6.3 模型核心测试

```bash
python scripts/run_smoke_tests.py
```

预期：

```text
4 smoke tests passed
```

### 6.4 五表端到端测试

```bash
python scripts/run_station_smoke_test.py
```

该测试使用临时合成数据，不读取或上传真实站点数据。它验证跨年份划分、五时距张量、前向传播、反向传播、PCD 闭合和 checkpoint 保存。

### 6.5 真实数据训练冒烟测试

```bash
python scripts/train_station.py --config configs/server_station.yaml --smoke
```

训练入口会先检查：

```text
year-split audit passed: train/normalization=[2024], client holdout=[2025]
```

如果使用旧的、可能含 2025 训练数据的处理结果，训练会拒绝启动。

### 6.6 正式训练

```bash
python scripts/train_station.py --config configs/server_station.yaml
```

输出：

```text
outputs/station/checkpoint_last.pt
outputs/station/config_resolved.json
```

这些文件已被 `.gitignore` 排除，不会上传 GitHub。

### 6.7 甲方测试集评估

只有在允许查看 2025 年评测结果时才执行：

```bash
python scripts/evaluate_station.py \
  --config configs/server_station.yaml \
  --checkpoint outputs/station/checkpoint_last.pt \
  --split test
```

评估按 15/30/60/240/1440 min 分别输出功率 RMSE、MAE、GHI/DNI/DHI RMSE 和 PCD 最大闭合误差。

## 7. 张量接口

| 张量 | 单样本形状 | 批量形状 |
|---|---|---|
| 历史序列 | `[16,13]` | `[B,16,13]` |
| REST2 物理模态 | `[5,26]` | `[B,5,26]` |
| 功率标签/输出 | `[5,1]` | `[B,5,1]` |
| GHI/DNI/DHI | `[5,3]` | `[B,5,3]` |
| 五时距天顶角 | `[5]` | `[B,5]` |

## 8. 上传 GitHub

### 方法 A：Git 命令行

在本项目根目录执行：

```bash
git init
git add .
git status
git commit -m "Initial release: PVMMOE REST2 PCD five-horizon model"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

在执行 `git commit` 前，必须检查 `git status`，确认没有以下文件：

- 真实 CSV/XLSX；
- 2025 年甲方评测数据；
- `data/processed/`；
- `.venv/`；
- `outputs/` 中的权重；
- `.pt/.pth/.ckpt`；
- 密码、令牌或服务器绝对路径。

### 方法 B：GitHub 网页上传

1. 在 GitHub 创建空仓库；
2. 不要让 GitHub 自动添加另一个 README 或 `.gitignore`；
3. 打开 `Add file → Upload files`；
4. 上传本目录中的全部文件和文件夹；
5. 再次确认没有真实数据、2025 数据、权重和 `.venv`；
6. 填写提交说明并提交。

推荐使用 Git 命令行，因为 `.gitignore` 能自动阻止敏感和大文件进入提交。

## 9. 上传前检查

Windows PowerShell：

```powershell
git status --short
git check-ignore -v data\raw\station\你的真实数据.csv
git check-ignore -v outputs\station\checkpoint_last.pt
```

真实数据和权重应显示被 `.gitignore` 命中。

完整人工检查清单见 [`GITHUB_UPLOAD_CHECKLIST_ZH.md`](GITHUB_UPLOAD_CHECKLIST_ZH.md)。

## 10. 文档

- [`RUN_GUIDE_ZH.md`](RUN_GUIDE_ZH.md)：完整环境、数据和运行说明；
- [`EXPERIMENT_2024_2025_ZH.md`](EXPERIMENT_2024_2025_ZH.md)：年份隔离与审计依据；
- [`MODEL_ARCHITECTURE_REPORT_ZH.md`](MODEL_ARCHITECTURE_REPORT_ZH.md)：模型汇报材料。

## 11. 说明

仓库当前未自动附加开源许可证。若计划公开开源，请在确认代码和数据授权后再选择并添加合适的 `LICENSE`；如果仅用于私有协作，可直接创建 Private repository。

# YLJ / Luoyang Parquet pipeline

The configuration-driven Parquet training, DDP evaluation, image handling, and
official horizon rules are documented in [docs/PARQUET_PIPELINE_ZH.md](docs/PARQUET_PIPELINE_ZH.md).
