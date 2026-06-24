# REST2 / Clear-Sky Physics Fusion 中文使用说明

这个项目用于把原本依赖图像、时间序列、气象三模态的 FARMS 思路，改造成只依赖时间序列和气象数据也能运行的版本。

当前项目的核心目标是：

1. 用 `pvlib / REST2` 生成晴空辐照度物理先验。
2. 用天气预报或气象变量修正晴空先验，形成 `weather_prior`。
3. 用时序模型学习真实观测中的局地修正。
4. 自动比较 `baseline / REST2 / weather_prior / REST2+weather_prior`，并生成推荐使用哪一个模型的 `model_selection.csv`。

当前结论是：`REST2` 保留为可选物理模块和诊断模块；真实数据到来后，优先验证 `weather_prior` 是否稳定优于 `baseline`，再判断是否需要继续使用 `REST2+weather_prior`。

## 项目结构

```text
rest2_physics_fusion/
  configs/
    base.yaml                         # 主配置文件：站点、特征、训练参数、设备
  data/
    mock_raw/                         # 模拟原始数据
    mock_enriched/                    # 带溯源字段的调试数据
    mock_model_ready/                 # 固定 schema 的训练入口 CSV
    real_model_ready/                 # 建议真实数据放这里
    folsom_model_ready/               # Folsom CSV 转换后的训练入口
  scripts/
    generate_mock_training_csv.py      # 生成模拟数据
    excel_to_model_ready.py            # 真实 Excel 一键转 model_ready
    folsom_to_model_ready.py           # Folsom irradiance/weather CSV 一键转 model_ready
    diagnose_physics_consistency.py    # 检查数据是否适合验证 REST2 / weather_prior
    run_dataset_pipeline.py            # 一键完整实验入口，推荐使用
    run_ablation.py                    # 单独跑 ablation
    analyze_ablation.py                # 分析 ablation 并生成 model_selection
    train.py                           # 单模型训练
    evaluate.py                        # 评估 checkpoint
    export_predictions.py              # 导出预测诊断 CSV
    export_selected_predictions.py      # 批量导出 model_selection 选中模型的预测 CSV
    evaluate_pooled_predictions.py      # 跨站点 pooled / macro 综合评估
    export_physics_vectors.py          # 导出物理向量，方便接入 PVMMOE 等外部融合模型
  src/
    rest2_physics_fusion/
      data/
      physics/
      models/
      training/
  tests/
```

## 当前模型

当前主模型结构是 `clear_sky_weather_fusion`：

```text
时间序列编码器
+ 气象序列编码器
+ pvlib / REST2 晴空物理特征
+ clear-sky index 预测分支
+ residual 修正分支
+ 可选 weather_prior 融合分支
```

`weather_prior` 的形式是：

```text
prior_pred = clear_sky_horizon * k_weather

final_prediction =
  prior_weight * prior_pred
  + (1 - prior_weight) * serial_pred
```

其中：

```text
clear_sky_horizon         目标预测时刻的晴空辐照度
k_weather                 根据天气和时序特征学习到的天气衰减系数
serial_pred               原结构化时序预测分支
prior_weight              模型学习到的物理先验权重
```

## 数据要求

训练入口必须是固定 schema 的 `model_ready` CSV。也就是说，真实数据最终需要被整理成和 `data/mock_model_ready/*.csv` 一样的列结构。

最关键的基础列包括：

```text
dtime
source_type
station_name
input_ghi
target_ghi_5min
target_ghi_4h
target_ghi_1d
```

主要气象/时序列包括：

```text
temp_c
wind_speed
wind_dir
precip
pwv_cm
temp_c_is_observed
wind_speed_is_observed
wind_dir_is_observed
precip_is_observed
pwv_cm_is_observed
pressure_pa_is_observed
aod700_is_observed
weather_is_joined
```

主要物理列包括：

```text
ghi_clear_target
dni_clear_target
dhi_clear_target
mu0_target
apparent_zenith
dni_extra
pressure_pa
pwv_cm
aod700
t_clr_dd
clear_sky_index_last
smart_persistence_ghi
clear_sky_index_current
weather_attenuation_prior
weather_adjusted_clear_sky_ghi
clear_sky_residual_last
clear_sky_gap
ghi_clear_horizon_5min
ghi_clear_horizon_4h
ghi_clear_horizon_1d
mu0_horizon_5min
mu0_horizon_4h
mu0_horizon_1d
weather_adjusted_clear_sky_ghi_horizon_5min
weather_adjusted_clear_sky_ghi_horizon_4h
weather_adjusted_clear_sky_ghi_horizon_1d
```

如果真实原始数据暂时没有这些物理列，需要先用现有的物理特征生成流程补齐，再进入训练。当前 `data/mock_model_ready/` 就是训练入口格式的参考样例。

## 从原始真实 Excel 转成训练数据

如果真实数据格式和下面目录中的样例完全相同：

```text
C:\Users\ADMIN\Desktop\新建文件夹
```

也就是包含这些文件：

```text
solar_history.xlsx
解放电站数据.xlsx
庆达电站数据.xlsx
天气预报（4小时）.xlsx
天气预报（一天）.xlsx
新数据描述和使用.markdown
```

当前项目中已经有一键转换脚本：

```text
scripts/excel_to_model_ready.py
```

它会读取这些 Excel，完成：

```text
读取原始 Excel
站点数据与 solar_history 气象数据按时间戳合并
用 pvlib / REST2 计算太阳几何和晴空辐照度
根据 PWAT、降水、风速、上一时刻 clear-sky index 生成 weather_prior
用未来时刻的真实/已有 GHI 生成 target_ghi_5min、target_ghi_4h、target_ghi_1d
输出固定 schema 的 model_ready CSV
```

运行命令：

```powershell
cd C:\Users\ADMIN\Desktop\rest2_physics_fusion

C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\excel_to_model_ready.py --input-dir "C:\Users\ADMIN\Desktop\新建文件夹" --model-ready-output-dir data\real_model_ready --enriched-output-dir data\real_enriched --physics-output-dir outputs\real_physics_csv --latitude 29.919 --longitude 100.641 --altitude-m 0.0 --timezone Asia/Shanghai --clear-sky-backend auto
```

输出文件：

```text
data/real_model_ready/train_solar_history.csv
data/real_model_ready/train_jiefang_station.csv
data/real_model_ready/train_qingda_station.csv
data/real_model_ready/train_forecast_4h.csv
data/real_model_ready/train_forecast_1d.csv
```

同时会输出便于排查字段来源的文件：

```text
data/real_enriched/train_solar_history.csv
data/real_enriched/train_jiefang_station.csv
data/real_enriched/train_qingda_station.csv
data/real_enriched/train_forecast_4h.csv
data/real_enriched/train_forecast_1d.csv
```

以及物理特征检查文件：

```text
outputs/real_physics_csv/physics_solar_history.csv
outputs/real_physics_csv/physics_jiefang_station.csv
outputs/real_physics_csv/physics_qingda_station.csv
outputs/real_physics_csv/physics_forecast_4h.csv
outputs/real_physics_csv/physics_forecast_1d.csv
```

这些 `data/real_model_ready/*.csv` 可以直接进入后续训练和 pipeline。

如果你只想导出物理特征 CSV，也可以使用旧的物理导出脚本：

```text
scripts/build_physics_csv.py
```

它的作用是读取这些 Excel，计算太阳几何、晴空辐照度、REST2/pvlib 物理特征，并导出物理特征 CSV。

运行命令：

```powershell
cd C:\Users\ADMIN\Desktop\rest2_physics_fusion

C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\build_physics_csv.py --input-dir "C:\Users\ADMIN\Desktop\新建文件夹" --output-dir outputs\physics_csv --latitude 29.919 --longitude 100.641 --altitude-m 0.0 --timezone Asia/Shanghai --clear-sky-backend auto
```

输出文件：

```text
outputs/physics_csv/physics_solar_history.csv
outputs/physics_csv/physics_jiefang_station.csv
outputs/physics_csv/physics_qingda_station.csv
outputs/physics_csv/physics_forecast_4h.csv
outputs/physics_csv/physics_forecast_1d.csv
```

注意：`build_physics_csv.py` 的输出是“物理特征 CSV”，不是最终训练入口 `model_ready` CSV。真正的一键训练数据转换请使用 `excel_to_model_ready.py`。

要进入训练，还需要固定 schema 的 `model_ready` CSV。`model_ready` 相比 `physics_csv` 还必须包含：

```text
target_ghi_5min
target_ghi_4h
target_ghi_1d
固定列顺序
训练所需 mask/source 字段
horizon-aligned clear-sky/weather-prior 字段
```

因此完整流程是：

```text
原始 Excel
  ↓
excel_to_model_ready.py
  ↓
real_model_ready CSV
  ↓
run_dataset_pipeline.py 训练和对比模型
```

目前 README 中后续所有训练命令都默认输入已经是 `model_ready` CSV。如果你手里的真实数据还停留在 `C:\Users\ADMIN\Desktop\新建文件夹` 这种原始 Excel 阶段，应先运行 `excel_to_model_ready.py`。

当前代码中 `data/mock_model_ready/*.csv` 是最终训练 schema 的参考。转换真实数据时，建议先打开任意一个 mock 文件，对齐列名和列顺序：

```text
data/mock_model_ready/train_jiefang_station.csv
data/mock_model_ready/train_qingda_station.csv
data/mock_model_ready/train_solar_history.csv
data/mock_model_ready/train_forecast_4h.csv
data/mock_model_ready/train_forecast_1d.csv
```

转换完成后，建议真实训练 CSV 放到：

```text
data/real_model_ready/
```

例如：

```text
data/real_model_ready/train_jiefang_station.csv
data/real_model_ready/train_qingda_station.csv
```

然后再运行：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\real_model_ready --csv-files train_jiefang_station.csv --output-root outputs\pipeline_real_jiefang --target-columns target_ghi_5min target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline rest2_calibrated weather_prior weather_prior_rest2 --train-selected
```

一句话总结：`excel_to_model_ready.py` 是真实 Excel 到训练 CSV 的一键入口；`build_physics_csv.py` 只用于单独检查物理特征。

如果用于真实电站训练，建议只导出站点级训练表：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\excel_to_model_ready.py --input-dir "C:\Users\ADMIN\Desktop\新建文件夹" --model-ready-output-dir data\real_station_model_ready --enriched-output-dir data\real_station_enriched --physics-output-dir outputs\real_station_physics_csv --latitude 29.919 --longitude 100.641 --altitude-m 0.0 --timezone Asia/Shanghai --clear-sky-backend auto --station-only
```

这时天气历史文件仍会被读取并 join 到站点数据里，但不会把 `solar_history / forecast_4h / forecast_1d` 当成独立训练任务。

## 从 Folsom CSV 转成训练数据

Folsom 数据集可以迁移到当前框架，而且非常适合用来验证“无图像、只用时序和气象数据”的路线。当前转换脚本只读取：

```text
C:\Users\ADMIN\Desktop\dataset\Folsom_irradiance.csv
C:\Users\ADMIN\Desktop\dataset\Folsom_weather.csv
```

不会读取 `2014/2015/2016` 图像文件夹。也就是说，这条路线完全不依赖图像模态。

Folsom 原始列包括：

```text
Folsom_irradiance.csv:
timeStamp, ghi, dni, dhi

Folsom_weather.csv:
timeStamp, air_temp, relhum, press, windsp, winddir, max_windsp, precipitation
```

转换脚本会完成：

```text
读取 irradiance/weather CSV
按 timeStamp 精确合并
训练用 dtime 保持为 UTC 基准时间，避免夏令时回拨造成重复本地时间
enriched 调试文件额外保留 America/Los_Angeles 本地时间字符串
用实测 ghi 作为 input_ghi
用未来实测 ghi 生成 target_ghi_5min、target_ghi_4h、target_ghi_1d
用实测气温、湿度、气压、风速、降水补充气象特征
用温度和相对湿度确定性估计 pwv_cm
用 pvlib / REST2 计算太阳几何和晴空辐照度
输出固定 schema 的 model_ready CSV
```

注意：这里补充的物理量不是随机生成的。它们来自 Folsom 的实测辐照度、实测气象字段，以及 `pvlib / REST2` 的确定性物理计算。`pwv_cm` 因为原始 Folsom 没有直接提供，所以由温度和相对湿度用经验物理公式估计，并会保留来源标记。

Folsom 的原始 `timeStamp` 按 UTC 处理。不要先手动转换成本地 naive 时间，否则会在夏令时回拨日出现两个 `01:00:00`，触发 `AmbiguousTimeError`。

建议先跑小样本 smoke test：

```powershell
cd C:\Users\ADMIN\Desktop\rest2_physics_fusion

C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\folsom_to_model_ready.py --dataset-dir C:\Users\ADMIN\Desktop\dataset --model-ready-output data\folsom_model_ready_smoke\train_folsom.csv --enriched-output data\folsom_enriched_smoke\train_folsom.csv --max-rows 20000
```

如果 smoke test 通过，再转换完整 Folsom：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\folsom_to_model_ready.py --dataset-dir C:\Users\ADMIN\Desktop\dataset --model-ready-output data\folsom_model_ready\train_folsom.csv --enriched-output data\folsom_enriched\train_folsom.csv
```

转换后先检查物理一致性：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\diagnose_physics_consistency.py --csv-dir data\folsom_model_ready --csv-files train_folsom.csv --output-dir outputs\folsom_physics_consistency
```

再运行完整 ablation：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\folsom_model_ready --csv-files train_folsom.csv --output-root outputs\pipeline_folsom --target-columns target_ghi_5min target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline rest2_calibrated weather_prior weather_prior_rest2 --train-selected
```

如果只想快速确认训练能跑通，可以先用 smoke 数据和少量 seed：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\folsom_model_ready_smoke --csv-files train_folsom.csv --output-root outputs\pipeline_folsom_smoke --target-columns target_ghi_5min --seeds 42 --variants baseline weather_prior --epochs 1
```

当前本机 smoke test 已经验证：Folsom 小样本可以成功生成固定 50 列 `model_ready` CSV，并能跑通 `baseline` 和 `weather_prior` 训练流程。因此它是后续替代服务器真实数据、先验证模型方向的一个好选择。

## 配置文件怎么改

主要修改 `configs/base.yaml`。

### 修改站点信息

如果真实电站经纬度不同，修改：

```yaml
site:
  latitude: 29.919
  longitude: 100.641
  altitude_m: 0.0
  timezone: Asia/Shanghai
```

### 修改默认预测目标

默认目标在：

```yaml
data:
  target_column: target_ghi_5min
```

可选值：

```text
target_ghi_5min
target_ghi_4h
target_ghi_1d
```

实际运行时也可以不改配置，直接通过命令行传：

```powershell
--target-columns target_ghi_5min target_ghi_4h target_ghi_1d
```

或者单模型训练时传：

```powershell
--target-column target_ghi_4h
```

### 修改训练设备

CPU 训练：

```yaml
training:
  device: cpu
```

GPU 训练：

```yaml
training:
  device: cuda
```

如果服务器有多张 GPU，通常先用环境变量指定卡，例如：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
```

然后配置里保持：

```yaml
training:
  device: cuda
```

### 修改训练轮数、batch size、学习率

```yaml
training:
  batch_size: 32
  epochs: 3
  lr: 0.001
  device: cpu
```

真实数据上建议先小跑：

```yaml
epochs: 3
```

确认流程没问题后再改大，例如：

```yaml
epochs: 20
```

### 修改模型开关

默认不要在配置里全局开启 REST2 或 weather_prior：

```yaml
model:
  use_rest2_calibration: false
  use_weather_prior_fusion: false
```

推荐让 `run_dataset_pipeline.py` 自动比较并生成 `model_selection.csv`，再决定每个数据集和目标使用哪个模型。

如果要手动强制开启某个模型，使用命令行参数：

```powershell
--model-type weather_prior
```

可选模型：

```text
baseline
physics_feature_only
clear_sky_power_prior
weather_prior_weak
rest2_calibrated
weather_prior
weather_prior_rest2
```

## 完整运行教程

下面假设项目路径是：

```powershell
C:\Users\ADMIN\Desktop\rest2_physics_fusion
```

先进入项目：

```powershell
cd C:\Users\ADMIN\Desktop\rest2_physics_fusion
```

Python 路径使用：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe
```

为了命令短一些，下面仍然直接写完整 Python 路径。

## 第 1 步：生成或准备数据

如果只是测试代码流程，可以生成 mock 数据：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\generate_mock_training_csv.py --raw-output-dir data\mock_raw --enriched-output-dir data\mock_enriched --model-ready-output-dir data\mock_model_ready
```

真实数据建议放到：

```text
data/real_model_ready/
```

例如：

```text
data/real_model_ready/your_real_data.csv
```

注意：真实 CSV 必须已经是 `model_ready` 固定 schema。如果还只是原始 Excel 或原始 CSV，需要先做物理特征生成和字段对齐。

## 第 2 步：检查数据物理一致性

先检查数据是否适合验证 REST2 / weather_prior：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\diagnose_physics_consistency.py --csv-dir data\real_model_ready --csv-files your_real_data.csv --output-dir outputs\physics_consistency_real
```

如果使用 mock 数据：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\diagnose_physics_consistency.py --csv-dir data\mock_model_ready --csv-files train_jiefang_station.csv --output-dir outputs\physics_consistency_mock
```

输出文件：

```text
outputs/physics_consistency_real/physics_consistency_summary.csv
outputs/physics_consistency_real/physics_consistency_segments.csv
```

重点看这些列：

```text
target_weather_prior_corr
weather_prior_mae_gain_vs_current_ghi
clear_sky_upper_exceed_rate
night_nonzero_target_rate
rest2_validation_validity
physics_consistency_warning
```

如果出现：

```text
rest2_validation_validity = poor_for_rest2_validation
```

说明这个数据可以测试代码是否跑通，但不适合判断 REST2 是否真的有效。mock 数据目前就是这种情况。

## 第 3 步：一键完整实验

推荐使用一键入口：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\real_model_ready --csv-files your_real_data.csv --output-root outputs\pipeline_real --target-columns target_ghi_5min target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline rest2_calibrated weather_prior weather_prior_rest2
```

真实电站数据更推荐只跑站点级 CSV：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\real_station_model_ready --station-only --output-root outputs\pipeline_real_station --target-columns target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline physics_feature_only clear_sky_power_prior weather_prior_weak weather_prior rest2_calibrated weather_prior_rest2 --train-selected
```

这条命令会自动完成：

```text
schema validation
physics consistency diagnosis
baseline / REST2 / weather_prior / REST2+weather_prior ablation
decision analysis
model_selection.csv generation
```

输出目录：

```text
outputs/pipeline_real/
```

关键输出：

```text
schema_validation.csv
physics_consistency/physics_consistency_summary.csv
ablation_results.csv
ablation_summary.csv
ablation_decisions.csv
promote_candidates.csv
watch_candidates.csv
model_selection.csv
model_selection.yaml
```

如果希望 pipeline 在生成 `model_selection.csv` 后，继续自动训练最终选择的模型，加上：

```powershell
--train-selected
```

完整命令：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\real_model_ready --csv-files your_real_data.csv --output-root outputs\pipeline_real --target-columns target_ghi_5min target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline rest2_calibrated weather_prior weather_prior_rest2 --train-selected
```

最终模型结果会写到：

```text
outputs/pipeline_real/selected_models/
```

## 第 4 步：只替换 CSV 文件名

以后换真实数据，只需要改这两个参数：

```powershell
--csv-dir data\real_model_ready
--csv-files your_real_data.csv
```

例如换成另一个电站：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\real_model_ready --csv-files station_B.csv --output-root outputs\pipeline_station_B --target-columns target_ghi_5min target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline rest2_calibrated weather_prior weather_prior_rest2 --train-selected
```

如果要一次跑多个 CSV：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\real_model_ready --csv-files station_A.csv station_B.csv station_C.csv --output-root outputs\pipeline_real_multi --target-columns target_ghi_5min target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline rest2_calibrated weather_prior weather_prior_rest2 --train-selected
```

如果想跑目录下所有训练 CSV：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\real_model_ready --pattern train_*.csv --output-root outputs\pipeline_real_all --target-columns target_ghi_5min target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline rest2_calibrated weather_prior weather_prior_rest2 --train-selected
```

## 第 5 步：单独训练某个模型

如果不想跑完整 pipeline，也可以手动训练。

训练 baseline：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\train.py --config configs\base.yaml --train-csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h --model-type baseline --output-dir outputs\checkpoints_baseline
```

训练 REST2：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\train.py --config configs\base.yaml --train-csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h --model-type rest2_calibrated --output-dir outputs\checkpoints_rest2
```

训练 weather_prior：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\train.py --config configs\base.yaml --train-csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h --model-type weather_prior --output-dir outputs\checkpoints_weather_prior
```

训练 REST2 + weather_prior：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\train.py --config configs\base.yaml --train-csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h --model-type weather_prior_rest2 --output-dir outputs\checkpoints_weather_prior_rest2
```

## 第 6 步：使用 model_selection 自动训练

如果已经跑过 pipeline 或 `analyze_ablation.py`，会得到：

```text
outputs/pipeline_real/model_selection.csv
```

可以直接用选择表训练：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\train.py --config configs\base.yaml --train-csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h --model-selection-csv outputs\pipeline_real\model_selection.csv --output-dir outputs\checkpoints_selected
```

这时脚本会根据 `dataset + target_column` 自动选择：

```text
baseline
rest2_calibrated
weather_prior
weather_prior_rest2
```

## 第 7 步：评估模型

训练结束后会输出：

```text
best_checkpoint=...
```

用这个 checkpoint 评估：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\evaluate.py --checkpoint outputs\checkpoints_selected\best.pt --csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h
```

输出类似：

```text
{'mae': ..., 'rmse': ...}
```

## 第 8 步：导出预测诊断

导出逐行预测结果：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\export_predictions.py --checkpoint outputs\checkpoints_selected\best.pt --csv data\real_model_ready\your_real_data.csv --output outputs\diagnostics\selected_predictions.csv
```

诊断 CSV 中会包含：

```text
prediction
target
absolute_error
clear_sky_base
k_pred
residual_pred
serial_pred
prior_pred
k_weather
weather_k_prior
prior_weight
rest2_gate
rest2_effective_blend
```

如果想比较 baseline 和候选模型：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\analyze_prediction_diagnostics.py --baseline-csv outputs\diagnostics\baseline_predictions.csv --candidate-csv outputs\diagnostics\weather_prior_predictions.csv --candidate-name weather_prior --output-dir outputs\diagnostics\weather_prior_compare
```

输出：

```text
diagnostic_compare.csv
diagnostic_summary_overall.csv
diagnostic_summary_segments.csv
diagnostic_rest2_calibration.csv
```

## 第 8.5 步：跨站点综合评估

真实电站数据应先形成站点级 `model_ready`：

```text
解放站点观测 + solar_history/forecast 辅助特征 -> train_jiefang_station.csv
庆达站点观测 + solar_history/forecast 辅助特征 -> train_qingda_station.csv
```

然后分别训练/预测每个站点，最后综合评估。不要把 `solar_history / forecast_4h / forecast_1d` 当作平行训练数据集。

如果 pipeline 已经使用 `--train-selected` 训练完选中模型，可以批量导出预测：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\export_selected_predictions.py --model-selection-csv outputs\pipeline_real_station\model_selection.csv --csv-dir data\real_station_model_ready --checkpoint-root outputs\pipeline_real_station\selected_models --output-dir outputs\pipeline_real_station\selected_predictions
```

再做综合评估：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\evaluate_pooled_predictions.py --prediction-csvs outputs\pipeline_real_station\selected_predictions\jiefang_station_target_ghi_4h_baseline_predictions.csv outputs\pipeline_real_station\selected_predictions\qingda_station_target_ghi_4h_baseline_predictions.csv --dataset-names jiefang qingda --output-dir outputs\pipeline_real_station\pooled_eval_4h
```

输出三类结果：

```text
per_station_metrics.csv     # 每个站点单独 MAE/RMSE
pooled_metrics.csv          # 合并所有测试样本后的总体 MAE/RMSE
macro_metrics.csv           # 各站点指标等权平均，避免样本多的站点支配结论
```

## 第 9 步：导出物理向量给 PVMMOE

如果需要把当前物理分支作为一个外部模态接入 PVMMOE，可以直接从 `model_ready` CSV 导出物理向量。

导出结果包括：

```text
*_physics_vectors.npy          # shape = [N, physics_dim]
*_physics_index.csv            # 每一行对应的 dtime、station、target 等索引信息
*_physics_columns.txt          # 物理向量每一维对应的列名
*_physics_metadata.json        # 完整元数据，包含 shape、列名、路径、目标列
```

基础命令：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\export_physics_vectors.py --csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h --output-dir outputs\physics_vectors_real
```

如果 PVMMOE 希望直接读取归一化后的向量，加上：

```powershell
--normalize
```

完整示例：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\export_physics_vectors.py --csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h --output-dir outputs\physics_vectors_real --split all --normalize
```

如果想导出和训练测试集一致的 test 部分：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\export_physics_vectors.py --csv data\real_model_ready\your_real_data.csv --target-column target_ghi_4h --output-dir outputs\physics_vectors_real --split test --normalize
```

如果 PVMMOE 侧更方便读 CSV，可以加：

```powershell
--write-csv
```

注意：`.npy` 和 `physics_index.csv` 按行号一一对应。PVMMOE 接入时应使用 `physics_metadata.json` 中的 `physics_columns` 确认每一维含义。

## REST2、weather_prior、REST2+weather_prior 怎么判断

真实数据上建议先看：

```text
ablation_decisions.csv
model_selection.csv
```

如果结果是：

```text
weather_prior > baseline
weather_prior_rest2 <= weather_prior
```

说明天气修正先验有用，但 REST2 校准没有额外帮助，后续应继续正式化 `weather_prior`。

如果结果是：

```text
rest2_calibrated > baseline
weather_prior_rest2 > weather_prior
```

说明 REST2 在真实数据上有独立贡献，可以继续探索 REST2 与天气先验的更合理组合。

如果真实数据上出现：

```text
4h: 所有候选都是 reject
1d: high_variance 或 weak_or_mixed
model_selection 只选择 baseline
```

这是正常的安全行为，不代表代码失败。它表示当前物理分支没有稳定超过 baseline，最终训练应该继续使用 baseline。

这时优先检查：

```text
physics_consistency_summary.csv
power_like_score
power_prior_recommendation
target_clear_sky_scale_median_day
clear_sky_upper_exceed_rate
night_nonzero_target_rate
target_weather_prior_corr
weather_prior_mae_gain_vs_current_ghi
```

如果 `power_prior_recommendation = try_clear_sky_power_prior`，说明目标可能更像电站功率而不是 GHI，应优先测试：

```powershell
--variants baseline physics_feature_only clear_sky_power_prior weather_prior_weak
```

如果只是天气先验高方差，优先测试弱融合：

```powershell
--variants baseline weather_prior_weak weather_prior
```

如果连续两轮真实数据中 `REST2 / weather_prior / weather_prior_rest2` 都是 reject，则不要继续复杂化 REST2，应把 REST2 固定为离线诊断特征。

如果物理一致性诊断显示：

```text
poor_for_rest2_validation
```

不要用这份数据直接否定 REST2，应先检查：

```text
时间戳是否错位
夜间目标是否异常非零
目标列是功率还是辐照度
单位是否一致
目标是否大量超过晴空上界
```

## 常用命令汇总

生成 mock 数据：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\generate_mock_training_csv.py --raw-output-dir data\mock_raw --enriched-output-dir data\mock_enriched --model-ready-output-dir data\mock_model_ready
```

检查物理一致性：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\diagnose_physics_consistency.py --csv-dir data\mock_model_ready --output-dir outputs\physics_consistency
```

一键 pipeline：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\run_dataset_pipeline.py --config configs\base.yaml --csv-dir data\mock_model_ready --csv-files train_jiefang_station.csv --output-root outputs\pipeline_jiefang --target-columns target_ghi_5min target_ghi_4h target_ghi_1d --seeds 42 43 44 45 46 --variants baseline physics_feature_only clear_sky_power_prior weather_prior_weak weather_prior rest2_calibrated weather_prior_rest2 --train-selected
```

单独训练：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\train.py --config configs\base.yaml --train-csv data\mock_model_ready\train_jiefang_station.csv --target-column target_ghi_4h --model-type weather_prior --output-dir outputs\checkpoints_weather_prior
```

评估：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\evaluate.py --checkpoint outputs\checkpoints_weather_prior\best.pt --csv data\mock_model_ready\train_jiefang_station.csv --target-column target_ghi_4h
```

导出预测：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\export_predictions.py --checkpoint outputs\checkpoints_weather_prior\best.pt --csv data\mock_model_ready\train_jiefang_station.csv --output outputs\diagnostics\weather_prior_predictions.csv
```

导出物理向量：

```powershell
C:\Users\ADMIN\AppData\Local\Programs\Python\Python38\python.exe scripts\export_physics_vectors.py --csv data\mock_model_ready\train_jiefang_station.csv --target-column target_ghi_4h --output-dir outputs\physics_vectors_mock --normalize
```

## 注意事项

1. `data/mock_model_ready/` 只用于测试代码流程，不适合判断 REST2 真实效果。
2. 真实数据必须先整理成固定 `model_ready` schema。
3. `REST2` 不建议全局默认开启，应通过真实数据 ablation 判断。
4. `weather_prior` 是当前最值得优先验证的物理分支。
5. 如果服务器上使用 GPU，把 `configs/base.yaml` 中 `training.device` 改成 `cuda`。
6. 如果出现缺列错误，先对照 `data/mock_model_ready/*.csv` 补齐字段。
