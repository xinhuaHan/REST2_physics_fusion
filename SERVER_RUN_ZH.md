# 服务器部署与五表五时距数据说明

## 1. 固定任务配置

正式配置文件为 `configs/server_station.yaml`，当前已写入：

| 项目 | 值 |
|---|---:|
| 经度 | 100.641°E |
| 纬度 | 29.919°N |
| 海拔 | 4300 m |
| 时区 | Asia/Shanghai |
| 采样间隔 | 15 min |
| 历史窗口 | 16 步，即 4 h |
| 预测时距 | 15、30、60、240、1440 min |
| 输出步数 | 5 |
| PWAT 单位 | cm，与 Folsom/REST2 一致 |
| 夜间负功率 | 截断至 0 |
| 标准气压 | 自动按海拔计算，约 59.27 kPa |

气压使用标准大气公式：

```text
P(h) = 101325 × (1 - 2.25577×10⁻⁵ × h)^5.25588
P(4300 m) = 59268.17 Pa
```

如果以后取得站点实测气压，可把 `default_pressure_pa: null` 改成实测 Pa 值，实测值会覆盖标准大气估算。

## 2. 路径和文件配置

代码不含 Windows 绝对路径。服务器可直接修改 YAML，也可设置：

```bash
export STATION_DATA_ROOT=/data/pv_station/raw
export STATION_PROCESSED_DIR=/data/pv_station/processed
```

五个输入文件支持 CSV 和 XLSX：

| 角色 | 默认文件 | 必需列 |
|---|---|---|
| 功率与 GHI | `site_power_ghi.csv` | `dtime, observe_power, observe_ghi` |
| 三分量辐照度 | `site_irradiance.csv` | `dtime, observe_ghi, observe_dni, observe_dhi` |
| 4 h 天气预报 | `forecast_4h.csv` | `dtime, GHI, TEMP, WS, WD, PREC, PWAT, SDWE, interval` |
| 1 d 天气预报 | `forecast_1d.csv` | 同上 |
| 站点实测天气 | `site_weather.csv` | `dtime, GHI, TEMP, WS, WD, PREC, PWAT, SDWE` |

预报表建议同时包含 `report_date, report_time, update_time, govern_flag`。若前两个发布时间列缺失，程序使用 `dtime - interval` 推导发布时间。

## 3. 五时距对齐与防泄漏

每一个样本以历史窗口末端 `t₀` 为预测起点，同时构造：

```text
t₀+15min, t₀+30min, t₀+60min, t₀+240min, t₀+1440min
```

对任意时距 `h`，只接受满足以下条件的预报记录：

```text
forecast_issue_time <= target_valid_time - h = t₀
```

- 15/30/60/240 min：4 h 预报为主，1 d 预报可作安全回退。
- 1440 min：1 d 预报为主，4 h 预报仅在确实提前发布且覆盖时作为回退。
- 缺失对齐点只允许把更早有效时刻的预报向前保持，不使用未来行反向插值。
- 再缺失时使用预测起点的实测天气持久性，不使用目标时刻实测天气。
- 划分按时间顺序进行，归一化统计量只由训练段计算。

最长时距为 1 天，因此完整样本至少需要“4 h 历史 + 1 d 未来标签”连续无缺口。

## 4. 输出张量

预处理后的单样本形状：

| 张量 | 形状 | 含义 |
|---|---|---|
| `serial` | `[16, 13]` | 过去 4 h 的功率、辐照度、天气历史 |
| `physics` | `[5, 26]` | 五个目标时距各自的 REST2 物理向量 |
| `physics_raw` | `[5, 26]` | 未标准化物理量，供 PCD/诊断使用 |
| `future_zenith` | `[5]` | 五个目标时刻太阳天顶角 |
| `target` | `[5, 1]` | 五时距功率标签 |
| `irradiance_target` | `[5, 3]` | 五时距 GHI/DNI/DHI 辅助标签 |

批量输入后分别为 `[B,16,13]`、`[B,5,26]`、`[B,5,1]` 和 `[B,5,3]`。

## 5. 运行命令

```bash
cd PVMMOE_REST2_PCD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/check_station_tables.py --config configs/server_station.yaml
python scripts/prepare_station_dataset.py --config configs/server_station.yaml
python scripts/run_station_smoke_test.py
python scripts/train_station.py --config configs/server_station.yaml --smoke
python scripts/train_station.py --config configs/server_station.yaml
python scripts/evaluate_station.py \
  --config configs/server_station.yaml \
  --checkpoint outputs/station/checkpoint_last.pt \
  --split test
```

评估结果按五个时距分别给出功率 RMSE/MAE 和 GHI/DNI/DHI RMSE，同时输出 PCD 最大闭合误差。
