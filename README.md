# REST2–PCD–PVMMoE 光伏功率预测

系统通过统一的配置驱动 Parquet 接口支持 YLJ 与 Luoyang，包含 REST2 物理特征、PCD 可行域投影、PVMMoE Top-k 专家、可选图像模态、分布式训练和正式指标导出。

## 两个数据集

| 项目 | YLJ | Luoyang |
|---|---|---|
| 配置 | `configs/ylj_parquet.yaml` | `configs/luoyang_parquet.yaml` |
| 数据文件 | `YLJ-Unified_format-with_DNI_DHI.parquet` | `Luoyang-Unified_format-V1-with_DNI_DHI.parquet` |
| 原始颗粒度 | 15 min | 5 min |
| 预测范围 | 未来 240 min | 未来 240 min |
| 输出 | 16 步，t+15…t+240 | 48 步，t+5…t+240 |
| 正式评测点 | t+15、t+240 | t+15、t+240 |
| 容量/NRMSE 分母 | 468 MW | 48629.73（原功率单位） |
| 图像 | 关闭 | 可选异步天空图像 |
| 训练精度 | FP32 | FP32 |

输出步数由 `240 / forecast_step_minutes` 校验或推导；正式评测依据导出记录中的 `horizon_minutes` 选择，不使用固定数组下标。

## 数据接口

### YLJ

- 读取真实 `-onsite` / `-NWP_` 字段；现场 GHI 和估算 DNI/DHI 用于物理输入与辅助监督。
- PWAT 按已确认的 mm 转换为 REST2 所需 cm。
- 功率容量及 NRMSE/NMAE 分母固定为 468 MW。
- 夜间负功率按显式配置截断至 0；缺失功率不插值、不后向填充。

### Luoyang

- 站点参数为纬度 34.700、经度 112.285、海拔 220 m、时区 Asia/Shanghai。
- `GHI-onsite`、`estimated_DNI-onsite`、`estimated_DHI-onsite` 是“值列表 + 时间戳列表”；适配器按内部时间戳精确匹配主时间轴，不依赖列表位置。
- `msl-NWP_forecast` 根据海拔换算站点气压；风速由 `u10/v10-NWP_forecast` 合成。
- 图像从 `[issue_time-75 min, issue_time]` 选择，拒绝未来图像，缺图由 mask 表示。

### 公共训练与评测

- 同一 `ConfigurableParquetDataset` 返回 serial、REST2、功率、辐照度以及可选图像张量。
- 动态 Top-k MoE 的 DDP 使用 `find_unused_parameters=True`。
- 未归一化 REST2 特征含 Pa 量级气压，两个配置使用 FP32，避免 FP16 溢出。
- checkpoint 保存字段顺序、训练集归一化、容量、站点信息、数据 schema 和物理量来源。
- 导出 `point_predictions.csv` 与 `official_test_metrics.json`；指标使用原始功率单位及固定容量分母。

## 安装

要求 Python 3.9+，依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

## 训练

YLJ：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/train_parquet.py \
  --config configs/ylj_parquet.yaml \
  --output-dir outputs/ylj_parquet
```

Luoyang：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/train_parquet.py \
  --config configs/luoyang_parquet.yaml \
  --output-dir outputs/luoyang_parquet
```

数据集在加载时直接校验字段、颗粒度、时间戳和目标完整性，校验失败会终止训练。

## 正式评测

YLJ：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/evaluate_parquet.py \
  --config configs/ylj_parquet.yaml \
  --checkpoint outputs/ylj_parquet/checkpoint_last.pt \
  --output-dir outputs/ylj_official_test \
  --split test
```

Luoyang：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/evaluate_parquet.py \
  --config configs/luoyang_parquet.yaml \
  --checkpoint outputs/luoyang_parquet/checkpoint_last.pt \
  --output-dir outputs/luoyang_official_test \
  --split test
```

评测输出：

- `point_predictions.csv`：`issue_time,target_time,horizon_minutes,y_true,y_pred`；
- `official_test_metrics.json`：t+15/t+240 的 overall、hard-delta、方向准确率、NRMSE/NMAE，以及 PCD 诊断。

## 关键文件

- `src/pv_physics_moe/data/parquet_dataset.py`：统一 Parquet、时间戳 list 和图像适配。
- `src/pv_physics_moe/parquet_config.py`：两数据集配置与预测步数校验。
- `scripts/train_parquet.py`：单卡/8 卡训练。
- `scripts/evaluate_parquet.py`：分布式评测与导出。
- `src/pv_physics_moe/evaluation.py`：按 `horizon_minutes` 计算正式指标。
- `docs/PARQUET_PIPELINE_ZH.md`：数据契约与实现细节。

真实 Parquet、图像、checkpoint、日志和评测输出不会提交到仓库。
