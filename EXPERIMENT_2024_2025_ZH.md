# 2024 年训练、2025 年甲方评测实验设置

## 1. 划分原则

本实验不再对 2024—2025 全部样本按 70%/15%/15% 混合切分，而是按完整日历年隔离：

```text
2024 年完整样本
├─ 前 85%：train
└─ 后 15%：validation

2025 年完整样本
└─ test：甲方评测集
```

“完整样本属于某年”表示该样本的以下时间全部落在同一年：

- 4 h 历史窗口起点；
- 当前预测起点；
- 最长 1 d 预测目标时刻。

因此，训练样本不会因为跨年历史或跨年目标而读取 2025 数据；测试样本也会从历史窗口完整进入 2025 后才开始，便于甲方仅使用 2025 文件独立复现。

## 2. 训练/验证隔离

2024 年内部按时间顺序划分，不随机打乱切分。初始边界为 85%/15%，随后对训练集末端执行 1440 min purge：

```text
最后一个训练样本的 1 d 目标时间 < 第一个验证样本的预测起点
```

这避免同一未来时段同时出现在训练标签和验证输入/标签中。被 purge 的边界样本不用于任何拟合。

## 3. 2025 年不会影响训练

以下统计量只由 2024 年 `train` 子集重新拟合：

- 13 维历史序列均值和标准差；
- 26 维物理向量均值和标准差；
- 功率损失缩放系数；
- GHI/DNI/DHI 损失缩放系数。

`train_station.py` 只打开 `train_end_indices.npy`。2025 年 `test_end_indices.npy` 不参与梯度更新、早停、超参数选择或归一化。

## 4. 配置

```yaml
experiment:
  fixed_year_split: true
  training_years: [2024]
  holdout_years: [2025]
  train_fraction_within_training_years: 0.85
```

如需复现实验，不要将 `fixed_year_split` 改为 `false`。

## 5. 运行

```powershell
python scripts\prepare_station_dataset.py --config configs\server_station.yaml
python scripts\train_station.py --config configs\server_station.yaml --smoke
python scripts\train_station.py --config configs\server_station.yaml
```

只有在允许查看甲方评测结果时才运行：

```powershell
python scripts\evaluate_station.py --config configs\server_station.yaml --checkpoint outputs\station\checkpoint_last.pt --split test
```

如果已经完成过全数据预处理，只想重新生成固定年份索引和归一化，可运行：

```powershell
python scripts\prepare_station_dataset.py --config configs\server_station.yaml --resplit-only
```

## 6. 审计信息

`metadata.json` 会记录：

```text
split_policy
training_years
holdout_years
train_fraction_within_training_years
validation_fraction_within_training_years
train_validation_embargo_minutes
normalization_fit_years
client_holdout_locked
split_samples
split_time_ranges
```

`normalization.json` 的四组统计量也会分别标记：

```text
fit_split: train
fit_years: [2024]
```

这些字段可作为向甲方说明“2025 年没有参与训练”的可审计证据。
