# 洛阳数据与可选图像模态契约

## 结论

融合模型现在支持洛阳所需的三种输入模态，但当前五表式站点数据预处理器不能直接
读取洛阳Parquet。接入洛阳时只需新增数据适配器，不需要再次修改融合层、MoE或
REST2-PCD核心。

## 模型输入

模型始终需要：

- `serial`: `[B,T,C]`，洛阳配置应为5分钟采样、16个历史点、9个时序字段；
- `physics`及`physics_raw`: `[B,H,26]`，由REST2输入构造；
- `future_zenith`或`future_cos_zenith`: `[B,H]`。

启用 `model.use_image: true` 后可额外传入：

- `images`: `[B,F,3,64,64]`；
- `image_valid_mask`: `[B,F]`，有效图像为1，缺失或补齐位置为0；
- `image_time_offsets`: `[B,F]`，每张图相对于 `issue_time` 的分钟偏移，历史图像
  应小于或等于0。

图片可以与5分钟时序不对齐。建议从 `[issue_time-75min, issue_time]` 收集最多
16张图片，按真实时间戳排序；超过16张时均匀抽样，不足时补零并由掩码标记。
整个样本没有图片时，可以省略以上三个图片字段，或传入全零掩码，两种方式等价。

关闭 `model.use_image: false` 时，模型不会创建或调用图像编码器，即使批次中包含
图片字段也不会改变原两模态路径。

## 洛阳Parquet字段映射

- 时间：`timestamp`
- 预测目标：`final_power`
- 图片相对路径列表：`asi_path`
- 图片时间戳列表：`asi_path_timestamps`
- 9个时序字段：
  `final_power`、`GHI_mean_observe`、`GHI_mean_forecast`、`msl_forecast`、
  `t2m_forecast`、`u10_forecast`、`v10_forecast`、`u100_forecast`、
  `v100_forecast`

洛阳适配器还必须完成以下工作：

1. 根据站点经纬度和每个未来目标时间计算太阳天顶角或 `mu0`；
2. 将 `msl_forecast` 转换或确认成Pa，作为REST2的 `pressure_pa`；
3. 使用历史 `GHI_mean_observe` 作为 `input_ghi/previous_ghi`；
4. 未提供的 `pwv_cm`、`aod700` 等大气量使用配置默认值，并记录该降级；
5. 生成48个5分钟未来目标时，令模型 `forecast_horizon=48`，并同步配置未来时间；
6. 严禁选取晚于 `issue_time` 的图像，且归一化统计只能来自训练集。

仅修改 `serial_input_dim=9`、采样间隔和预测长度并不足以直接训练；如果没有生成
26维REST2物理特征和未来太阳几何输入，模型会因数据契约不完整而拒绝运行。
