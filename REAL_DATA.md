# Folsom 真实数据接入

原始数据目录：`C:\Users\ADMIN\Desktop\dataset`。预处理脚本只读取该目录，不会修改任何原始 CSV 或图像。

数据契约：16 个连续分钟的 `[GHI,DNI,DHI,气象]` 序列和 16 张天空图像，预测 15 分钟后的 `[GHI,DNI,DHI]`。REST2 为目标时刻生成 26 维物理模态，PCD 在物理单位中保证 `GHI=cos(zenith)*DNI+DHI`。按目标年份划分：2014/2015/2016 对应训练/验证/测试。

```powershell
python scripts\prepare_folsom_dataset.py --dataset "C:\Users\ADMIN\Desktop\dataset"
python scripts\train_folsom.py --smoke
python scripts\train_folsom.py
```

预处理结果采用 `.npy` 内存映射和图像惰性读取，不复制原始天空图像。所有归一化统计量只由 2014 训练样本拟合；`physics_raw` 和目标始终保留 W/m² 物理单位。
