# GitHub 上传检查清单

## 必须上传

- [ ] `.gitignore`
- [ ] `.gitattributes`
- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] `pyproject.toml`
- [ ] `configs/server_station.yaml`
- [ ] `src/pv_physics_moe/` 完整源码
- [ ] `scripts/` 中的站点预处理、训练、评估和测试脚本
- [ ] `tests/test_integrated_model.py`
- [ ] `tests/test_station_server_pipeline.py`
- [ ] `EXPERIMENT_2024_2025_ZH.md`
- [ ] `MODEL_ARCHITECTURE_REPORT_ZH.md`
- [ ] `RUN_GUIDE_ZH.md`
- [ ] `data/raw/station/.gitkeep`
- [ ] `outputs/.gitkeep`

## 禁止上传

- [ ] 2024 年真实训练数据
- [ ] 2025 年甲方评测数据
- [ ] 任何 CSV/XLSX 原始数据
- [ ] `data/processed/`
- [ ] `.venv/`
- [ ] `outputs/station/`
- [ ] `.pt/.pth/.ckpt` 模型权重
- [ ] `__pycache__/` 和 `.pytest_cache/`
- [ ] API 密钥、GitHub Token、服务器密码
- [ ] 含用户名、内网地址或敏感绝对路径的本地配置

## 提交前命令

```bash
git add .
git status
```

认真检查 `Changes to be committed`，确保只有代码、配置模板和文档。

```bash
git commit -m "Initial release: PVMMOE REST2 PCD five-horizon model"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

## 上传后检查

- [ ] GitHub 首页正确显示 README
- [ ] 仓库中不存在真实数据
- [ ] 仓库中不存在 2025 年文件
- [ ] 仓库中不存在 checkpoint
- [ ] `configs/server_station.yaml` 显示 2024 train / 2025 holdout
- [ ] 在另一目录执行 `git clone` 后能安装依赖
- [ ] `python scripts/run_smoke_tests.py` 显示 4 项通过
- [ ] `python scripts/run_station_smoke_test.py` 端到端通过

## 公开仓库前

- [ ] 确认原 PVMMOE、REST2、PCD 代码的许可证和再发布要求
- [ ] 确认项目代码归属与甲方保密要求
- [ ] 确认是否需要添加开源 `LICENSE`

如果上述授权尚未确认，建议先创建 GitHub Private repository。
