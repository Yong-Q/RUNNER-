# RuNNer / MACE-POLAR 气体主动学习工作流

中文教程和工程实现一起维护：从委员会采样，到 VASP 单点候选、标签汇总和训练任务准备。原生 Slurm，统一 eV / Å / eV·Å⁻¹，计时使用 **us/atom/step**。

**先读 [HTML 教程](docs/index.html)**。GitHub 文件页显示源码；下载仓库后直接用浏览器打开 `docs/index.html`，或在仓库运行 `python -m http.server 8000` 后打开 `/docs/`。页面支持离线阅读、手机、打印及交互式力分歧演示。

## 当前状态

- MACE-POLAR 原生委员会：CO₂、CH₄、N₂ 输入；实跑 CO₂ 周期盒短采样、筛选、计时。
- RuNNer 2.0：本机编译成功；官方 Ag₃ **4G 两成员委员会实际推理**，可通过同一接口计算分歧与计时。它不是气体模型。
- VASP：生成独立单点输入、Slurm 脚本；记录排序、电子状态、结构和输入摘要；拒收未收敛/不匹配的标签。
- 训练准备：MACE / PolarMACE 多种子任务；RuNNer **2G** 数据转换、训练及原生委员会组装。4G 电荷标签和两阶段训练尚未接通。
- 所有阶段写新目录，训练和 VASP 执行有独占记录；组间重复结构、来源泄漏和非法标签有测试。

**现已完成 7 个真实 VASP 标签、3 个 MACE-POLAR 微调成员和 3 个 RuNNer 2G 训练成员。** 新 MACE-POLAR 完成 5 步短采样；RuNNer 因分歧过大在初始帧停止。当前权重未达到生产要求，未完成长时 MD 与不确定度校准。详见 [本轮结果](evidence/round001/verification.json)。

关键核验：原始 MACE-POLAR 在同一个 CO₂ 构型上，与匹配的 ORCA ωB97M-V/def2-TZVPD 单点基本一致；相对 PBE-D3 的大差异不能直接称为原模型误差。原始 checkpoint 保留不变；本轮 PBE 微调模型只作流程试验。详见 [参考方法说明](docs/REFERENCE_METHOD.md) 和 [实际力核验](evidence/round001/force_audit/verification.json)。VASP 的能量/力差分自洽，但 520→600 eV 仍带来约 0.0229 eV/Å 的力变化，不能视为数值精度已收敛。

## 两个 Python 环境

RuNNer 当前接口要求 NumPy 2，而已验证的 MACE-POLAR 配对使用 NumPy 1，因此分别运行，共享 extxyz 数据。

```bash
# Python 3.10，RuNNer / 数据 / VASP 工作流环境
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-runner.txt

# 另一独立环境运行 MACE，不在已有 RuNNer 环境中覆盖依赖
python3.10 -m venv .venv-mace
.venv-mace/bin/python -m pip install -r requirements-mace.txt
```

本机已配置：

- RuNNer：`/home/qiuyong/bza/MLLP/RUNNER-/.venv/bin/python`
- MACE：`/home/qiuyong/bza/MLLP/committee_probe_20261022/env/bin/python`
- VASP 赝势：`VASP_PP_PATH=/home/script/vasp_pp/`，只在本机读取。
- RuNNer 共享库：`build/runner_v4/libRuNNer.so`；上游源码固定版本见 `evidence/provenance.json`。

权重、第三方源码、环境、POTCAR、计算结果均不上传。官方 MACE-POLAR 模型下载入口在 [release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_polar_1)，需要 S/M/L 三个 `.model` 文件。下载完成后：

```bash
.venv/bin/python scripts/setup_local.py --model-dir /actual/model/directory
.venv/bin/python scripts/run_cli.py doctor --config local/runner_ag.json
.venv-mace/bin/python scripts/run_cli.py doctor --config local/polar.json
```

`setup_local.py` 会拒绝替换已有配置。首次运行 RuNNer 示例还需获取其上游源码；它从上游 regression 示例复制 Ag₃ 模型到 `local/`。实际气体用途请改用训练好的气体模型目录。

## RuNNer 构建

```bash
git clone https://gitlab.com/runner-suite/runner2.git external/runner2
git -C external/runner2 checkout 8046a9b86d4a9b3aafc386abcd47fef5d3a86607
sbatch scripts/build_runner.slurm
```

构建脚本是本集群站点配置，使用共享 GCC/CMake/OpenBLAS/FFTW 和已有 Conda sysroot。首次配置需要可执行的 m4 放在 `local/bin/m4` 或 PATH；本机从已有 `/usr/bin/m4` 获取。别的集群应根据 [官方构建说明](https://runner-suite.gitlab.io/runner2/installation/source/) 调整路径。此构建为无 MPI、支持 OpenMP 的共享库；不声称复现论文优化性能。脚本中固定的日志名再次构建前应改为新名字，保留旧构建证据。

## 原生 Slurm 小测试

先检查脚本内仓库与 Python 路径，再提交：

```bash
sbatch scripts/smoke_polar.slurm
sbatch scripts/smoke_runner.slurm
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList
```

RuNNer 的银示例可能在初始构型就因分歧过大停止；这验证停止分支，不是稳定性通过。MACE 示例输出到带 job ID 的新目录。`benchmark.json` 的计时是整个委员会的能量/力调用；不同模型和不同体系的小测试不能直接计算速度比。

## 从候选到训练输入

```bash
# PBE-D3(BJ) 文件是参数格式示例；先确定适合研究目标的参考方案
.venv/bin/python scripts/run_cli.py prepare-vasp \
  --input examples/initial_candidates.extxyz \
  --reference configs/reference_pbe_d3.example.json \
  --out runs/labels_round_001

# 检查输入后提交实际单点；每个 frame 目录各有 job.slurm
sbatch runs/labels_round_001/frame_00000/job.slurm

.venv/bin/python scripts/run_cli.py collect \
  --tasks runs/labels_round_001 --out runs/dataset_round_001

# 先修改 config 中 python、foundation_model 或 RuNNer 模板/可执行文件路径
.venv/bin/python scripts/run_cli.py prepare-train \
  --input runs/dataset_round_001/labeled.extxyz \
  --config configs/train_polar.example.json --out runs/train_round_001
sbatch runs/train_round_001/seed_17/job.slurm
```

参考配置当前仅支持中性单重态无外场；NSW=0、IBRION=-1。采样候选必须保留来源 group；若缺失，整个输入文件视为同一个来源。至少两组独立来源才能划分训练/验证。训练完成后还需独立评估和校准，不会自动替换现有委员会。

## 组装 RuNNer 2G 委员会

```bash
.venv/bin/python scripts/run_cli.py assemble-runner \
  --training runs/train_runner_round_001 \
  --library build/runner_v4/libRuNNer.so --out runs/runner_committee_round_001
```

检查所有成员成功退出、描述符缩放和结构一致后，输出 `committee.json`。这不替代精度验收。首轮试验数据、模型错误及停止结果都按实际情况保留。

## 验证与维护

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_cli.py --help
```

- [中文 HTML 教程](docs/index.html)
- [设计与数据约束](docs/DESIGN.md)
- [验证证据](evidence/verification.json)
- [来源与 GitHub skills](evidence/provenance.json)

文档使用从 GitHub 拉取的 Anthropic `doc-coauthoring` 和 `frontend-design` 技能，并进行了独立读者检查。它们位于本机 `external/skills/`，不是项目运行时依赖。

本仓库提供工作流代码；RuNNer、MACE、runnerase、VASP 和模型权重分别遵循各自许可。VASP 软件及 POTCAR 需要用户已有的合法安装，不随仓库分发。
