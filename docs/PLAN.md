# 实施计划

目标：可直接运行、可逐阶段续跑的 RuNNer/MACE 主动学习项目。
依据：[DESIGN.md](DESIGN.md)。在用户已授权“搭建并同步”的范围内直接实施。

1. 环境与 RuNNer：固定 MACE/graph 兼容版本，安装 runnerase，Slurm 编译 libRuNNer/RuNNer.x，保留版本及验证证据。
2. 数据与委员会：先写测试验证分歧公式、重复构型筛选、来源分组划分及结果保护；实现 rnal/common.py、backends.py、sampling.py。
3. VASP 与训练：先写测试拒绝未收敛输出和错误原子顺序；实现 prepare/collect 和 MACE/RuNNer 训练输入生成。单点标签使用一致能量约定与显式参考设置。
4. 命令与文档：提供 doctor、sample、select、prepare-vasp、collect、prepare-train、benchmark、status 命令及 Slurm 包装。
5. 验证与同步：单元测试、真实模型 Slurm 冒烟、独立代码审查；记录缺失科学输入；正常提交推送 main，核对远端提交。

重点检查：失败/未收敛标签不得进入训练；ASE 的排序与边界条件；同一来源不得泄漏到验证集；重复提交不得覆盖结果；未校准/模拟/4G 电荷缺失的状态必须显式保留。
