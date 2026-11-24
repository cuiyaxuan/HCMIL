# HCMIL 星形胶质细胞 Braak 分析

本仓库提供论文 “Interpretable single-nucleus learning reveals an astrocyte program tracking local neocortical tau rather than Braak stage”（Yaxuan Cui、Jingwen Yan）的 ROSMAP DLPFC 独立分析代码，包含 pathway/covariate teacher、Astrocyte-only student、OOF 基因 knockout attribution 和基因发现流程。候选面板最多 20 个基因；若通过配置筛选标准的基因不足 20 个，输出会少于 20 个。

代码的核心分析文件与原始 standalone 项目逐文件核对未发现差异。公开版已在 IU Quartz H100 上完成输入检查、teacher OOF、scVI 特征生成和五个 Student seed 的 OOF 运行，指标与原始运行大体接近。**当前这轮公开版 H100 运行的 750 个基因 knockout 分块和最终汇总仍需完成**，完成前不能说最终基因发现结果已完整复现。详见[复现记录](docs/REPRODUCIBILITY.md)。

本仓库不包含受控访问的 ROSMAP H5AD、供体级数据、模型权重或 sealed test donor 名单。数据必须通过正式流程获取。这里的软件用于研究，不是临床工具。

## 环境安装

需要 Python 3.11 或 3.12，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/)。在项目根目录运行：

```bash
uv sync
uv run python examples/model_smoke_demo.py
uv sync --extra analysis
```

合成数据 demo 只验证模型接口和环境，不训练或评估模型，也不复现论文结果。

## 准备数据和通路资源

1. 按 ROSMAP 数据访问流程取得授权 H5AD，放入 `data/ROSMAP_enriched_metadata.h5ad`，或修改配置中的 `data.h5ad_path`。
2. 按各自的访问及再分发条款取得 Hallmark 和 Reactome GMT 文件，放入 `resources/`。默认配置使用 `hallmark.v2026.1.Hs.symbols.gmt` 与 `reactome.v2026.1.Hs.symbols.gmt`。详见[资源说明](resources/README.md)。
3. 检查 YAML 中的列名、筛选条件、协变量、split 和输出目录。
4. 在正式训练前检查 H5AD：

```bash
uv run --extra analysis python inspect_input.py \
  --config configs/student_gene_attribution.yaml
```

## 运行完整流程

在有足够内存和受支持 GPU 的计算资源上，使用同一配置依次运行：

```bash
uv run --extra analysis python prepare_standalone_data.py --config configs/student_gene_attribution.yaml
uv run --extra analysis python train_teacher.py --config configs/student_gene_attribution.yaml
uv run --extra analysis python train_scvi_and_build_cellscale_features.py --config configs/student_gene_attribution.yaml

for seed_index in 0 1 2 3 4; do
  uv run --extra analysis python train_ast_student_models.py \
    --config configs/student_gene_attribution.yaml --seed-index "$seed_index"
done

for task_id in $(seq 0 749); do
  uv run --extra analysis python attribute_student_genes.py \
    --config configs/student_gene_attribution.yaml --task-id "$task_id"
done

uv run --extra analysis python aggregate_gene_attribution.py --config configs/student_gene_attribution.yaml
uv run --extra analysis python run_gene_discovery.py --config configs/student_gene_attribution.yaml
uv run --extra analysis python workflow_status.py summary --config configs/student_gene_attribution.yaml
```

完整分析不适合在普通笔记本或登录节点上运行。SLURM 用户应将 Student 和 750 个 attribution task 作为 job arrays 提交，并按 `afterok` 依赖串接工作阶段；参见[SLURM 指引](docs/SLURM.md)。输出写入被 git 忽略的 `artifacts/`。

## 评估注意事项

不要用 sealed test 供体的标签调参、挑基因或选模型。报告实际队列、split、软件版本、随机种子、缺失值处理和相对论文的偏差。论文报告的 20-gene panel 未通过预注册的 sealed DLPFC test；SEA-AD MTG 转移结果不应描述为跨脑区的普遍验证。

## 引用与许可

请引用论文和本代码仓库。期刊 DOI 确定后请补入英文 README 和 `CITATION.cff`。代码采用 MIT license；受控数据与通路数据库受各自条款约束。
