# 公共健康与临床部署变化下的共形预测基准

这个仓库提供一套可复用的评测规则和工具：其他研究者可以接入自己的二分类
共形校准方法，在相同的队列、部署变化场景、校准抽样和指标下与基线比较。
**我们的论文实验是使用该基准的一个样例，图表和实验记录是样例输出。**

- `benchmark/`：基准接口、任务配置、数据与预测准备、统一评测和汇总命令。
- `examples/source_split.py`：可复制修改的外部方法示例。
- `examples/health-study/`：基于 Zenodo v1.4.15 整理的实验样例，不含论文正文或补充文章全文。
- `examples/legacy-experiments/`：重构前 GitHub 仓库的完整历史快照。
- `docs/`：评测规则、真实数据流程、方法接入和参考结果说明。

## 从接入自己的方法开始

在仓库根目录使用 Python 3.12：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-benchmark.txt
python -m benchmark fixture --workspace work/smoke
python -m benchmark evaluate --workspace work/smoke \
  --methods source_split source_mondrian target_split target_mondrian examples/source_split.py \
  --out work/smoke/results.csv
python -m benchmark compare --input work/smoke/results.csv --out work/smoke/comparison
```

这个快速样例使用合成数据，只检查软件流程，不构成真实健康数据的实验结果。
外部示例复现内置 source split；研究者可以复制该文件并实现自己的方法。

## 使用真实数据

```bash
python -m pip install -r requirements.txt
python -m benchmark prepare --workspace work/health
python -m benchmark evaluate --workspace work/health \
  --methods source_split source_mondrian target_split target_mondrian examples/source_split.py \
  --out work/health/results.csv
python -m benchmark compare --input work/health/results.csv --out work/health/comparison
```

准备流程获取 32 个公开数据文件，构建三个队列、12 个场景，训练 60 个固定源域模型。
默认每组抽取 300 个源域校准样本和 300 个目标域校准样本，重复五次。
五个方法共输出 1,500 条评测记录，以及按条件、按数据集和各数据集等权的汇总。
真实数据重建需要网络、足够磁盘和约 32 GB 内存；原始记录与概率数组不上传。
完整说明见 [真实数据流程](docs/REAL_DATA.md)。

方法只能得到其声明允许使用的校准信息，测试标签由评测程序保留。当前统一接口
针对固定二分类概率输出的校准方法；需要协变量的 OT/LR-QR 等另有论文样例运行器。
原论文主分析与新增 outcome-blind 协议有明确区别，不能混用结果。
见 [协议说明](examples/health-study/PROTOCOL_MAP.md) 和 [基准定义](docs/BENCHMARK.md)。

GitHub 接口版本为 2.0.0；Zenodo 论文支撑包版本为 1.4.15，
DOI [10.5281/zenodo.23461041](https://doi.org/10.5281/zenodo.23461041)。两者版本分别管理。
