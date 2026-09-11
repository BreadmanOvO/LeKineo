# SmolVLA LIBERO Agent

基于 Hugging Face LeRobot/SmolVLA 的具身智能训练与 Agent 工程项目。当前完成环境验证、LIBERO baseline 接口联调和无泄漏数据审计；训练、闭环评测与 Agent runtime 将按日迭代加入。

## 数据审计

数据集固定为 `HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e`。

```bash
python scripts/inspect_dataset.py --out /path/to/data_statistics.json
python scripts/make_task_split.py --out /path/to/task_split.json
python scripts/inspect_temporal_samples.py \
  --out-dir /path/to/sample_cards \
  --report /path/to/temporal_audit.json
```

大型数据、模型权重、checkpoint、视频和本机验收结果不进入本仓库。可复现的代码、配置、测试和公开文档以本仓库为唯一真源。

