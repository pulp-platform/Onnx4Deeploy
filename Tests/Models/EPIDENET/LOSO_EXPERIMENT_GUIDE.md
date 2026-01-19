# Leave-One-Subject-Out (LOSO) Cross-Subject Fine-tuning 实验指南

## 📋 实验目标

评估 EpiDeNet 模型的**跨受试者泛化能力**和**个性化微调效果**。

---

## 🎯 实验设计

### 1. **实验流程**

```
对于每个受试者 i (i = 1, 2, ..., N):

  Phase 1: 预训练（Pre-training）
    ├─ 训练数据: 所有受试者数据 EXCEPT 受试者 i
    ├─ 训练轮数: 100-200 epochs
    ├─ 学习率: 0.001
    └─ 得到: 全局预训练模型

  Phase 2: 微调（Fine-tuning）
    ├─ 微调数据: 受试者 i 的部分数据 (50, 100, 200, 300, 400, 500 样本)
    ├─ 训练轮数: 30-50 epochs
    ├─ 学习率: 0.0001 (降低10倍)
    └─ 得到: 针对受试者 i 的个性化模型

  Phase 3: 评估（Evaluation）
    ├─ 测试数据: 受试者 i 的剩余数据
    └─ 指标: 准确率、F1-score
```

### 2. **对比实验**

| 实验组 | 描述 | 目的 |
|--------|------|------|
| **Baseline** | 只用全局模型，不微调 | 评估跨受试者泛化能力 |
| **Finetune-50** | 用 50 个样本微调 | 评估少样本学习能力 |
| **Finetune-100** | 用 100 个样本微调 | |
| **Finetune-200** | 用 200 个样本微调 | |
| **Finetune-300** | 用 300 个样本微调 | |
| **Finetune-500** | 用 500 个样本微调 | 评估充分数据下的效果 |

---

## 🚀 使用方法

### **快速开始**

```bash
# 运行完整 LOSO 实验
python loso_finetune_experiment.py \
    --data_path /path/to/EOG_data/csv/ \
    --output_dir loso_results \
    --pretrain_epochs 100 \
    --finetune_epochs 30 \
    --pretrain_lr 0.001 \
    --finetune_lr 0.0001 \
    --finetune_samples 50 100 200 300 400 500
```

### **自定义参数**

```bash
# 快速测试（减少 epochs）
python loso_finetune_experiment.py \
    --pretrain_epochs 50 \
    --finetune_epochs 20 \
    --finetune_samples 100 300 500

# 只测试几个样本数
python loso_finetune_experiment.py \
    --finetune_samples 100 200

# 更激进的微调学习率
python loso_finetune_experiment.py \
    --finetune_lr 0.0005
```

---

## 📊 实验输出

### **1. Results CSV**
```
loso_results/
└── loso_finetune_results.csv

Columns:
- subject: 受试者 ID
- finetune_samples: 微调样本数 (0 表示无微调)
- accuracy: 测试准确率 (%)
- phase: 'no_finetune' or 'finetune'
- improvement: 相比无微调的提升 (%)
```

### **2. Visualization**
```
loso_results/
└── loso_finetune_plot.png

图1: 每个受试者的准确率 vs 微调样本数
图2: 平均准确率提升 vs 微调样本数
```

---

## 🔬 关键参数说明

### **Pre-training 参数**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `pretrain_epochs` | 100-200 | 预训练轮数，数据多可以增加 |
| `pretrain_lr` | 0.001 | 标准的 Adam 学习率 |
| `batch_size` | 256 | 根据 GPU 内存调整 |

### **Fine-tuning 参数**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `finetune_epochs` | 30-50 | 微调轮数，样本少可以减少 |
| `finetune_lr` | 0.0001 | 通常是预训练的 1/10 |
| `finetune_samples` | [50, 100, 200, 300, 500] | 测试不同数据量的效果 |

### **微调策略**

```python
# 策略 1: 微调所有层（默认）
optimizer_ft = optim.Adam(model.parameters(), lr=finetune_lr)

# 策略 2: 只微调最后几层
for param in model.parameters():
    param.requires_grad = False
model.fcn.weight.requires_grad = True
model.fcn.bias.requires_grad = True
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=finetune_lr)

# 策略 3: 分层微调（不同层不同学习率）
optimizer_ft = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': finetune_lr * 0.1},
    {'params': model.conv2.parameters(), 'lr': finetune_lr * 0.1},
    {'params': model.conv3.parameters(), 'lr': finetune_lr * 0.5},
    {'params': model.fcn.parameters(), 'lr': finetune_lr}
])
```

---

## 📈 预期结果

### **典型场景**

1. **无微调 (Baseline)**
   - 准确率: 60-75%
   - 说明模型有一定的跨受试者泛化能力

2. **少样本微调 (50-100 samples)**
   - 提升: +5-10%
   - 显著改善，但可能不够稳定

3. **中等样本微调 (200-300 samples)**
   - 提升: +10-15%
   - 性能稳定，接近个性化最优

4. **充分样本微调 (500+ samples)**
   - 提升: +15-20%
   - 达到或接近针对该受试者的最优性能

### **分析指标**

```python
# 主要指标
1. Mean Accuracy across subjects (平均准确率)
2. Standard Deviation (标准差 - 评估稳定性)
3. Improvement over baseline (相比基线的提升)
4. Data efficiency (样本效率曲线)

# 额外分析
- 哪些受试者受益最多？
- 微调带来的提升是否稳定？
- 最小有效微调样本数是多少？
```

---

## 🎓 实验变体

### **变体 1: 不同微调策略对比**

```python
# 实验组:
1. Full fine-tuning (所有层)
2. Partial fine-tuning (只微调后几层)
3. Layer-wise fine-tuning (分层微调)
4. LoRA fine-tuning (低秩适应)
```

### **变体 2: 不同数据增强**

```python
# 在微调阶段添加数据增强:
1. Time shifting
2. Gaussian noise
3. Amplitude scaling
4. Time warping
```

### **变体 3: Few-shot Learning**

```python
# 极少样本场景:
finetune_samples = [5, 10, 20, 50]

# 可能需要:
- Meta-learning (MAML)
- Prototypical networks
- Metric learning
```

---

## 📝 实验报告模板

### **结果表格**

| Subject | No FT | FT-50 | FT-100 | FT-200 | FT-300 | FT-500 |
|---------|-------|-------|--------|--------|--------|--------|
| Sub01   | 65.2  | 70.1  | 73.5   | 76.8   | 78.2   | 80.1   |
| Sub02   | 68.5  | 73.2  | 75.9   | 78.5   | 80.1   | 81.7   |
| Sub03   | 62.1  | 67.8  | 71.2   | 74.5   | 76.3   | 78.9   |
| ...     | ...   | ...   | ...    | ...    | ...    | ...    |
| **Mean**| 66.3  | 71.5  | 74.8   | 77.9   | 79.5   | 81.2   |
| **Std** | 3.2   | 2.8   | 2.5    | 2.1    | 1.9    | 1.7    |

### **结论示例**

```markdown
## 实验结论

1. **跨受试者泛化**:
   - 无微调准确率: 66.3% ± 3.2%
   - 表明模型具有一定的跨受试者泛化能力

2. **微调效果**:
   - 50 样本: +5.2% (达到 71.5%)
   - 300 样本: +13.2% (达到 79.5%)
   - 500 样本: +14.9% (达到 81.2%)

3. **数据效率**:
   - 只需 100-200 个样本即可获得显著提升 (+8-11%)
   - 300 样本后收益递减

4. **建议**:
   - 实际应用中，推荐使用 200-300 个样本进行个性化微调
   - 可以在用户使用初期收集数据，在线微调模型
```

---

## 🛠️ 故障排查

### **问题 1: 微调后性能下降**
```
原因: 学习率过大，导致过拟合或灾难性遗忘
解决: 降低学习率 (0.00005) 或减少 epochs
```

### **问题 2: 微调提升不明显**
```
原因: 学习率过小，或训练轮数不够
解决: 适当增加学习率 (0.0002) 或增加 epochs
```

### **问题 3: 不同受试者差异大**
```
原因: 个体差异大，需要更多微调数据
解决: 增加微调样本数，或使用更强的数据增强
```

### **问题 4: GPU 内存不足**
```
解决: 减小 batch_size (128 或 64)
```

---

## 📚 参考文献

相关的 LOSO 和微调实验文献：

1. **EEGNet**: Lawhern et al., 2018
2. **Deep Transfer Learning**: Schirrmeister et al., 2017
3. **Subject Adaptation**: Dose et al., 2018
4. **Meta-Learning for BCI**: Banville et al., 2021

---

## ✅ 实验检查清单

- [ ] 确认数据路径正确
- [ ] 检查所有受试者数据完整
- [ ] 设置合理的 epochs 和学习率
- [ ] 预留足够的 GPU/CPU 时间
- [ ] 创建结果保存目录
- [ ] 准备实验记录表格
- [ ] 运行实验脚本
- [ ] 保存模型检查点
- [ ] 生成可视化图表
- [ ] 撰写实验报告

---

**祝实验顺利！** 🎉
