# PyTorch → 大模型训练师：理论结合实践学习计划

> 起点技能：Python、Linux 基础 ｜ 参考节奏：每周 10~12 小时，总周期约 15 周
> 核心原则：**每个理论点必须绑定一个代码任务，每个阶段必须有可验证的验收标准**。禁止"先刷完课再动手"。

## 参考时间线（2026年10月起，每周约12小时）

| 周次 | 阶段 |
|---|---|
| 10.05 ~ 10.11 | 阶段0：环境搭建 |
| 10.12 ~ 10.25 | 阶段1：PyTorch 核心机制 |
| 10.26 ~ 11.15 | 阶段2：训练工程与调参 |
| 11.16 ~ 12.13 | 阶段3：手写 Transformer 到 GPT |
| 12.14 ~ 2027.01.10 | 阶段4：LLM 微调实战 |
| 2027.01.11 ~ 01.31 | 阶段5：对齐与部署 |
| 2027.02 起 | 求职冲刺（作品集 + 面试） |

---

## 阶段0：环境搭建（1周内完成）

- [ ] 安装 WSL2 (Ubuntu)——用你的 Linux 基础，和未来生产环境保持一致
- [ ] 安装 miniconda，创建名为 `torchlearn` 的独立 conda 环境（`torchlearn` 只是环境名，不是软件包），并在环境内装 CPU 版 PyTorch（前期够用）
- [ ] 注册 Weights & Biases（免费），跑通第一段 wandb.log() 记录
- [ ] 建 GitHub 仓库 `llm-trainer-journey`，所有阶段代码进这一个仓库
- [ ] （可选）注册 AutoDL，熟悉租卡流程（阶段4才用得上，每小时1~2元）

---

## 阶段1：PyTorch 核心机制（2周）

> 📋 14天逐日详细版见 [pytorch-stage1-detail.md](pytorch-stage1-detail.md)

**理论点**：张量与广播、autograd 计算图与反向传播、nn.Module、Dataset/DataLoader、device 与 no_grad

**配套学习材料**
- [ ] Karpathy《The spelled-out intro to neural networks and backpropagation》（micrograd 视频）——纯 Python 手写反向传播，理论结合实践的巅峰
- [ ] 李沐《动手学深度学习》PyTorch 版 第 2~4 章（含 B 站配套视频）
- [ ] PyTorch 官方 "60 Minute Blitz"

**动手任务（按顺序做）**
- [ ] 只用 `tensor` + `autograd`，不用 nn、不用 optimizer，手写线性回归；手动推导梯度并和 `backward()` 结果对比
- [ ] 用 `nn.Module` + `optim.SGD` 重写，拟合正弦函数
- [ ] 自定义 `Dataset` + `DataLoader` 读一个 CSV 做二分类
- [ ] MNIST 上训练 MLP，达到 ≥97% 测试准确率

**验收标准**：不查资料默写训练循环五件套（数据加载→前向→算loss→backward→step）；说清 `loss.backward()` 时计算图里发生了什么。

---

## 阶段2：训练工程与调参（3周）

**理论点**：优化器家族（SGD动量/Adam/AdamW 的区别）、学习率调度（warmup、cosine）、正则化（dropout、weight decay）、BatchNorm vs LayerNorm、混合精度、梯度裁剪

**动手任务**
- [ ] CIFAR-10 上训练一个 CNN 基线（自己搭，不用 ResNet 预训练）
- [ ] 设计实验矩阵：至少对比 3 种学习率 × 2 种优化器，全部记录到 W&B
- [ ] 给基线加上：数据增强、lr scheduler、梯度裁剪、混合精度，逐一记录对收敛的影响
- [ ] 故意制造过拟合（小数据集+大模型），练习诊断与修复

**验收标准**：产出一份调参实验记录表（配置+曲线+结论）；能脱口而出"训练不收敛的 10 条排查清单"。

---

## 阶段3：手写 Transformer 到 GPT（4周，最重要）

**理论点**：self-attention 数学（能手推 Q/K/V）、multi-head、位置编码、LayerNorm 在 LLM 里为何用 Pre-Norm、BPE tokenizer、采样策略（temperature/top-p）

**配套学习材料**
- [ ] 精读《Attention is All You Need》
- [ ] Karpathy《Let's build GPT from scratch》视频 + nanoGPT 仓库
- [ ] Karpathy《Let's build the GPT Tokenizer》（BPE）

**动手任务**
- [ ] 跟做视频后，**合上视频从零默写**：MultiHeadAttention → TransformerBlock → 完整小 GPT
- [ ] 在莎士比亚字符级语料上训练，采样出通顺文本
- [ ] 换成中文小语料重复一遍（体验 tokenizer 的差异）
- [ ] 用 Hugging Face Transformers 微调 BERT 做文本分类——体会"从零实现"和"生态工具"两种视角

**验收标准**：白板手写 scaled dot-product attention；能逐个解释 `d_model、n_head、context_length、dropout` 对模型的影响；自己的 GPT 能生成像样的文本。

---

## 阶段4：LLM 微调实战（4周，就业核心）

**理论点**：预训练→SFT→对齐三阶段范式；LoRA 数学（为什么低秩矩阵有效）；chat template 与 loss mask（只对回答部分计算损失）；量化原理（NF4/GGUF）

**配套学习材料**
- [ ] 精读 LoRA、QLoRA、InstructGPT 三篇论文
- [ ] Hugging Face PEFT + TRL 官方文档

**动手任务**
- [ ] **数据工程**：下载开源中文指令数据集（BELLE / Alpaca-GPT4-zh），写清洗脚本、构建 chat template 样本、验证 loss mask 正确性
- [ ] 用 `Trainer` + `PEFT` 对 Qwen3 0.6B（或 Qwen2.5 0.5B）做 LoRA SFT——不用 LLaMA-Factory，先手写完整流程
- [ ] 用 LLaMA-Factory 复现同一个微调，对比效果，读懂它每个配置项对应你手写的哪段代码
- [ ] 评估：记录 loss 曲线、训练前后生成样例对比（各 20 条）、写一页评估结论

**验收标准**：GitHub 完整项目（数据处理+训练+评估+README），这是求职作品集的第一个核心项目。

---

## 阶段5：对齐与部署（3周）

**理论点**：RLHF 三件套（RM/PPO 概念即可）；DPO 推导（理解它和 RLHF 的关系）；GRPO 思想（DeepSeek-R1 路线）

**动手任务**
- [ ] 精读 DPO 论文
- [ ] 用 TRL `DPOTrainer` 在自己阶段4的模型上做小规模偏好对齐（开源偏好数据集起步）
- [ ] 用 vLLM 部署你的模型，暴露 OpenAI 兼容 API
- [ ] Gradio 搭一个在线 demo，放 DPO 前后对比

**验收标准**：一条可访问的 demo 链接 + 一份《DPO 对齐实验报告》。作品集齐活。

---

## 贯穿全程的习惯

1. **实验记录**：从阶段2开始所有实验进 W&B，截图存进仓库 `experiments/` 目录
2. **git**：每个任务一个 commit，commit message 写清楚做了什么实验
3. **输出倒逼输入**：每阶段结束写一篇博客/README，面试官看的就是这个
4. **每周复盘**：周日 30 分钟，对照本清单勾进度，落后的砍理论不砍动手

## 避坑清单

- ❌ 数学完美主义：数学补到"看懂公式、能推导 attention"即可，边写代码边补
- ❌ 一上来玩 7B/14B：先 0.5B 跑通全流程，规模是最后才加的东西
- ❌ 只跟教程抄代码：每个阶段都有"合上资料默写"环节，抄会和写差一个offer
- ❌ 考证培训班：企业看作品集和微调实战，不看证书
