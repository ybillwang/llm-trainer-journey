# 阶段1详细版：PyTorch 核心机制（14天逐日计划）

> **每天时长**：工作日约 1~1.5 小时，周末约 2.5~3 小时（周合计约 11 小时）
> **规则**：先看材料（≤当天时长的一半），剩余时间全部动手。看懂 ≠ 会写，以任务完成为准。

## 总览

| Day | 主题 | 时长 | 交付物 |
|---|---|---|---|
| 1 | 张量基础 | 1h | `day01_tensor_basics.ipynb` |
| 2 | 广播机制专攻 | 1h | `day02_broadcast.py` |
| 3 | autograd 计算图 | 2.5h | `day03_autograd.py` |
| 4 | micrograd 上半场 | 2.5h | `day04_micrograd.ipynb` |
| 5 | micrograd 下半场 | 2.5h | `day05_micrograd.ipynb` |
| 6 | 核心①：纯 tensor 线性回归 | 1h | `day06_linreg_scratch.py` |
| 7 | 核心②：nn.Module 重写 | 1h | `day07_nn_rewrite.py` |
| 8 | 训练循环标准化 | 1h | `day08_train_loop.py` |
| 9 | 核心③：Dataset/DataLoader | 1.5h | `day09_dataset.py` |
| 10 | 核心④：MNIST MLP 冲刺 | 2.5h | `day10_mnist.py` |
| 11 | MNIST 打磨 + 调试工具箱 | 2.5h | `day11_regularize_debug.py` |
| 12 | device 无关 + checkpoint | 1h | `day12_device_ckpt.py` |
| 13 | 验收①：闭卷默写 | 1.5h | `day13_closed_book.py` + `day13_diff_notes.md` |
| 14 | 验收②：口述 + 复盘 | 1h | `stage1_summary.md` |

## 环境速查（WSL Ubuntu + conda 环境 `torchlearn`）

- **每天开工第一步**：`conda activate torchlearn`
- **项目目录**：放 WSL 用户目录（如 `~/torchlearn/`），**别放 `/mnt/c/`**——跨盘 IO 慢一个量级；Windows 侧翻文件走 `\\wsl.localhost\Ubuntu\home\<用户名>\...`
- **W&B**（Day 6 前完成）：`wandb login`，API key 在 [wandb.ai/authorize](https://wandb.ai/authorize) 复制，登录态存 `~/.netrc`
- **一次性安装**：
  - Day 4 前：`pip install graphviz` + `sudo apt install graphviz`（两个都要，前者是 Python 接口，后者才是 `dot` 程序本体）
  - Day 5 前：`pip install scikit-learn matplotlib`
- **GPU（可选）**：WSL2 的 CUDA 穿透 Windows 侧 NVIDIA 驱动，Linux 侧**不要**装驱动。`python -c "import torch; print(torch.cuda.is_available())"` 为 False 也完全不影响，本阶段 CPU 够用

## 本阶段四份核心材料

| 材料 | 用法 |
|---|---|
| 李沐《动手学深度学习》PyTorch 版（[zh.d2l.ai](https://zh.d2l.ai)，免费在线+B站视频） | 理论主线，按天下发的章节看 |
| Karpathy《Building micrograd》视频（约2.5小时） | Day 4~5 集中攻，理解反向传播的机械原理 |
| PyTorch 官方《60 Minute Blitz》 | 快速过，遇到 API 细节回来查 |
| W&B 账号（阶段0已建） | 从 Day 6 起所有训练任务都记录 |

---

## 第一周

### Day 1（约1h）张量基础

- **看**：李沐 2.1 数据操作 + 2.2 数据预处理
- **做**：把 2.1 里每个张量操作自己敲一遍并各改一个参数观察变化（reshape / 索引切片 / 拼接 / 逐元素运算）
- **交付**：
  - [ ] `day01_tensor_basics.ipynb`

---

### Day 2（约1h）广播机制专攻

- **看**：李沐 **2.1.3「广播机制」**（规则在这节，Day 1 扫过今天精读）+ **2.3.6「降维」/2.3.6.1「非降维求和」**（任务B的API依据。书里用 `axis=` 和 `keepdims=`，PyTorch 惯例是 `dim=` 和 `keepdim=`——已在 torch 2.14 实测四种拼写全部可用，书上代码原样能跑，不必当坑背）；权威补充：[PyTorch 官方广播文档](https://pytorch.org/docs/stable/notes/broadcasting.html)
- **做**：先笔算预测结果 shape 再运行验证：`(3,1)+(1,4)`、`(4,3)+(3,)`、`(2,1,4)+(3,1)`；再用广播实现"每行减去该行均值"，禁止写 for 循环
- **交付**（`day02_broadcast.py`）：
  - [ ] 6 组形状预测（预测写在注释里，运行验证）
  - [ ] 无循环版"行减行均值"，并 assert 对比双循环版
  - [ ] ≤10 行的广播规则自我总结
- **为什么重要**：attention mask、归一化在 LLM 里全靠广播，这是后面阶段的隐性地基

---

### Day 3（约2.5h，周末）autograd 计算图

- **看**：李沐 **2.5「自动微分」全节**（2.5.1 标量 backward 基本流程 / 2.5.2 非标量为什么必须先 `.sum()` / 2.5.3 `detach()` 分离计算 / 2.5.4 Python 控制流也能求梯度——这就是"动态图"的含义）；Blitz 的 Autograd 小节当 10 分钟快读；权威补充：[PyTorch autograd 官方文档](https://pytorch.org/docs/stable/autograd.html)
  - 读法提示①：书上 `x.requires_grad_(True)` 和构造函数传 `requires_grad=True` 等价（书里有注释），两种都合法
  - 读法提示②：书里多处出现 `x.grad.zero_()` 不是仪式感——PyTorch 的梯度默认**累积**，不清零的话第二次 backward 后梯度翻倍（任务C现场踩）
- **做**（以下数值均已在 torch 2.14.1 实测，写完可直接对答案）：
  - **任务A**：求 `y = x²·sin(x)` 在 x=π/2 处的导数，与解析解 `2x·sin(x)+x²·cos(x)` 对比。该点 sin(π/2)=1、cos(π/2)=0，所以 autograd 和解析解都应恰好等于 **π≈3.1415927**，一眼验算，最后用 `assert torch.allclose` 收尾
  - **任务B**：`f = (a+b)·c`（取 a=2、b=3、c=4）。先在纸上画计算图（a,b→s=a+b→f=s·c），逐节点手推：∂f/∂s=c、∂s/∂a=1 ⇒ ∂f/∂a=c，同理 ∂f/∂b=c、∂f/∂c=a+b（链式法则就是"反向走图、逐段乘局部梯度"）；再用 autograd 验证（正确答案：f=20，梯度 4/4/5）。这个形状就是 Day 4 micrograd 的开场示例——今天手推的每一步，后天都要亲手写成代码
  - **任务C（坑①·梯度累积）**：对同一个 x 连续 backward 两次不清零，打印 grad 观察 6 → 12 翻倍，再用 `x.grad.zero_()` 修复（末尾下划线=原地操作）。这是 Day 6 手动更新前必须清零 `w.grad` 的原因，也是五件套里 `optimizer.zero_grad()` 存在的原因
  - **任务D**：跑书上 2.5.3 的 detach 例子并解释为什么 `x.grad == u`（u 已脱离计算图变成常数，梯度走不到 y 那条路）；顺带想想：阶段4 的 loss mask「只对回答部分计算损失」，本质就是这套"把张量当常数"的机制
  - **任务E（2.5.4·动态图，约10分钟）**：跑书上的控制流函数 `f(a)`，取 a=1.0 / -1.0 / 0.001 各验证一次 `a.grad == d/a`（实测 k 分别为 1024 / 102400 / 1048576，把"k 怎么来的"手推进注释：循环跑完后图里只剩一条纯乘法链，f(a)=k·a，导数即 k=d/a）。关键领悟：**循环和 if 不进计算图**——图是"实际执行了什么"的记录，同一个函数换个输入会走出深度不同的图，这就是"动态图"三个字的含义
  - **坑②（读报错）**：对向量直接 `y.backward()` 会报 `RuntimeError: grad can be implicitly created only for scalar outputs`——把报错完整读一遍，它告诉你两条出路：`y.sum().backward()` 或 `y.backward(gradient=...)`。后者的 gradient 向量 =「给 y 每个分量预填的起始梯度」（Day 4 micrograd 里 `self.grad = 1.0` 的向量版），传全 1 就等价于 `sum()`——想通这一点，backward 的合同就通了
- **搞懂五个词**：`requires_grad` / `backward()` / `.grad`（调用前是 None，且只保留在**叶子**张量上）/ `detach()` / `zero_()`（书上用了但没强调，新手第一大 bug 源）。（`no_grad()` 2.5 原文没讲，它首次出现在 3.1 的 sgd 函数里，Day 6 会见到）
- **交付**（`day03_autograd.py`）：
  - [ ] 任务A + assert
  - [ ] 任务B：计算图和手推梯度值写进注释，assert 验证
  - [ ] 坑①：打印两次 backward 的 grad 展示累积并修复
  - [ ] 任务D：detach 前后 `x.grad` 对比
  - [ ] 任务E：三个 a 值的 k 手推 + `a.grad == d/a` 验证
  - [ ] ≤10 行用自己的话总结「`backward()` 那一刻计算图里发生了什么」（从标量出发反向走图、逐节点乘局部梯度——这就是 Day 14 口述第一问）
- **为什么重要**：`loss.backward()` 是训练五件套的第四步、整个深度学习的引擎；"忘了 zero_grad 导致 loss 怎么都不降"是接下来两周最高频的 bug。今天在三个标量上搞懂它，Day 4~5 在 micrograd 里亲手把它造一遍

---

### Day 4（约2.5h，周末）micrograd 上半场

- **开场准备**（10分钟，实测你的环境缺这两样）：`pip install graphviz` + `sudo apt install graphviz`——缺一个画图就报错；不装也能完成主体任务，只是画不了图。视频看 [YouTube 原版](https://www.youtube.com/watch?v=VMj-3S1tku0)（英文吃力就开自动中文字幕，B站有搬运但时间点会偏移，按下面段落主题对齐）；讲义对照：[nn-zero-to-hero/lectures/micrograd](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd) 里有 `micrograd_lecture_first_half_roughly.ipynb`（视频原稿上半场）——**当答案看，自己新建空白 notebook 跟敲，禁止复制粘贴**
- **看+做**（同一件事，边看边敲）：视频 **0:00 → 1:27:05**，共 87 分钟，1.25 倍速约 70 分钟，加暂停预算约 100 分钟：

  | 段落 | 时间点 | 节奏与任务 |
  |---|---|---|
  | 导数直觉（单输入/多输入） | 0:00–19:09 | 可 1.5 倍速——Day 3 已懂。唯一重点：`lol()` 数值梯度检查套路，后面反复用 |
  | Value 类雏形 + draw_dot 可视化 | 19:09–32:10 | 跟敲：`data`/`grad`/`_prev`/`_op`/`label` 五件套，跑通 draw_dot 画出表达式 |
  | 手动反向传播 #1（L=d·f） | 32:10–51:10 | **全片心脏，1 倍速**。他在纸上/代码里填每个 grad 之前暂停，先自己推——这就是 Day 3 任务B 的加长版 |
  | 单步优化预览 | 51:10–52:52 | 注意更新方向为什么带负号——Day 6 手写 SGD 的雏形 |
  | 手动反向传播 #2（tanh 神经元） | 52:52–1:09:02 | 跟敲 `tanh`；他问 dO/dN 时暂停，自己从 tanh 公式推出 `1−o²` 再看答案 |
  | 每个运算的自动 backward | 1:09:02–1:17:32 | 跟敲 `__add__`/`__mul__`/`tanh` 里的 `_backward` 闭包，每个方法写一行注释说明"这一步在计算图上做了什么" |
  | 整图 backward（拓扑排序） | 1:17:32–1:22:28 | 跟敲 `build_topo` + `backward()`；听讲前先想清楚为什么不能按创建顺序调 `_backward` |
  | 「一个节点用两次」的 bug | 1:22:28–1:27:05 | 他演示 `a+a` 梯度错成 1 时**先暂停**——答案就是你 Day 3 坑① 见过的累积机制，预测对再看解释 |

- **交付**（`day04_micrograd.ipynb`）：
  - [ ] 自己的 Value 类（五件套 + 三个运算的 `_backward` 闭包 + `backward()`），每方法一行图注
  - [ ] 手动反向传播 #1 复现 + `lol()` 数值梯度检查 assert
  - [ ] `a+a` bug 演示：`=` 版输出 **1.0**、`+=` 版输出 **2.0**（已在 torchlearn 环境实测）
  - [ ] 拓扑排序 `backward()` 跑通后与手填的 grad assert 对比
  - [ ] 文末一句话：「`backward()` = 拓扑排序后逆序调用每个节点的 `_backward`，把 `out.grad` × 局部梯度 `+=` 进输入的 `grad`」
- **串接**：Day 3 的 ASCII 计算图 ↔ `draw_dot` 画出的图（`_prev` 就是邻接表）；Day 3 坑① 梯度累积 ↔ 1:22:28 的 `+=` bug（同一机制：一处是用户忘清零，一处是作者忘累加）；`lol()` 数值检查 ↔ 你 `allclose` 习惯的专业版（这也正是你跳过的李沐 2.4 极限定义第一次派上用场）；Day 3 的纯乘法链 ↔ Value 链表，backward 就是逆着链表乘局部梯度
- **时间不够时**：砍到 1:09:02（手动部分全部完成即达标），自动 backward + bug 修复顺延到 Day 5 开头补完——理解优先于自动化

---

### Day 5（约2.5h，周末）micrograd 下半场

> 从视频 **1:27:05「拆 tanh」** 接续看起；若 Day 4 顺延了自动 backward 部分，先补完再开始。

- **环境**（2分钟）：今天要用 sklearn 的月牙数据集，`pip install scikit-learn matplotlib`
- **看+做**：视频 **1:27:05 → 2:25:33**，约 58 分钟，1.25 倍速约 47 分钟，加暂停跟敲预算约 90 分钟：

  | 段落 | 时间点 | 节奏与任务 |
  |---|---|---|
  | 拆 tanh + 补全运算符 | 1:27:05–1:38:30 | 跟敲 `exp` / `/`（实现成 `**-1`）/ `**` / 取负 / `-` 的 `_backward` 闭包。**关键领悟：运算的粒度是任意的**——tanh 可以是一个原子 op 也可以拆成 exp/div 的组合，只要每个 op 知道局部梯度，链式法则会把账算平。这就是 PyTorch 里几千个原子 op 的设计哲学 |
  | PyTorch 对照实验 | 1:38:30–1:46:38 | 跟敲 torch 版同一段计算，梯度逐位对比相等。注意他特意转 double——float32 精度下小数点后第 6~7 位就会分叉，这是数值精度的第一次体感（阶段 4 混合精度训练时你会想起这段） |
  | Neuron / Layer / MLP | 1:46:38–1:59:06 | 跟敲三个类。盯紧 `parameters()` 方法——它就是 Day 7 `net.parameters()` 的原型，optimizer 和模型之间的唯一接口今天第一次现身 |
  | 损失函数 | 1:59:06–2:11:15 | make_moons 二分类：max-margin loss + L2 正则（`alpha * sum(p*p for p in model.parameters())`）。L2 这一项就是 PyTorch `weight_decay` 参数的手工版，也是 Day 11"旋钮"家族的另一员 |
  | 手动训练循环 | 2:11:15–2:25:33 | **全片闭环时刻**：`forward → loss.backward() → 手动更新 → 清零` 跑 100 轮，lr 线性衰减（1.0 → 0.1）。跑通后 accuracy 应到 **95%+**。看清一件事：这五行的结构和 Day 6~8 你要写的训练五件套**一模一样**，只是数据从月牙换成图片、模型从 Value-MLP 换成 nn.Module |

- **练习**（约 60 分钟，视频简介的 Colab 练习本，或 nn-zero-to-hero 仓库 [lectures/micrograd](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd) 的讲义 notebook）：共三道，难度递增——
  1. 单变量函数的解析导数（热身，Day 3 任务A 的水平）
  2. 含除法/指数的多变量函数**手推全梯度**再对 autograd 答案（这是重点，把 Day 3 任务B 的手推能力升级到有除法和 exp 的图上）
  3. 小型 MLP 指定权重的梯度手推——**先纸笔推再用代码验证，不许一上来就敲**
  - 做不完至少完成 ②
- **交付**（`day05_micrograd.ipynb`，接着 Day 4 的 notebook 写）：
  - [ ] 补全后的 Value 类（五个新 op 各一行图注）
  - [ ] torch 对照实验 cell：两边梯度 assert 对比
  - [ ] Neuron/Layer/MLP + 月牙训练循环跑通（accuracy ≥95%）
  - [ ] 练习 ② 的手推过程（写成注释或拍照贴图）+ autograd 验证
  - [ ] 文末一句话：「我的 Value、torch.Tensor、nn.Linear 是同一件事的三个抽象层级——明天开始穿正式外套」
- **产出**：一个完全属于你的微型自动微分引擎 + 一个手动训练出的 MLP。这就是后面读 PyTorch 源码不慌的底气
- **串接**：MLP 的 `parameters()` → Day 7 `optim.SGD(net.parameters(), ...)`；L2 正则项 → PyTorch 的 `weight_decay`；手动五件套循环 → Day 6 裸写版、Day 8 函数化版。今天跑通的训练循环，本质上你已经把阶段 1 剩下的路预演了一遍
- **时间不够时**：PyTorch 对照段（1:38:30–1:46:38）只看不动手；练习只做 ②。**Neuron/Layer/MLP 和训练循环不能砍**——没有这两段，micrograd 就只是一堆数学练习，没有变成"引擎"

---

### Day 6（约1h）核心任务①：纯 tensor 线性回归

- **环境**（一次性，5分钟）：首次用 wandb 先 `wandb login` 粘贴 API key（见「环境速查」）
- **对照看**：李沐 3.1 线性回归从零实现（书上同样用 autograd 求梯度，但手写 SGD 更新步、不用 optim——你要做的正是它的最小版，nn+optim 到 3.2 才引入）
  - 读法提示①：书上 `synthetic_data` 的真值是 w=[2, -3.4]、b=4.2，你可以简化成单变量 `y = 3.7·x − 1.2 + ε`（ε~N(0, 0.1²)，500 个样本），对答案更直观
  - 读法提示②：书上 `sgd` 函数包着 `with torch.no_grad():`——参数更新绝不能进计算图，否则下一次 backward 会把"更新"本身也微分进去
  - 读法提示③：书上 loss 是求和、更新时除以 batch_size，你用 `.mean()` 就不用除
- **做**：
  - **任务A（25分钟）**：`torch.randn(500, 1)` 造 X 算 y 加噪声；初始化 `w = torch.randn(1, 1, requires_grad=True)`、`b = torch.zeros(1, requires_grad=True)`；训练循环四步：`y_hat = X @ w + b` → `loss = ((y_hat - y) ** 2).mean()` → `loss.backward()` → `with torch.no_grad(): w -= lr * w.grad; b -= lr * b.grad` 然后 `w.grad.zero_(); b.grad.zero_()`。lr=0.03，跑 10 个 epoch
  - **任务B（10分钟）**：每 epoch `wandb.log({"loss": loss.item()})`（开头 `wandb.init(project="torchlearn-stage1", name="day06-linreg-scratch")`），跑完去网页看曲线是否光滑下降
  - **任务C（对答案）**：打印最终 w、b 与真值。正常水平：10 epoch 后 w 误差 <0.01、b 误差 <0.05 量级。loss 不降先查清零没有；曲线锯齿太大就把 lr 砍半看变化
- **交付**（`day06_linreg_scratch.py`）：
  - [ ] 造数据 + 手动训练循环，循环里每行一句注释说明"这行在计算图内还是图外"
  - [ ] wandb run 链接或曲线截图
  - [ ] 注释回答两问：更新为什么包 no_grad / backward 后为什么必须清零（一句话版：Day 3 坑①）
- **为什么重要**：训练五件套的第一次完整组装——数据、模型（这里退化成两个参数）、loss、backward、更新，今天全裸写一遍，Day 7 看 nn/optim 的封装才知道每件衣服底下是什么
- **串接**：Day 3 任务C 梯度累积 → 循环里每次 `zero_()` 的原因；micrograd 视频 51:10 单步优化预览（更新为什么带负号）→ 今天 `w -= lr * w.grad` 的完整版

---

### Day 7（约1h）核心任务②：nn.Module 重写

- **看**：李沐 3.2 线性回归的简洁实现；Blitz 的 Neural Networks 小节（10分钟快读，重点看 `nn.Module` 的 `forward` 约定和 `net.parameters()` 这个迭代器）
  - 读法提示：书上 `nn.Linear(2, 1)` 的 2 是特征数，你的单变量版就是 `nn.Linear(1, 1)`；训练完打印 `net[0].weight` / `net[0].bias`——和 Day 6 手动更新的 w、b 是同一个东西穿了件外套
- **做**：
  - **任务A（15分钟）**：用 `nn.Linear` + `nn.MSELoss()` + `optim.SGD(net.parameters(), lr=0.03)` 重写 Day 6，assert 学到的参数同样逼近真值。体会封装映射：`optimizer.zero_grad()` ↔ 你手写的 `w.grad.zero_()`，`optimizer.step()` ↔ 你手写的 `-=` 更新——五件套骨架一行没变，只是三件穿上了标准外套
  - **任务B（40分钟，今天的主菜）**：拟合 `y = sin(x)`。`X = torch.linspace(-math.pi, math.pi, 1000).reshape(-1, 1)`，跑两条 wandb run：
    - run1 直线版：`nn.Linear(1, 1)`，500 epoch——预期**失败**：loss 降到 ~0.19 就躺平（理论最优直线 y≈0.304·x，残差就这量级，一条直线再怎么优化也弯不成波浪）
    - run2 MLP 版：`nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 1))`，Adam lr=0.01、2000 epoch（CPU 秒级跑完）——loss 应降到 **1e-3 量级以下**。打印采样点上预测 vs 真值对照，肉眼看贴合度
  - **思考题**（答案写进注释）：① 线性模型为什么拟合不了 sin？（容量不足=欠拟合的最小样本：假设空间里只有直线，目标不在里面）② 加隐藏层后发生了什么？（Tanh 提供非线性"折弯"，64 个折弯的线性组合逼近任意曲线——通用逼近定理的体感版）③ 回归任务输出层为什么不加激活？（输出要覆盖整个实数域，Tanh/Sigmoid 会把输出压扁到有限区间）
- **交付**（`day07_nn_rewrite.py`）：
  - [ ] nn 版线性回归 + assert
  - [ ] sin 拟合双 run（失败版+成功版对比）
  - [ ] 思考题注释
  - [ ] 一句话总结：「nn.Module 打包参数和 forward，optim 打包更新规则，训练循环不变」
- **为什么重要**：从今天起一切模型都是 nn.Module——阶段 3 手写 GPT 的每个 Block 也是同样的 `__init__` + `forward` 结构。`net.parameters()` 是 optimizer 与模型之间的唯一接口，搞懂它就看懂了 PyTorch 的组装方式

---

## 第二周

### Day 8（约1h）训练循环标准化

- **做**：把 Day 7 代码重构成三个函数，以后每天复用：

  ```python
  def train_one_epoch(model, loader, loss_fn, optimizer):
      model.train()
      # 五件套循环：forward → loss → zero_grad → backward → step

  def evaluate(model, loader, loss_fn):
      model.eval()
      with torch.no_grad():
          # 只做前向，累加 loss/正确数

  # 外层 train_loop 逐 epoch 调这两个 + wandb.log
  ```

- **任务A（30分钟）**：用重构后的 train_loop 重跑 Day 7 的 sin 拟合，loss 曲线应与原版重合；再手动切 train/val（前 800 点训练、后 200 点验证），两条 loss 曲线都记进 wandb
- **任务B（15分钟）**：思考题写进注释——① 为什么评估时不需要计算图？（不建图 = 不存中间激活、不记录操作，省内存 + 省掉 backward 的簿记开销；评估只要预测值不要梯度）② `model.eval()` 和 `no_grad()` 是一回事吗？（**不是**。eval() 切换 Dropout/BatchNorm 的推理行为，no_grad() 关梯度记录——两个都要写，各管各的。Day 11 加 dropout 后漏写 eval()，val 指标会诡异波动，到时回来对这句话）
- **交付**（`day08_train_loop.py`）：
  - [ ] 三函数结构 + train/val 双曲线 wandb run
  - [ ] 思考题注释
  - [ ] 文件头注释写五件套口诀：`model / loader / loss_fn / optimizer / train loop`
- **为什么重要**：函数化之后，后面每天的任务都是"换模型换数据，循环不动"。这套结构会一路用到阶段 4 的 SFT——只是塞进去的东西越来越贵
- **串接**：Day 6 的手动三行（backward / no_grad 更新 / `zero_`）被收进 train_one_epoch 尾部；Day 3 的 detach → `no_grad` 是它的"整个代码块版"

---

### Day 9（约1.5h）核心任务③：自定义 Dataset/DataLoader

- **做**：
  - **任务A（数据管线，40分钟）**：`sklearn.datasets.load_breast_cancer()`（569 样本 × 30 特征，二分类）。**顺序是今天的考点**：① 先 `train_test_split(test_size=0.2, random_state=42, stratify=y)` ② `StandardScaler().fit(X_train)`——**只 fit 训练集** ③ 用同一个 scaler 分别 transform 训练集和验证集。然后写自定义 `Dataset`（`__init__` 存 X,y / `__len__` / `__getitem__`），`DataLoader(train_ds, batch_size=32, shuffle=True)`，**val loader 不 shuffle**
  - **坑①（类型，几乎必踩）**：`torch.from_numpy(X)` 是 float64，模型权重是 float32，前向一乘就报 `expected scalar type Double but found Float`——`.float()` 转掉。这条报错值得完整读一遍，它把 PyTorch 的类型系统讲得比文档清楚
  - **坑②（形状）**：`BCEWithLogitsLoss` 要求输出和标签形状严格一致——模型输出 (B,1)、标签 (B,) 会直接报 target size mismatch，`y.view(-1, 1)` 对齐。和 Day 2 广播对照着看：广播是"形状不同但兼容"，这里是"不兼容就报错"，边界在哪要心中有数
  - **任务B（模型，30分钟）**：MLP `30 → 16 → 1`，`BCEWithLogitsLoss`（输出层**不加 sigmoid**——和 Day 10 的 CE 不加 softmax 同一设计哲学：log 域的数值稳定计算都收进 loss 里了）。Adam lr=1e-3，30 epoch，val accuracy 正常 **97%+**（数据集很干净，到不了先查标准化顺序）
  - **任务C（作弊对照组，10分钟）**：故意犯一次数据泄漏——先全量标准化再划分，对比两次 val accuracy。泄漏版会虚高一点。亲手跑一次比背十次"不能泄漏"管用
- **交付**（`day09_dataset.py`）：
  - [ ] 正确顺序管线 + 自定义 Dataset
  - [ ] float64 / 形状两坑的报错与修复注释
  - [ ] 泄漏对照实验结果
  - [ ] ≤5 行总结「为什么标准化只能用训练集统计量」（验证集扮演"未来数据"，它的均值方差在训练时不该存在——用了就是偷看答案，模型在新数据上的表现会被高估。阶段 4 做训练/评测切分同理）
- **为什么重要**：真实项目里大部分时间花在数据管线而不是模型。Dataset/DataLoader 这套接口到阶段 4 处理 tokenized 文本时原样复用，只是 `__getitem__` 返回的从特征向量变成 token 序列

---

### Day 10（约2.5h，周末）核心任务④：MNIST MLP 冲刺

- **对照看**：李沐 4.1~4.3 多层感知机（4.1 看"为什么需要隐藏层和非线性激活"——Day 7 的 sin 实验就是答案；4.2 从零实现扫结构即可，你 Day 6 已会手动版；4.3 简洁实现和你现在的写法几乎一样）
- **环境提示**：首次运行 `torchvision.datasets.MNIST(download=True)` 会下载约 60MB 到 `./data/`（项目目录位置见「环境速查」）
- **做**：
  - **任务A（baseline，40分钟）**：`transform=transforms.ToTensor()`（隐式干了两件事：PIL→张量、0-255 缩放到 [0,1]——记住这个隐式行为，哪天自己写 transform 就知道要不要手动 /255）；`nn.Flatten()` 拍平成 784；网络 `784 → 128 → 64 → 10` 全 ReLU；`CrossEntropyLoss` + `Adam(lr=1e-3)`；5 epoch，用 Day 8 的 train_loop 原样跑。预期：test accuracy **97.5% 上下**（≥97% 达标）
  - **陷阱①（标签格式，跑一次）**：标签保持 0-9 的类别索引（long），**不要 one-hot**——CE 直接吃索引。注意新版 PyTorch 的 CE 也接受概率标签，one-hot 喂进去不报错但语义全错，是静默 bug；好习惯是喂之前 `assert y.dtype == torch.long`
  - **陷阱②（双重 softmax，跑一次）**：模型最后一层加 `nn.Softmax(dim=1)` 再喂 CE——能跑、loss 也降，但收敛明显变慢。它不报错，是最阴的新手 bug：CE = LogSoftmax + NLLLoss 已内置 softmax，模型输出要保持 logits
  - **体感练习（两条 wandb run）**：lr=10 → loss 爆成 nan、accuracy 锁死 ~10%（随机猜的水平）；lr=1e-6 → loss 几乎纹丝不动。加上 baseline 三条曲线放一张图——这就是 lr 搜索的直觉：先差 100 倍看方向，再二分逼近
- **交付**（`day10_mnist.py`）：
  - [ ] baseline ≥97%（wandb run）
  - [ ] 两个陷阱各一条 run + 现象注释
  - [ ] lr 发散/停滞对照 run
  - [ ] 注释回答：CE 的输入是 logits 还是概率？为什么？（logits。内部 log_softmax 用 log-sum-exp 技巧避免上溢，自己先 softmax 反而丢掉这层保护还压缩梯度）
- **为什么重要**：MNIST 是行业 hello world，也是 Day 13 闭卷默写的素材。今天把 Dataset + 五件套 + wandb 全串通一遍，Day 13 就是开卷变闭卷的 replay
- **时间不够**：陷阱②可只跑一次不记 run，lr 对照必须做

---

### Day 11（约2.5h，周末）MNIST 打磨 + 调试工具箱

- **做**：
  - **任务A（dropout，40分钟）**：每个隐藏层后加 `nn.Dropout(0.2)` 同种子重跑——MNIST 这种任务上 dropout 大概率**让指标持平或略降**（baseline 根本没过拟合）。这本身就是要记录的结论：正则化不是免费午餐，先见过拟合证据（train 远高于 val）再上正则。然后制造过拟合来验证：hidden 加到 512、epoch 拉到 30，train acc 冲 99.9%、val 停在 98.x——这时再加 dropout 才有正面效果。三个 run（baseline / 过拟合 / 过拟合+dropout）wandb 对比，结论写注释
  - **任务B（lr scheduler，20分钟）**：baseline 加 `StepLR(optimizer, step_size=2, gamma=0.5)` 或 `CosineAnnealingLR(T_max=5)`，每 epoch 末尾 `scheduler.step()`，把当前 lr（`scheduler.get_last_lr()[0]`）也记进 wandb。观察 loss 曲线尾部：lr 变小后抖动收敛、更平滑
  - **任务C（调试工具箱，40分钟）**：故意制造三个 shape bug 各修一次，过程写注释——① 忘拍平：(B,1,28,28) 直接喂 Linear ② 层间维度不接：128 出接 32 入 ③ 广播暗坑：把 (B,) 的标签 reshape 成 (B,1) 再和 (B,10) 的输出做比较——Day 2 的广播规则在这里变成**不报错**的错误（(B,1) vs (B,10) 广播成 (B,10)，accuracy 算出毫无意义的值）。修法套路固定：`print(x.shape)` 或写 `assert x.shape == (B, 10)` 逐层二分定位。行业标准动作：每个模型的 forward 写完先塞一个 dummy batch 验证 shape，再开始训练
  - **思考题**：CE = LogSoftmax + NLLLoss，PyTorch 为什么把两步合一？（数值稳定性：softmax 里 exp 大数会上溢，先减 max 再 exp 数学等价但数值安全——这就是 log-sum-exp trick，CE 内部全程 log 域计算，你手动拆两步就丢了保护。顺带一问：推理时多分类输出层为什么连 softmax 都不用加？——argmax 在 softmax 前后结果一致，softmax 是单调变换）
- **交付**（`day11_regularize_debug.py`）：
  - [ ] dropout 三 run 对比 + 结论注释
  - [ ] scheduler run + lr 曲线
  - [ ] 三个 shape bug 的制造/修复记录
  - [ ] log-sum-exp 一句话解释
- **为什么重要**：今天全是工程卫生习惯——不无脑正则、scheduler 收尾、dummy batch 验 shape、对"不报错但语义错"的广播保持警惕。这些习惯在阶段 3 手写 GPT 时每条都能省你半小时

---

### Day 12（约1h）device 无关写法 + checkpoint

- **做**：
  - **任务A（device 无关化，20分钟）**：文件开头 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`，三处改动：`model.to(device)`、每个 batch `X, y = X.to(device), y.to(device)`、启动时打印 device。WSL2 里 CUDA 可用性的判断见「环境速查」；False 也完全不影响，MNIST 一个 epoch CPU 也就十几秒
  - **任务B（checkpoint，25分钟）**：训练完 `os.makedirs("checkpoints", exist_ok=True)` + `torch.save(model.state_dict(), "checkpoints/mnist_mlp.pt")`；然后**新起进程**（或同文件重建一个同构模型）`model.load_state_dict(torch.load(path))` + `model.eval()`，对测试集前 20 张图预测并打印 预测 vs 真值。两个要点：① `load_state_dict` 要求结构完全一致——改一层就 load 不进，这正是"存的是参数字典、不是模型类"的含义；② 跨设备加载用 `torch.load(path, map_location=device)`，阶段 4 租卡训练、回本地推理时必用
  - **任务C（可选，10分钟）**：把 `{"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}` 存成一个 dict——这才是"训练可恢复"的完整 checkpoint，HuggingFace Trainer 存的就是这个结构
  - **思考题（注释）**：为什么存 `state_dict()` 而不是 `torch.save(model)` 整个对象？（后者 pickle 了类的完整路径，代码一改结构、一换目录就 load 不回来；state_dict 是纯张量字典，与代码解耦）
- **交付**（`day12_device_ckpt.py`）：
  - [ ] device 无关版 MNIST
  - [ ] save → 新进程 load → 预测 20 张图对照
  - [ ] 思考题注释
- **为什么重要**：这两条是"写玩具"和"写工程"的分水岭。阶段 4 在租来的 GPU 上训几小时，没有 checkpoint 断一次全没；device 写死了，换台机器就是一堆 device mismatch 报错

---

### Day 13（约1.5h）验收①：闭卷默写

- **做**：新建空文件 `day13_closed_book.py`，**关掉所有参考资料**，计时 40 分钟从零默写 MNIST 完整脚本。默完对照这份清单自查：① Dataset/DataLoader ② nn.Module（`__init__` + `forward`）③ 五件套训练循环 ④ no_grad + model.eval() 的评估、accuracy = `(logits.argmax(1) == y).float().mean()` ⑤ checkpoint 保存
- **卡壳处理**：卡住超过 10 分钟的地方标 `# STUCK:` 继续往下写，**不许翻资料**。写完与 Day 12 代码 diff，把每个 `# STUCK:` 和 diff 差异列成清单——这就是你"以为会了其实没会"的精确地图
- **高频遗忘点**（写完再看，别提前偷看）：`optimizer.zero_grad()`（放循环开头是惯例）、`model.eval()`、评估的 `no_grad`、argmax 比标签而不是 logits 直接比、shuffle 只给 train loader、标签 long 类型
- **交付**：
  - [ ] `day13_closed_book.py`
  - [ ] `day13_diff_notes.md`（差异清单，每条一句话根因）
- **达标线**：不查资料能跑通、accuracy ≥96%（比 Day 10 放宽 1%，闭卷允许瑕疵）。做不到就把薄弱点留到 Day 14 复盘时补，然后**重默一次**——闭卷默写的价值一半在第二次

---

### Day 14（约1h）验收②：口述 + 复盘

- **做**：
  - **任务A（口述五问，30分钟）**：不看笔记，把五问各用 ≤3 分钟讲出来（录音或对墙讲都行），讲完再翻前 13 天交付物核对：
    1. `loss.backward()` 时计算图里发生了什么？（答案骨架在 Day 3 交付⑥ + Day 4 micrograd：从 loss 标量出发按逆拓扑序走图，逐节点调 `_backward`，把 `out.grad` × 局部梯度**累加**进叶子的 `.grad`——能讲出"拓扑排序"和"累加"两个词才算过关）
    2. 为什么评估要 `no_grad()`？（Day 8 思考题①；能补一句"eval() 和 no_grad() 各管什么"加分）
    3. 广播在什么情况下会悄悄给出错误结果？（Day 2 规则 + Day 11 任务C③：形状兼容时**不报错**地算出语义错误的结果——危险不在报错在沉默，防御是 dummy batch + shape assert）
    4. 为什么标准化只能用训练集统计量？（Day 9 交付④ + 任务C 的泄漏对照数字）
    5. CrossEntropyLoss 输入的应该是 logits 还是概率？为什么？（Day 10 交付④ + Day 11 的 log-sum-exp）
  - **任务B（写阶段总结，20分钟）**：`stage1_summary.md`——五问完整版答案 + 14 天交付物清单 + `day13_diff_notes` 暴露的薄弱点 + 每条薄弱点的补丁动作。这是阶段 2 开始前你重读的入口
  - **任务C（10分钟）**：勾掉主计划里阶段 1 的验收项，往阶段 2 期待清单写具体的事——参考：「CNN（卷积/池化为什么在图像上有效）→ ResNet 残差连接解决了什么退化问题 → 迁移学习微调一个真实图像分类任务」
- **交付**：
  - [ ] `stage1_summary.md`
  - [ ] 主计划勾选更新
- **达标线**：五问都能不假思索讲出「机制 + 一个代码级证据」。讲不出的那一问，答案就在对应那天的交付物里——回去补完再勾验收，别带着缺口进阶段 2

---

## 时间不够时的砍单顺序

保底的三个核心任务是 ①纯tensor线性回归、③Dataset/DataLoader、④MNIST（它们对应验收标准）。可压缩的：Day 2 广播练习减半、Day 11 并入 Day 10、Day 12 推迟。**micrograd（Day 4~5）不建议砍**——它是"理论结合实践"这个计划理念的样板，砍了它阶段 3 手写 GPT 会吃力。
