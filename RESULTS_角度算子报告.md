# Angular Temporal Modeling：Standard vs Quaternion

依据 `Angular_Temporal_Standard_vs_Quaternion_快速验证_Codex_Instruction.docx`。
PTB-XL，fold 1–8 / 9 / 10，阈值 0.5，**seed 42**。

固定 RLA 表示与已验证的 1280 ms 时间范围，**只改变 angular 序列上的时间算子**：

> 同一条 A₁:T　→　Standard Temporal Encoder　vs.　Quaternion Temporal Encoder

---

## 1. 设置

Angular 固定为 20 ms 一阶关系 `A_t = [dot, cross_x, cross_y, cross_z]`，
两个模型读取的 A 张量**逐元素完全相同**；R、L 分支也完全相同。

| | Angular 编码器 | Angular 宽度 | Angular 参数 | 总参数 | 实测 RF |
|---|---|---|---|---|---|
| RLA-Standard | 4 个普通实值通道，标准时间卷积 | 44 (= 2Q) | 79,508 | 168,025 | 640 / **1280 ms** |
| RLA-Quaternion | `q = dot + cross_x i + cross_y j + cross_z k`，Hamilton 卷积 | 88 (= 4Q) | 78,870 | 168,795 | 640 / **1280 ms** |
| 差 | | | −638 (−0.80%) | +770 (+0.46%) | 相同 |

宽度取 2Q / 4Q（Q=22）：四元数卷积每层 `4Q²K` 权重、实数卷积 `W²K`，`W=2Q` 时两者相等
——这是项目一期 §7.1 已确立的公平性约定。§4 要求「选择最接近的参数量，不改变 RF 配参数」，
两边 RF 都锁死在 640 samples。

**架构说明：** 上一轮的单编码器 RLA-Long 把 8 个通道拼接后送入同一个 stem，
**没有可分离的 angular encoder**，因此不满足 §6 的复用条件，其 0.9186 未被采用。
本轮改为分支架构（R、L、A 各自独立编码器 → 投影 → concat → LayerNorm → 线性头），
两个模型均新训。

---

## 2. 结果

| Model | Macro AUROC | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|
| RLA-Standard | **0.9130** | .9384 | .9047 | .9244 | .9074 | .8901 |
| RLA-Quaternion | 0.9123 | .9370 | .9072 | .9202 | .9109 | .8862 |

| Quaternion − Standard | Δ |
|---|---|
| **Macro AUROC** | **−0.0007** |
| NORM | −0.0014 |
| MI | +0.0025 |
| STTC | −0.0042 |
| CD | +0.0035 |
| HYP | −0.0039 |

---

## 3. 分析

**判定：Indistinguishable（§8 第二行）。**
−0.0007 深在 0.005 噪声带内，两种归纳偏置在 seed 42 下无法区分。

**逐类差的绝对值（最大 0.0042）比 macro 差（0.0007）大得多，且符号互相抵消**：
MI、CD 为正，NORM、STTC、HYP 为负，没有任何一致模式。这是随机波动的特征，
而非系统性效应——若 Hamilton 耦合真的提供了结构性优势，应当在若干类别上同向体现。

**这是 Quaternion 假设的第三次独立检验，三次结论一致：**

| 检验 | 框架 | Δ Macro AUROC |
|---|---|---|
| 一期 M2 − M1 | 几何取代 raw，3 seed | −0.0012 |
| 融合 F2 − F1 | 几何叠加 raw，3 seed | +0.0003 |
| **本轮 Quat − Std** | **分支、参数对齐、RF 对齐，1 seed** | **−0.0007** |

三次全部落在 ±0.0012 以内。**本轮是其中最严格的一次**：A 张量逐元素相同、
R/L 分支相同、RF 相同、angular 参数差 0.8%、总参数差 0.46%，唯一变量就是
`nn.Conv1d` 与 Hamilton 分块耦合。在这样的条件下仍测不出差异。

**一个附带观察（非本轮对照变量）：** 本轮两个模型（0.9130 / 0.9123）均低于上一轮
单编码器 RLA-Long 的 0.9186，差约 0.006，提示三分支拆分本身可能有代价。
但两者架构差异不止一处（三个独立 stem、三个投影、分支更窄），且均为单 seed，
**故表 13 的绝对值不应与上一轮直接并列比较**，此处不作结论。

---

## 4. 结论（<250 字）

在固定 RLA 表示、固定 1280 ms 时间范围、A 张量逐元素相同、R/L 分支与融合头完全一致、
angular 参数差 0.8% 的严格对照下，Quaternion − Standard = **−0.0007**，落在项目
快速筛查噪声带（0.005）之内；五类差值符号混杂、幅度均 ≤0.0042 且相互抵消，
未见任何一致的结构性优势方向。

**结论：Hamilton-structured angular temporal modeling 未出现值得补多 seed 的信号。**
按 §8 第二行，不扩展 Quaternion architecture，本线停止。§9 亦明确禁止因单 seed
未赢而继续堆叠 Quaternion layer。

该结论与此前两次独立检验（M2−M1 = −0.0012，F2−F1 = +0.0003）方向一致，
三次均在 ±0.0012 内，构成一致的否定证据。本轮全部为矩阵化实值/四元数卷积对照，
不涉及 SO(3) invariance/equivariance 主张。

---

## 5. 这一结果在论文故事中的位置

按 §10：

> Temporal evolution = scientific question　｜　Quaternion = angular modeling choice

本轮的否定结果**不影响上一轮 temporal context 的正向发现**
（Long − Short = +0.0093，其中 CD +0.0332）。它只关闭了「Quaternion 作为独立方法
贡献」这条支线——而该支线现已被三次独立检验一致关闭。

主线仍是 temporal evolution：下一步应是补 RLA-Short / RLA-Long 的多 seed 确认
该正向信号，而非继续在算子结构上投入。

---

## 附：本轮约束核对

未改变 A 的 20 ms lag；未引入 multi-scale 或 second-order；未加入 attention /
Transformer / gating / 新 fusion；未同时调整 temporal RF；未依据 test fold 调参；
未因单 seed 结果继续堆叠 Quaternion 模块；未作任何 SO(3) invariance/equivariance 声明。
`q_t` 在代码与报告中均按 full-angle relation descriptor 表述，未称其为 physical
rotation quaternion。M0–M4、M0-Wide、F1–F4、R/RA/RU/RLA 及 temporal 系列的定义与
checkpoint 均未改动（旧 checkpoint 仍可 `strict=True` 加载，`tables.md` 逐字节一致）。

QuaternionConv 的 Hamilton 实现经手算用例验证：
`(1+2i+3j+4k) ⊗ (5+6i+7j+8k) = −60+12i+30j+24k`，反向为 `−60+20i+14j+32k`，
并验证 `i⊗j=k`、`j⊗i=−k`、`k⊗k=−1`；四个权重分量均获得有限非零梯度。
