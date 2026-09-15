# RLA Temporal Context 消融

依据 `RLA_Temporal_Context_快速验证_Codex_Instruction.docx`。
PTB-XL，fold 1–8 训练 / 9 验证 / 10 测试，阈值 0.5，**seed 42**。

固定已验证的 RLA 表示，**只改变模型能利用的时间范围**，回答一个问题：
更长时间范围内的 R/L/A evolution，是否比局部瞬时动态提供额外诊断信息？

---

## 1. 设置

R/L/A 定义完全沿用现有实现，未作任何改动：
`R_t = ‖V_t‖`，`L_t` 由 raw XYZ 前向差分，`A_t` 为 20 ms 一阶 `[dot, cross]`。
三个模型输入张量**逐元素相同**，head、loss、optimizer、scheduler、early stopping 一致。

| Variant | depth | kernel | width | 实测 RF | 目标 | Params |
|---|---|---|---|---|---|---|
| RLA-Short | 1 | 5 | 127 | 45 samples / **90 ms** | ~100 ms | 168,153 |
| RLA-Medium | 2 | 7 | 76 | 190 samples / **380 ms** | ~400 ms | 166,293 |
| RLA-Long | 4 | 5 | 64 | 640 samples / **1280 ms** | ~1.2 s | 168,453 |

RLA-Long **就是**已有的 `RLA` 实验，其 seed42 结果直接复用，未重训（§7 步骤 5）。
宽度是数值解出来的，使三者参数量落在 1.3% 以内，**让 RF 成为唯一的系统性变量**（§4）。
Medium 用 kernel 7 是因为只调 depth 只能得到 260 ms 或 600 ms，离 400 ms 太远；
§3 允许通过 block 数 / dilation / kernel / pooling 达成目标 RF。

RF 由 `receptive_field_samples()` 按编码器**实际的 (kernel, stride, dilation) 序列**
逐层计算，含 pooling，不依据变体名称声称（§5）。

---

## 2. 结果

| Model | RF (ms) | Params | Macro AUROC | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|---|---|
| RLA-Short | 90 | 168,153 | 0.9093 | .9349 | .9167 | .9230 | .8828 | .8894 |
| RLA-Medium | 380 | 166,293 | 0.9141 | .9383 | .9122 | .9252 | .9044 | .8906 |
| **RLA-Long** | 1280 | 168,453 | **0.9186** | .9434 | .9173 | .9260 | .9160 | .8903 |

| 对比 | Δ Macro AUROC |
|---|---|
| Medium − Short | **+0.0048** |
| Long − Medium | **+0.0045** |
| **Long − Short** | **+0.0093** |

### 逐类 Long − Short

| Class | Δ |
|---|---|
| **CD** | **+0.0332** |
| NORM | +0.0085 |
| STTC | +0.0031 |
| HYP | +0.0009 |
| MI | +0.0006 |

---

## 3. 分析

**判定：longer temporal context helps（§9 第一行）。**
这是本系列实验中**第一个明确的正向结果**。

严格单调 Short < Medium < Long，两段增益几乎相等（+0.0048 / +0.0045），
总增益 +0.0093 是噪声带 0.005 的约 1.9 倍。且该对照在参数量对齐（差 ≤1.3%）、
输入逐元素相同的条件下取得，**RF 是唯一变量**。

**增益几乎全部来自 CD（+0.0332，占绝对主导）。** 这在机制上完全自洽：
CD（传导阻滞）的判读依据是 QRS 时限与束支传导形态，本身就是跨越 100+ ms 的时间结构，
而 90 ms 的感受野连一个完整 QRS 波群都覆盖不了。MI（+0.0006）与 HYP（+0.0009）几乎
不受影响，符合它们更依赖瞬时振幅与局部形态的性质。

这同时与一个更早的观察对上：一期中 M3（多尺度）在 CD 上是全场最好的
（.8860 vs M2 的 .8752）。当时该信号被 HYP 的大幅退化淹没，现在在「固定表示、
只变时间范围」的干净对照下重新显现，两处证据指向同一结论。

**但必须克制表述。** +0.0093 仅为噪声带的 1.9 倍，且为 seed 42 单点。
§10 明确要求正式 claim 必须补多 seed。当前只能说「seed42 下存在强正向信号」，
不能说「已证实」。本轮全部为 Real 模型，**不能据此对 Quaternion 作任何推断**（§12）。

---

## 4. 结论（<300 字）

固定 RLA 表示、仅改变感受野的三档对照显示：Macro AUROC 随时间范围单调上升，
90 ms → 380 ms → 1280 ms 对应 0.9093 → 0.9141 → 0.9186，Long−Short = +0.0093，
约为既有 seed 波动带（0.005）的 1.9 倍。三者输入逐元素相同、参数量差 ≤1.3%，
因此该差异可归因于时间范围本身。逐类拆解显示增益高度集中于 CD（+0.0332），
其余四类均在 0.001–0.009，与 CD 依赖 QRS 时限这一跨百毫秒时间结构的临床事实一致。

**结论：temporal context 显示出额外诊断价值，证据方向明确但强度为单 seed 级别。**
建议按 §9 第一行补 Short/Long 这一对的多 seed 确认单调性与幅度稳定，再进入
temporal architecture 设计。本轮未涉及 Quaternion，不对其有效性作任何推断。

---

## 5. 下一步（待确认，未启动）

```bash
python -m qdg suite --experiments RLA_short --seeds 42 43 44
python -m qdg suite --experiments RLA --seeds 42 43 44
```

只补 Short 与 Long 这一对即可确认主效应；Medium 可暂不补。

---

## 附：一处计算修正

§5 要求实现真 RF calculator 后，发现原有闭式公式**漏算了块间三个
`avg_pool(2,2)` 各自的 kernel 贡献**：

| | 旧闭式（错） | 逐层实测（对） |
|---|---|---|
| 4-block backbone（M0–M4、RLA-Long） | 605 / 1210 ms | **640 / 1280 ms** |
| `M1_mlp` 对照 | 5 / 10 ms | **40 / 80 ms** |

该字段仅用于记录，不参与任何计算，**所有已训练模型与 AUROC 结果均不受影响**。
但此前报告中「感受野 605 点（1.21 s）」应更正为 640 点（1.28 s）；
`M1_mlp` 亦不应描述为「去掉时间维混合」——kernel 1 去除的是块内*可学习*的时间混合，
固定 pooling 仍跨时间聚合，其真实感受野为 80 ms。该对照的结论（0.8596）不变。

## 附：本轮约束核对

未改动 R/L/A 定义；A 固定 20 ms 一阶，未引入 multi-scale 或 second-order；
未加入 Quaternion、attention、gating、Transformer；未做超参数搜索；
未依据 fold 10 选择 RF 或修改网络；未自动补 seed 或进入后续架构设计。
M0–M4、M0-Wide、F1–F4、R/RA/RU/RLA 的定义与 checkpoint 均未改动
（旧 checkpoint 仍可 `strict=True` 加载，`tables.md` 逐字节一致）。
