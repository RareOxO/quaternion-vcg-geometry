# Quaternion-VCG Dynamic Geometry — Phase One (PTB-XL)

Implementation of `Quaternion_VCG_动态几何_实现方案_公式排版修订版.md`. The question is not
whether VCG can be "quaternionized", but whether an explicit quaternion **relation
descriptor** between two 3-D directions is a better representation than raw XYZ, and
whether the Hamilton interaction rule beats an ordinary real one on identical input.

Phase one deliberately excludes radial branch, scale attention, Transformer, ECG+VCG
fusion, EMD, curvature and torsion.

## The representation

12-lead ECG → Kors → VCG `V_t = [X_t, Y_t, Z_t]` → unit direction `u_t = V_t / (‖V_t‖ + eps)`.

For a lag Δ the study's descriptor is

```
q_t^Δ = [ u_t · u_{t+Δ} ,  u_t × u_{t+Δ} ]  =  [cos θ, n sin θ]
```

Three things the code keeps strictly apart (`qdg/geometry.py`):

| function | value | note |
|---|---|---|
| `pure_quaternion_product(u, v)` | `[−u·v, u×v]` | the raw algebraic Hamilton product |
| `relation_descriptor(u, v)` | `[+u·v, u×v]` | what this study uses |
| rotation quaternion | `[cos(θ/2), n sin(θ/2)]` | **not** what this is |

The descriptor is a full-angle object, so it is *not* a standard physical rotation
quaternion. `second_order(u, Δ) = q(u_t, u_{t+Δ}) − q(u_{t−Δ}, u_t)`, a plain difference —
`inverse(q_prev) ⊗ q_next` is *not* used, since the descriptor is not a half-angle rotation.

Scales are declared in milliseconds and converted with `round(lag_ms · fs / 1000)`, so
10/20/40/80 ms become 5/10/20/40 samples at 500 Hz. Scales are separate quaternion
*channels*; they are never packed into the r/i/j/k components.

`‖q_first‖ = 1` holds by construction. Re-normalization is a config flag
(`model.renormalize`, default `false`) and is never applied silently; `qdg check-geometry`
reports the actual drift on real waveforms.

## The models

| Model | Input | Operator |
|---|---|---|
| M0 | raw VCG `[X, Y, Z]` | real TCN |
| M1 | first-order `[dot, cross]` @ 20 ms | real TCN |
| M2 | *identical numbers to M1* | Hamilton `QuaternionConv1d` TCN |
| M3 | first-order @ 10/20/40/80 ms | quaternion TCN |
| M4 | M3 + second-order difference | quaternion TCN |

Stem, depth, kernel, pooling, head, optimizer and schedule are shared by all five; only
the feature block and the interaction rule are ablated. A quaternion conv of Q channels
holds `4Q²K` weights and a real conv of width `C` holds `C²K`, so `real_width = 2·quaternions`
equalizes them exactly (per-channel bias/gain terms leave a small remainder — Table 3
prints every row's parameter count).

## Raw + Geometry fusion (双分支 addendum)

M1–M4 replace the raw signal, so M1 − M0 measures a *replacement gap*, not an added
value. The fusion ladder keeps M0 intact as one branch and adds a geometry branch:

| Model | Raw branch | Geometry branch |
|---|---|---|
| M0-Wide | M0 widened to 92 ch (capacity control, no geometry) | — |
| F1 | M0 | M1 (real, 20 ms) |
| F2 | M0 | M2 (quaternion, 20 ms) |
| F3 | M0 | M3 (quaternion, 4 scales) |
| F4 | M0 | M4 (+ second order) |

Late fusion only: each branch keeps its own stem, so an effect is attributable to the
geometry features rather than to a wider first layer. The join is identical for F1–F4 —
project each branch to `fusion_dim`, concatenate, LayerNorm, one linear to the classes —
with no attention, gating or hidden MLP. M0–M4 are untouched; their checkpoints still
load and `runs/tables.md` is byte-stable.

`M0-Wide` matches F1's parameter count to 0.09% (342,889 vs 342,597) and F4's to 2.1%,
so `F1 − M0-Wide` separates geometry from raw capacity. Component gains are redefined:
`F1 − M0` (real geometry complementarity), `F1 − M0-Wide` (vs extra capacity),
`F2 − F1`, `F3 − F2`, `F4 − F3`. Results go to `runs/fusion_tables.md`.

```bash
python -m qdg suite --stage stage-a --seeds 42 43 44   # M0, M0-Wide, F1 — judge F1 first
python -m qdg suite --stage fusion --seeds 42 43 44    # the whole ladder
```

## Representation diagnostic (v2 addendum)

Neither replacing raw XYZ with geometry nor adding geometry to it helps, so this stage
stops adding structure and asks *where* the information is lost. All variants are Real,
single 20 ms scale, seed 42 only.

| Variant | Input | Ch | Question |
|---|---|---|---|
| R | `r = ‖V‖` | 1 | How much does magnitude alone carry? |
| RA | `r` + M1's 20 ms `[dot,cross]` | 5 | Does magnitude restore the angular-only loss? |
| RU | `r` + `u = V/(r+eps)` | 4 | Is `(r, u)` a lossless reparameterization of raw XYZ? |
| RLA | `r` + linear `ΔV` + `[dot,cross]` | 8 | Does linear dynamics recover more? |

RU is the audit, not a candidate: `V = r·u`, so if RU ≈ M0 while RA ≪ RU, the loss is in
the `u → [dot,cross]` compression rather than in magnitude.

Linear velocity is built on **raw XYZ**, never on `u`: §2 defines `l = (V_{t+1}−V_t)/dt`;
the code emits `(V_{t+1}−V_t)/vcg_std`, which is that velocity divided by the constant
`Fs·vcg_std`. Both scalers (`vcg_std`, and `r`'s `sqrt(Σ vcg_std²)`) come only from the
training folds and are already in the existing cache, so no re-prepare is needed and
amplitude is never removed. `qdg feature-stats` prints the p01/p50/p99 of every block.

```bash
python -m qdg feature-stats                          # scale check before training
python -m qdg suite --stage diagnostic --seeds 42    # R, RA, RU, RLA; M0/M1 are reused
```

Results and the supported / partially supported / not supported verdict go to
`runs/diagnostic_tables.md`.

## RLA temporal context ablation

RLA recovers M0's accuracy, so this stage fixes that representation and varies only
how much time the model can see. Same RLA input, same head, same recipe; seed 42.

| Variant | Depth | Kernel | Width | Actual RF | Params |
|---|---|---|---|---|---|
| RLA-Short | 1 | 5 | 127 | 45 samples, 90 ms | 168,153 |
| RLA-Medium | 2 | 7 | 76 | 190 samples, 380 ms | 166,293 |
| RLA-Long | 4 | 5 | 64 | 640 samples, 1280 ms | 168,453 |

RLA-Long **is** the existing `RLA` experiment, so its seed-42 run is reused rather than
retrained. Widths are solved so the parameter counts stay within 1.3% of Long, keeping
the receptive field the only systematic variable.

The receptive field is computed by `receptive_field_samples()` from the encoder's real
`(kernel, stride, dilation)` sequence, pooling included — never asserted from a name.
That correction is why the 4-block encoder reports 640 samples here and 605 in earlier
notes: the closed form it replaced dropped the three `avg_pool(2,2)` layers.

```bash
python -m qdg suite --stage temporal --seeds 42   # RLA_short, RLA_medium; Long is reused
```

Results and verdict go to `runs/temporal_tables.md`.

## Angular temporal operator: Standard vs Quaternion

Longer context helps, so this stage fixes the context and asks *how* the angular
sequence should be modelled. One A tensor, two temporal operators.

| | Angular encoder | Angular width | Angular params | Total params | RF |
|---|---|---|---|---|---|
| RLA-Standard | 4 real channels, standard conv | 44 | 79,508 | 168,025 | 1280 ms |
| RLA-Quaternion | `q = dot + cross_x i + cross_y j + cross_z k`, Hamilton conv | 88 | 78,870 | 168,795 | 1280 ms |

Both use the branched architecture (`RLAB`): R, L and A each get their own encoder,
joined by the same projection + concat + LayerNorm + linear. The R and L branches, the
A tensor, the receptive field, the fusion, the head and the recipe are shared; only the
angular operator differs. Angular widths are 2Q and 4Q so the two hold matched weight
counts (0.8% apart); totals are 0.46% apart.

**The previous single-encoder `RLA` cannot serve as the Standard control**: it
concatenates all eight channels before one stem, so it has no separable angular encoder.
Both models are therefore trained fresh, and `RLA-Long`'s 0.9186 is not reused here.

`q` is a full-angle relation descriptor, not a physical rotation quaternion, and a
vanilla QuaternionConv carries no SO(3) guarantee. The claim under test is only whether
Hamilton scalar-vector coupling is a better inductive bias for this sequence.

```bash
python -m qdg suite --stage angular --seeds 42
```

Results and verdict go to `runs/angular_tables.md`.

## Data

PTB-XL 1.0.3 only. 12-lead, 10 s, 500 Hz, band-pass 0.5–100 Hz. Labels are the five
diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label. Official `strat_fold`:
1–8 train (17084), 9 validation (2146), 10 test (2158). Loss is `BCEWithLogitsLoss` with
positive weighting; the primary metric is macro AUROC with per-class AUROC alongside.
Model selection uses fold 9 only; fold 10 is read once, by `evaluate`.

## Usage

```bash
conda activate quaternion_py3xx
pip install -e .

python -m qdg sanity                 # §6 theory checks, expected vs actual
python -m pytest -q                  # full unit suite
python -m qdg audit                  # verify every PTB-XL record file is present
python -m qdg prepare                # build the memmap cache (~5 GB)
python -m qdg check-geometry         # descriptor norm drift on real waveforms
python -m qdg profile                # parameter counts per experiment

python -m qdg train --experiment M2  # one model, then evaluate on fold 10
python -m qdg suite --seeds 42 43 44 # everything, then write the tables
python -m qdg tables --root runs     # regenerate tables from existing runs
```

`suite` skips runs that already have `best_test_metrics.json`, so it is resumable.

## Outputs

`runs/tables.md` holds the four tables of §10, generated from the runs, never typed:

1. main results M0–M4 with per-class AUROC,
2. temporal-scale analysis (10/20/40/80 ms and the combination),
3. Real Conv vs Real MLP vs Quaternion Conv on identical input, with parameter counts,
4. component increments M1−M0, M2−M1, M3−M2, M4−M3.

## Layout

```
qdg/geometry.py       Kors, directions, Hamilton product, relation descriptor, 1st/2nd order
qdg/quaternion_nn.py  QuaternionLinear, QuaternionConv1d, real and quaternion encoders
qdg/models.py         M0-M4 from one shared backbone
qdg/data.py           PTB-XL cache and dataset
qdg/engine.py         training and evaluation
qdg/experiments.py    experiment registry and table generation
qdg/sanity.py         §6 checks and the numerical norm report
```
