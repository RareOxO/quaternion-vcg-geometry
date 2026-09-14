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
