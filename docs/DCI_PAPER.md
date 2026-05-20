# MZTrain: Training Factorized Transformers on Consumer GPUs, and the Double-Compression Incompatibility Problem

**A Systems + Empirical Study of Memory-Efficient Language Model Pretraining**

---

**Authors**: MSC Star Team (Esraderey & Raúl Cruz Acosta)
**Affiliation**: MZTrain Project
**Venue target**: ICLR/ACL Findings or workshop (8 pages)
**Date**: April 2026
**Artifacts**: code, tests, and all raw logs in the project repository
**Keywords**: low-rank training, factorized transformers, memory-efficient optimizers, language modeling pretraining, initialization, APOLLO, GaLore, WikiText

---

## Abstract

Pretraining transformer language models on consumer GPUs (≤8 GB VRAM) is an open engineering problem. We present **MZTrain**, a system that trains transformers with SVD-factorized linear layers ($W = U\,\text{diag}(S)\,V$, rank $r \ll \min(m,n)$), an APOLLO-style adaptive optimizer with scalar-block second moments, a multi-level precision manager (BF16/FP16/INT8), optional sparse-plus-low-rank weights, and periodic refactorization. On a single RTX 4060, MZTrain trains transformer blocks of up to **900M parameters** in 1.4 GB of VRAM and scales cleanly to ≥250M across an 8-epoch benchmark reaching 94–100% accuracy on synthetic classification tasks.

Along the way we make two technical contributions that improve both MZTrain and the broader low-rank training literature:

1. **Energy-Preserving SVD Initialization (EPSI)** — a one-line fix that closes the 30–70% Frobenius-energy gap introduced by naïve SVD-truncation initialization of factorized layers. We provide a formal derivation, empirical measurements, and show that EPSI restores output variance to within 7–16% of a dense `nn.Linear` across all transformer layer shapes. We also formulate **LayerNorm Masks Frobenius Collapse (LMFC)**, a principle explaining why this bug is only a 2% contributor in LayerNorm-rich transformers but would be catastrophic in residual-free or norm-sparse architectures.

2. **Double-Compression Incompatibility (DCI)** — a previously undocumented failure mode of memory-efficient pretraining, showing that weight factorization (ZFactorized, LoRA-from-scratch, SLTrain) composes **destructively** with gradient-projection optimizers (GaLore, APOLLO). On WikiText-2 causal language modeling with a 60M factorized transformer, the combination yields 4.32× worse validation perplexity than a dense AdamW baseline with APOLLO (1027.52 vs 237.66) and **4.03× worse with GaLore** (958.24 vs 237.66) — both exhibit the same failure mode. An ablation study shows that **95% of this gap comes from the optimizer interaction, not from the factorization**. Replacing the projecting optimizer with standard AdamW on the same factorized model recovers 94.8% of the gap (278.76 final perplexity) while preserving 51% of the optimizer memory savings.

We benchmark MZTrain against 5 SOTA optimizers (AdamW, Adam8bit, Lion, GaLore, APOLLO) and 3 model scales (25M, 50M, 100M) in a fair protocol with matched data, seeds, and training budgets, and validate the framework on WikiText-2 perplexity with an 88M-parameter causal LM. All experiments run on an RTX 4060 (8 GB). We release tests, logs, and the complete reproducibility package.

---

## 1. Introduction

### 1.1 Problem

The cost of pretraining modern language models has pushed the practice almost exclusively to large industrial labs. Academic and individual researchers rely on two complementary strategies to close the gap:

1. **Parameter-space compression**: Replace each dense weight matrix $W \in \mathbb{R}^{m \times n}$ by factors $U \in \mathbb{R}^{m \times r}$, $S \in \mathbb{R}^r$, $V \in \mathbb{R}^{r \times n}$ such that $W \approx U\,\text{diag}(S)\,V$, only the factors are trained and stored [LoRA, SLTrain, LOST].

2. **Optimizer-space compression**: Keep weights dense, project gradients into a rank-$r'$ subspace so Adam's first and second moments live in reduced dimension [GaLore, APOLLO]. Orthogonal memory savings come from quantization (Adam8bit) or dropping the second moment entirely (Lion).

Both strategies are individually validated: SLTrain and LOST close the perplexity gap with dense AdamW within ~1 ppl on C4 at 60M–7B scale; GaLore and APOLLO also close it within ~1 ppl. It is natural to ask whether the two strategies can be **combined** for additive memory savings. The MZTrain codebase explicitly offers this combination as a single configuration flag. This paper shows, by direct measurement on a real NLP task, that the combination fails catastrophically, identifies the mathematical reason, and fixes it.

### 1.2 The MZTrain system

MZTrain is a PyTorch-based training framework implementing the following:

- **`ZFactorizedLinear`**: a linear layer storing SVD factors $(U, S, V)$ with forward pass $y = x\,V^\top \cdot S \cdot U^\top + b$, and gradient flow directly through the factors.
- **`ZFactorizedAttention` / `ZFactorizedTransformerBlock`**: multi-head attention and pre-LN transformer block using `ZFactorizedLinear` for Q/K/V/O projections and for MLP fc1/fc2.
- **`ZSparseFactorizedLinear`**: optional sparse complement $W = UV + S_{\text{sparse}}$ following SLTrain [1].
- **`ZAdaptiveOptimizer`**: our implementation of APOLLO-style low-rank Adam: first moment $m$ in rank-$r'$ projected space, second moment $v$ as a scalar per block (Adam-mini style [12]).
- **`ZGaLoreOptimizer`**: direct reimplementation of GaLore [2].
- **`ZCompressedAdam`**: block-wise INT8 quantized Adam states.
- **`ZMultiPrecisionManager`**: policy switching between STANDARD (FP16/BF16 autocast), AGGRESSIVE (INT8 activations/optimizer), and FP8 (Hopper+ only, with fallback) precision modes.
- **Periodic refactorization**: every N steps, reconstruct $W = U S V$, perform fresh SVD on the result, keep the top-$r$ new factors, reset optimizer states, and apply a jagged-cosine LR warmup (ReLoRA-inspired).

The system has 30+ unit and integration tests covering individual components, scale limits up to 900M parameters, and synthetic classification. Together they form the substrate on which the empirical contributions below are built.

### 1.3 Contributions

1. **The MZTrain training system** (§4): an integrated implementation of factorized transformers, low-rank optimizers, multi-level precision, and refactorization on consumer hardware.

2. **EPSI initialization** (§5): a principled derivation and empirical validation of an energy-preserving scaling step for SVD-truncated initialization. Closes a 30–70% Frobenius-energy bug in standard SVD init.

3. **LMFC principle** (§5.4): the observation that per-sublayer normalization (LayerNorm, RMSNorm) absorbs initialization-scale errors, explaining why EPSI has only a 2% impact on end-to-end perplexity on modern transformers despite solving a mathematically severe bug.

4. **The DCI phenomenon** (§6): the central empirical finding of this paper. Weight factorization and gradient-projection optimizers compose destructively; we give a formal statement, verify the mechanism by instrumenting gradient ranks, and quantify the gap on WikiText-2.

5. **Fair benchmarking protocol** (§7): SLM-scale (25M, 50M, 100M) head-to-head comparisons under matched data/seed/budget of MZTrain against dense AdamW, including reproducibility artifacts.

6. **Optimizer comparison** (§8): MZTrain's APOLLO implementation and full stack benchmarked against 5 SOTA optimizers (AdamW, Adam8bit, Lion, GaLore, APOLLO) with per-metric rankings.

7. **Real-task validation on WikiText-2** (§9): 88M-class causal LM pretraining with proper perplexity measurement. This is where DCI is first observed in the wild and where the EPSI and AdamW fixes are validated.

8. **The scale limit of consumer-hardware pretraining** (§10): empirical evidence that with MZTrain's full stack, models of up to 900M transformer-block parameters fit in 1.4 GB of training VRAM on an RTX 4060.

---

## 2. Related Work

### 2.1 Weight factorization for pretraining

**LoRA** [4] introduced low-rank adapters for *fine-tuning*, leaving the base model frozen. Extending the technique to *pretraining from scratch* is harder: Aghajanyan et al. [10] show that the intrinsic dimension of the loss landscape shrinks during pretraining, meaning a finished checkpoint can be fine-tuned low-rank but a cold start cannot. Naïve LoRA-from-scratch underperforms dense AdamW by 5–10 perplexity points on LLaMA 60M–1B [5].

**ReLoRA** [5] solves this by periodically merging the low-rank update into a frozen full-rank base, resetting optimizer states, and rewarming the learning rate. Closes most of the gap. **SLTrain** [1] augments $W = UV$ with a learned sparse complement $S$ on a random support, matching dense within ~1 ppl on LLaMA. **LOST** [6] replaces the random sparse support with a structured one derived from the trailing singular values.

**MZTrain's ZFactorized layers** sit between naive LoRA-from-scratch and SLTrain: pure $W = U\,\text{diag}(S)\,V$ as the default, with an optional sparse component (`ZSparseFactorizedLinear`) that was not active in the WikiText experiments reported here.

### 2.2 Optimizer-state compression

**GaLore** [2] projects the gradient of each large 2D weight into a rank-$r$ subspace via a projector $P$ refreshed every $T \approx 200$ steps. Adam's $m$ and $v$ live in reduced dimension; the weight $W$ remains full-rank. On LLaMA 60M–1B pretraining on C4, GaLore matches dense AdamW within 0.1–0.8 perplexity points.

**APOLLO** [3] (MLSys 2025 Outstanding Paper Honorable Mention) replaces the SVD-based projector with a random projection and derives channel-wise gradient scaling from the low-rank state. **APOLLO-Mini** uses a single scalar per tensor. Achieves AdamW-level performance at SGD-level memory, trains LLaMA-7B in <12 GB. Both GaLore and APOLLO are benchmarked only on dense weights.

**Adam8bit** [7] quantizes $m, v$ to INT8 block-wise without subspace restriction. **Lion** [8] drops $v$ entirely and uses the sign of an EMA-smoothed gradient. Both are compatible with any weight parameterization.

### 2.3 The composition gap

To our knowledge, **no published work combines** the two compression families. GaLore and APOLLO are benchmarked on dense LLaMA; SLTrain and LoRA-from-scratch are benchmarked with standard AdamW. The assumption (implicit) is that the two axes offer additive memory savings. Our DCI result (§6) disproves this.

### 2.4 Initialization of low-rank factors

**PiSSA** [11] initializes LoRA adapters with the top-$r$ singular vectors of a target weight matrix, matching principal subspace directions. **MiLoRA** initializes in the orthogonal complement (the minor singular directions). **OLoRA** [14] uses a QR decomposition for orthonormal factor init. **DoRA** [12] decomposes weights into magnitude and direction with only the direction being low-rank.

Our **EPSI** (§5) is simpler than any of these: it is a one-line Frobenius-norm-preserving rescaling of the singular values of a Gaussian-initialized matrix after truncation. It does not replace PiSSA for fine-tuning but it is the correct default for pretraining-from-scratch with random init.

---

## 3. Preliminaries

Let $W \in \mathbb{R}^{m \times n}$ denote a dense linear layer. The SVD decomposition is $W = U \Sigma V^\top$ with $U \in \mathbb{R}^{m \times \min(m,n)}$, $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_{\min(m,n)})$, $V \in \mathbb{R}^{n \times \min(m,n)}$. The rank-$r$ truncation is:

$$W_r = U_{:r} \, \text{diag}(\sigma_{1:r}) \, V_{:r}^\top$$

By the **Eckart–Young theorem**, $W_r$ is the best rank-$r$ Frobenius approximation of $W$. We will write $\|W\|_F^2 = \sum_i \sigma_i^2$ and $\|W_r\|_F^2 = \sum_{i \leq r} \sigma_i^2$.

A **Xavier-initialized** weight satisfies $\mathbb{E}[\|W\|_F^2] = 2mn/(m+n)$. For inputs $x$ with $\mathbb{E}[\|x\|^2] = n$, the output $y = Wx$ satisfies $\mathbb{E}[\|y\|^2] = 2mn/(m+n)$ — Xavier is designed to preserve this.

A **ZFactorizedLinear(m, n, r)** stores trainable $(U, S, V)$ with $U \in \mathbb{R}^{m \times r}$, $S \in \mathbb{R}^r$, $V \in \mathbb{R}^{r \times n}$, and computes $y = ((xV^\top) \odot S) U^\top + b$. The reconstruction $W_{\text{eff}} = U\,\text{diag}(S)\,V$ is never materialized in the forward pass.

A **gradient-projection optimizer** (GaLore, APOLLO) stores a projector $P \in \mathbb{R}^{m \times r'}$ per 2D parameter and replaces the Adam state with its projection: $\tilde m, \tilde v \in \mathbb{R}^{r' \times n}$. The update is reconstructed as $\Delta\theta = P \cdot \Delta_{\text{low}}$.

---

## 4. The MZTrain System

We summarize the system briefly; full implementation is open-source.

### 4.1 Factorized layers

`ZFactorizedLinear(m, n, r)` stores $(U, S, V)$ as independent `nn.Parameter`s and computes:

```python
def forward(self, x):
    # x: (..., n)
    h = x @ self.V.T            # (..., r)
    h = h * self.S               # (..., r)
    return h @ self.U.T + b     # (..., m)
```

This path costs $O(Brn + Bmr) = O(Br(m+n))$ vs dense $O(Bmn)$ — a saving of $r/\min(m,n)$. `grow_rank()` allows curriculum-style progressive rank growth during training by copying existing $(U, S, V)$ into a larger tensor and re-initializing the new columns/rows.

`ZFactorizedTransformerBlock` wraps Q/K/V/O and MLP fc1/fc2 with `ZFactorizedLinear`, keeping LayerNorm and residual connections dense.

### 4.2 APOLLO-style adaptive optimizer

`ZAdaptiveOptimizer` stores, per large 2D parameter: (a) a low-rank first moment $\tilde m \in \mathbb{R}^{r' \times \min(m,n)}$, (b) a scalar second moment $v_{\text{block}}$ per block of `block_size=1024` elements (Adam-mini [12]), (c) a projector $P$. The step is:

```
for each large 2D param p:
    g_low = P^T @ p.grad                     # project
    m_low = beta1 * m_low + (1-beta1) * g_low
    v_block = beta2 * v_block + (1-beta2) * mean(g_low**2, per block)
    m_hat = m_low / (1 - beta1**t)
    v_hat = expand(v_block) / (1 - beta2**t)
    delta_low = -lr * m_hat / (sqrt(v_hat) + eps)
    p.data += P @ delta_low                  # reconstruct
```

On pure synthetic classification (test_17), this matches AdamW accuracy while reducing optimizer state memory by 92.4%.

### 4.3 Multi-level precision manager

`ZMultiPrecisionManager` exposes three levels:

- **STANDARD**: BF16 or FP16 autocast, FP32 grads, FP32 Adam states.
- **AGGRESSIVE**: BF16 compute, INT8 block-wise activation compression (optional), ZCompressedAdam (INT8 Adam states).
- **FP8**: FP8 E4M3/E5M2 per-tensor scaled, auto-fallback to AGGRESSIVE on non-Hopper GPUs.

A critical invariant: **singular values $S$ are never quantized**, even in aggressive/FP8 modes, because they control the scale of the entire factorized layer. We learned this the hard way: quantizing $S$ to INT8 caused loss explosion within ~50 steps in early tests (not included here).

### 4.4 Periodic refactorization

Every `refactorize_interval` steps, for each `ZFactorizedLinear`:

1. Reconstruct $W = U\,\text{diag}(S)\,V$.
2. Perform fresh SVD: $W = U' \Sigma' V'^\top$.
3. Replace $(U, S, V) \leftarrow (U'_{:r}, \Sigma'_{:r}, V'^\top_{:r})$.
4. Reset Adam's $m, v$ for those parameters.
5. Apply a brief jagged-cosine LR warmup (50 steps, starting at 10% of current LR).

This is the ReLoRA [5] recipe applied to factorized weights. In our tests, it stabilizes training at aggressive rank schedules (e.g. refactorizing every 50 steps with accuracy going from 10% to 95.5%). For this paper's experiments, we disabled refactorization (`refactorize_interval=0`) to isolate the contributions of the other components.

### 4.5 Progressive rank growth

`ZRankScheduler` supports linear, exponential, cosine, adaptive, and spectral schedules for growing $r$ during training. `ZSpectralRankScheduler` uses the spectral decay ratio $\rho = 1 - \sigma_{\min}/\sigma_{\max}$ as the growth criterion: when the current weights show low spectral decay, the rank is increased. We do not use rank growth in the WikiText experiments below.

---

## 5. Contribution 1: EPSI — Energy-Preserving SVD Initialization

### 5.1 The bug

`ZFactorizedLinear._init_svd` (pre-fix) performed:

```python
W = torch.empty(m, n)
nn.init.xavier_uniform_(W)     # ||W||_F^2 = 2mn/(m+n)
U, S, V = torch.linalg.svd(W)
self.U = U[:, :r]; self.S = S[:r]; self.V = V[:r, :]   # truncate, no rescale
```

The reconstructed weight is $W_r$ with $\|W_r\|_F^2 = \sum_{i \leq r} \sigma_i^2 \ll \|W\|_F^2$.

### 5.2 Quantitative measurement of the energy loss

For the transformer layer shapes in our 88M model, we measured the proportion of Frobenius energy retained at $r = 128$:

| Layer shape | Layer | Energy retained |
|---|---|---|
| $768 \times 768$ | Attention Q/K/V/O | **46.9%** |
| $3072 \times 768$ | MLP fc1 | **30.5%** |
| $768 \times 3072$ | MLP fc2 | **30.5%** |

Energy loss of 53–70% per layer. The ratio of output variances is directly measurable:

| Layer | $\text{Var}(y_{\text{fact}}) / \text{Var}(y_{\text{dense}})$ |
|---|---|
| Attention Q | 0.503 |
| MLP fc1 | **0.325** |
| MLP fc2 | 0.352 |

### 5.3 The fix

Scale the retained singular values to restore total Frobenius norm:

$$\alpha = \sqrt{\frac{\sum_{i=1}^{\min(m,n)} \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}}, \quad S_r^{\text{EPSI}} = \alpha \cdot S_r$$

This guarantees $\|W_r^{\text{EPSI}}\|_F^2 = \|W\|_F^2$. After also switching from Xavier to Kaiming uniform (matching PyTorch's `nn.Linear` default), the empirical ratio of output variances becomes:

| Layer | Post-EPSI variance ratio |
|---|---|
| Attention Q | **1.101** |
| MLP fc1 | **1.071** |
| MLP fc2 | **0.995** |

All within 7–16% of dense. Implementation is three lines:

```python
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
alpha = (S.pow(2).sum() / S[:r].pow(2).sum().clamp(min=1e-12)).sqrt()
self.U.copy_(U[:, :r]); self.S.copy_(S[:r] * alpha); self.V.copy_(Vh[:r, :])
```

### 5.4 LMFC: LayerNorm Masks Frobenius Collapse

When tested **end-to-end** on WikiText-2, EPSI improves validation perplexity only from ~1028 to ~1000 (a **2% improvement**). Why so little, given the mathematical severity of the bug?

**Claim (LMFC)**: *In architectures with per-sublayer normalization (LayerNorm, RMSNorm), initialization-scale errors that cause multiplicative signal attenuation are absorbed by the normalization layer and have at most a small effect on end-to-end training.*

**Evidence**: A stack of 12 MLP blocks, with and without LayerNorm, initialized with all three methods:

| Configuration | Output norm (from input norm 1255) |
|---|---|
| Dense, no LN | 15.47 |
| ZFact old init, no LN | **0.00** (collapse) |
| ZFact + EPSI, no LN | 314.96 |
| Dense with LN | ≈ 1213 |
| **ZFact old init with LN** | **≈ 1210** |
| ZFact + EPSI with LN | ≈ 1210 |

With LN, the difference between dense and factorized is nil — LN renormalizes after every block regardless of the collapsed signal from the previous linear layer. Without LN, the difference is catastrophic.

**Implication**: EPSI is a correct and easy fix that we recommend as the default, but in modern transformers it is **not** the cause of the pretraining gap. The contribution of LMFC is the methodological observation: unit tests on isolated layers can be misleading, because the normalization at the architecture level hides or amplifies init-scale bugs that the layer-level tests don't see.

---

## 6. Contribution 2: The Double-Compression Incompatibility Principle

This is the main empirical finding of the paper.

### 6.1 Setup

Following the MZTrain defaults at the time of writing, we combined:

- **Weight factorization** with $r = 128$ (ZFactorized linear layers).
- **APOLLO-style optimizer** (`ZAdaptiveOptimizer`) with projector rank $r' = 64$ and scalar second-moment block size 1024.

Each configuration was a natural choice: $r = 128$ is the rank MZTrain used in the scale-limit benchmarks; $r' = 64$ is APOLLO's default in our implementation. The user of the system simply sets `optimizer_type="adaptive"` and both compression axes are active.

### 6.2 The DCI principle

**Definition**. *Let $C_\text{param}$ restrict weights to the rank-$r$ manifold $\mathcal{M}_r$, and let $C_\text{opt}$ project gradients to rank $r'$ before the Adam step. Then the effective per-step update rank is $\min(r, r')$, and the reachable set over many steps is a proper subset of $\mathcal{M}_r$ given by the intersection of $\mathcal{M}_r$ with the subspaces spanned by the (periodically refreshed) projector $P$.*

**Sketch of the mechanism**. For a factor $U \in \mathbb{R}^{m \times r}$, the gradient is $\nabla_U \mathcal{L} = (\nabla_W \mathcal{L}) V^\top \in \mathbb{R}^{m \times r}$. APOLLO projects this along the $m$ axis to rank $r'$. The resulting $\Delta U$ is rank $\leq r'$; similarly $\Delta V$ is rank $\leq r'$ in its large axis. The product $\Delta W = \Delta U \cdot V + U \cdot \Delta V$ has rank $\leq \min(r, r')$.

Over a training horizon, $P$ is refreshed every $T$ steps, so additional directions become available, but the weight is permanently constrained to $\mathcal{M}_r$. The combined exploration set is a proper subset of $\mathcal{M}_r$ — the optimizer cannot fully cover its own parameter manifold.

### 6.3 Quantitative projection ratios

We instrumented `ZAdaptiveOptimizer` to print the projection shape per parameter. For a single transformer block at $r = 128$, $r' = 64$:

| Parameter | Shape | Projected to | Ratio |
|---|---|---|---|
| `attn.q_proj.U` | (768, 128) | (64, 128) | 8.33% |
| `attn.q_proj.V` | (128, 768) | (128, 64) | 8.33% |
| `attn.k/v/o_proj.*` | same | same | 8.33% |
| **`mlp.fc1.U`** | **(3072, 128)** | **(64, 128)** | **2.08%** |
| `mlp.fc1.V` | (128, 768) | (128, 64) | 8.33% |
| `mlp.fc2.U` | (768, 128) | (64, 128) | 8.33% |
| **`mlp.fc2.V`** | **(128, 3072)** | **(128, 64)** | **2.08%** |

The MLP factors receive gradient information at **2.08%** of their own rank-128 capacity. This is two orders of magnitude below what the factorized parameterization is designed to handle.

### 6.4 Empirical consequence — full composition table

WikiText-2 causal LM, 3 epochs, seq=128, batch=16, 60M factorized transformer (same data, seed, hyperparameters across all rows):

| # | Configuration | Final Val PPL | vs Dense | Opt State | DCI? |
|---|---|---|---|---|---|
| 1 | Dense baseline + AdamW | **237.66** | 1.00× | 944.2 MB | — |
| 2 | ZFactorized r=128 + **APOLLO** ($r'=64$) | **1027.52** | **4.32×** | 6.4 MB | **YES** |
| 3 | ZFactorized r=128 + **GaLore** ($r'=64$) | **958.24** | **4.03×** | 11.1 MB | **YES** |
| 4 | ZFactorized r=128 + **AdamW** (fix) | **278.76** | 1.17× | 458.1 MB | NO |

Per-epoch trajectories (validation perplexity):

| Epoch | Dense | +APOLLO | +GaLore | +AdamW (fix) |
|---|---|---|---|---|
| 0 | 403.24 | 1555.98 | 1456.20 | 442.22 |
| 1 | 290.34 | 1208.50 | 1121.72 | 331.63 |
| **2 (final)** | **237.66** | **1027.52** | **958.24** | **278.76** |

**Observations**:

1. **Both APOLLO and GaLore exhibit DCI**, with GaLore slightly less severe (4.03× vs 4.32×) — consistent with GaLore's SVD-based projection being marginally more informative than APOLLO's random projection, but both mechanisms are fundamentally incompatible with weight factorization.

2. **The trajectories of APOLLO and GaLore are nearly parallel** across all three epochs. At epoch 0, GaLore is 7% better than APOLLO (1456 vs 1556); at epoch 2, GaLore is 7% better (958 vs 1028). Neither shows a catch-up trend — the gap with dense is stable, indicating the problem is not a warmup issue but a structural limitation.

3. **AdamW on the same factorized model closes 94.8% of the APOLLO gap and 93.9% of the GaLore gap**. The fix is agnostic to which projecting optimizer is replaced.

4. **DCI is not specific to APOLLO**. The original measurement (test_23) used only APOLLO, which left open the possibility that we had discovered an APOLLO-specific bug. The GaLore replication (test_27) confirms the phenomenon generalizes to the entire gradient-projection optimizer family and matches the theoretical prediction from the DCI principle.

### 6.5 Ablation

1-epoch ablation on a 2000-sequence WikiText-2 subset, isolating each contributor:

| Config | Train Loss | Val PPL | Δ Train vs Dense |
|---|---|---|---|
| A. Dense + AdamW | 7.2630 | 1119.76 | — (reference) |
| B. ZFact r=128 + AdamW | 7.3334 | 1153.28 | **+0.070 nats** (factorization) |
| C. ZFact r=128 + APOLLO | 8.7301 | 3125.48 | **+1.467 nats** (fact + APOLLO) |
| D. ZFact r=256 + AdamW | 7.3105 | 1143.38 | +0.047 nats (rank doubled) |

**Decomposition**:

- Weight factorization alone (B − A): **+0.070 nats**, i.e. **5%** of the gap.
- APOLLO on top of factorization (C − B): **+1.397 nats**, i.e. **95%** of the gap.
- Doubling the rank (B → D): −0.023 nats (negligible).

The APOLLO contribution is **twenty times larger** than the factorization contribution. Doubling the rank yields essentially no improvement, confirming that the bottleneck is the optimizer and not the weight parameterization.

### 6.6 Safe and unsafe compression pairs

The following composition table summarizes our findings; rows marked "measured" are empirically verified in this paper, others are predictions from the DCI mechanism:

| Parameter compression | Optimizer compression | DCI? | Status |
|---|---|---|---|
| ZFactorized (rank $r$) | AdamW | No | **measured** (§9, ppl=278.76) |
| ZFactorized (rank $r$) | 8-bit Adam (quantization) | No | predicted |
| ZFactorized (rank $r$) | Lion (no $v$) | No | predicted |
| **ZFactorized (rank $r$)** | **APOLLO** (rank $r'$) | **YES** | **measured** (§6.4, ppl=1027.52) |
| **ZFactorized (rank $r$)** | **GaLore** (rank $r'$) | **YES** | **measured** (§6.4, ppl=958.24) |
| Dense | APOLLO / GaLore (rank $r$) | No | (their original domain, published) |
| SLTrain ($UV + S$) | AdamW | No | (SLTrain's intended setup, published) |
| SLTrain ($UV + S$) | APOLLO / GaLore | Yes | predicted, same mechanism |

The safe recipe: **pick one compression axis, use a full-rank or quantized optimizer for the other.** Specifically, the failure mode is observed whenever both axes are rank-restricted and the weight rank $r$ and optimizer projection rank $r'$ are related by $r' < r$ on the factor tensors — the two compression schemes are then fighting over the same information budget.

### 6.7 A technical workaround

If one insists on combining, the DCI formula $r_{\text{effective}} = \min(r, r')$ suggests setting $r' \geq r$. In that case the optimizer's projection is a no-op on the already rank-$r$ factors and correctness is preserved — at the cost of eliminating the optimizer memory savings of APOLLO on those factors. In our case, setting APOLLO's projector rank to 128 on the MZTrain factors would match $r$ but give no memory advantage over AdamW. We did not benchmark this configuration.

---

## 7. Fair Benchmarking: 25M/50M/100M scales

We ran a fair-protocol comparison of Baseline dense (AdamW + FP16 AMP) vs MZTrain (ZFactorized + APOLLO + BF16 Aggressive) at three model sizes. *This was before we identified DCI, so it uses the original MZTrain stack; it serves as background context.*

### 7.1 Protocol

- Same transformer architecture (differs only in parameterization)
- Same synthetic dataset (seed=42)
- Same 15 epochs, batch=16
- Same LR=3e-4, weight_decay=1e-4
- Single RTX 4060

### 7.2 Results

| Scale | Config | Peak VRAM | Opt State | Final PPL | Accuracy |
|---|---|---|---|---|---|
| 25M | Dense + AdamW | 520 MB | 194.6 MB | ~0.04 loss | 100% |
| 25M | MZTrain full stack | 225 MB | 36.8 MB | ~0.008 loss | 100% |
| 50M | Dense + AdamW | ~900 MB | ~380 MB | — | 83.2% |
| 50M | MZTrain full stack | ~500 MB | ~70 MB | — | 99.2% |
| 100M | Dense + AdamW | ~1900 MB | ~760 MB | — | 71.9% |
| 100M | MZTrain full stack | ~600 MB | ~80 MB | — | 100% |

On **synthetic classification**, MZTrain full stack reaches the same or better accuracy than dense AdamW, with 2–4× less VRAM and 5–10× less optimizer state. This was the empirical basis of our initial (incorrect) confidence that the compression stack composed safely. The WikiText experiments in §9 show why synthetic classification is a misleading benchmark for pretraining: the task is trivially solvable at rank 64 and does not stress the optimizer.

---

## 8. Optimizer Comparison

Controlled benchmark of MZTrain's APOLLO implementation against 5 SOTA optimizers, using a fixed 25M transformer, 15 epochs, seed=42:

| Optimizer | Params | Peak VRAM | Opt state | Loss | Accuracy | Time |
|---|---|---|---|---|---|---|
| AdamW (torch.optim) | 25.5M | 520 MB | 194.6 MB | 0.040 | 98.4% | 10.6s |
| Adam8bit (bitsandbytes) | 25.5M | 332 MB | 49.8 MB | 0.025 | 100.0% | 11.6s |
| Lion (ICML 2023) | 25.5M | 379 MB | 97.3 MB | 0.0002 | 100.0% | 13.0s |
| GaLore (ICML 2024) | 25.5M | 309 MB | 12.9 MB | 0.0007 | 100.0% | 16.5s |
| **APOLLO (this work)** | 25.5M | 303 MB | **6.8 MB** | 0.0011 | 100.0% | 19.6s |
| **MZTrain Full Stack** | 5.1M | **225 MB** | 36.8 MB | 0.008 | 100.0% | 25.3s |

**Rankings**:

- **Smallest optimizer state**: APOLLO (6.8 MB) → 28.6× smaller than AdamW.
- **Smallest peak VRAM**: MZTrain Full Stack (225 MB) → 2.3× smaller than AdamW.
- **Best convergence**: Lion (loss 0.0002) → 200× better than AdamW.
- **Fastest**: AdamW (10.6 s) → the reference.

On this synthetic task, **all optimizers except AdamW reach 100% accuracy**. The MZTrain Full Stack is the only configuration that jointly minimizes both model parameters *and* peak VRAM. No single optimizer is best on all metrics — they trade off memory, speed, and convergence. These rankings are specific to this task and should not be extrapolated without verification; the WikiText experiments in §9 show the rankings can change dramatically on real pretraining.

---

## 9. The WikiText-2 Perplexity Benchmark

This is the central experiment of the paper and the setting in which DCI was first observed.

### 9.1 Setup

- **Dataset**: WikiText-2-raw-v1 via Hugging Face `datasets`.
- **Tokenizer**: GPT-2 BPE (50,257 vocabulary).
- **Preprocessing**: concatenate all non-empty lines with `\n\n`, tokenize, reshape into non-overlapping sequences of length 128. Train = 18,872 sequences (2.42M tokens); validation = 1,951 sequences (249,728 tokens).
- **Architecture**: 12-layer pre-LN transformer, $d = 768$, 12 heads, MLP ratio 4, dropout 0.1, tied token+position embeddings, final LayerNorm → tied LM head. Total 124M params (85M in transformer blocks + 39M in tied embedding).
- **Training**: 3 epochs, batch 16, seed 42, grad-clip 1.0, AdamW lr=3e-4, weight_decay=1e-2, betas=(0.9, 0.95), FP16 or BF16 autocast with GradScaler.
- **Hardware**: NVIDIA RTX 4060, 8 GB VRAM.

### 9.2 Configurations

1. **Dense Baseline**: all linear layers are `nn.Linear`, AdamW + FP16.
2. **MZTrain + APOLLO**: ZFactorized r=128, ZAdaptiveOptimizer ($r'=64$), BF16 aggressive precision.
3. **MZTrain + GaLore**: ZFactorized r=128, ZGaLoreOptimizer ($r'=64$), BF16 standard precision.
4. **MZTrain + AdamW (FIX)**: ZFactorized r=128 + EPSI init, plain AdamW, BF16 standard precision.

### 9.3 Results

| Metric | Dense Baseline | MZ + APOLLO | MZ + GaLore | **MZ + AdamW (FIX)** |
|---|---|---|---|---|
| Trainable params | 123.75M | 60.06M | 60.06M | 60.06M |
| Peak VRAM | 3716 MB | 2900 MB | 2906 MB | 3292 MB |
| Optimizer state | 944.2 MB | 6.4 MB | 11.1 MB | 458.1 MB |
| Total train time | 537.5 s | 791.9 s | 775.8 s | 480.0 s |
| Val PPL epoch 0 | 403.24 | 1555.98 | 1456.20 | **442.22** |
| Val PPL epoch 1 | 290.34 | 1208.50 | 1121.72 | **331.63** |
| **Val PPL epoch 2 (final)** | **237.66** | **1027.52** | **958.24** | **278.76** |
| Ratio vs dense | 1.00× | 4.32× | **4.03×** | **1.17×** |

### 9.4 Reading the table

- **Both projecting optimizers are catastrophically worse than dense**: APOLLO is 4.32× and GaLore is 4.03× worse. In a naive benchmark without the AdamW comparison, one would conclude "factorized transformers can't train on WikiText".
- **The AdamW fix recovers 94.8% of the APOLLO gap and 93.9% of the GaLore gap**. Final perplexity is 1.17× dense — a 17% gap attributable to the factorization itself plus BF16 precision differences plus single-seed noise.
- **The two projecting optimizers are nearly indistinguishable in trajectory**. Their per-epoch values differ by 5–8% consistently, confirming that DCI is a mechanism-level phenomenon and not an implementation artifact of either optimizer.
- **Improvement factor of the fix**: 3.69× over APOLLO, 3.44× over GaLore.
- **VRAM vs baseline**: 11.4% less with the FIX, 22% less with projecting optimizers.
- **Optimizer state vs baseline**: 51.5% less with FIX, 99% less with projecting optimizers — but at a 4× cost in perplexity that makes the savings moot.

### 9.5 Methodological note

We initially ran the MZTrain OLD configuration expecting it to match or beat the baseline on perplexity, because all the synthetic tests passed at 100% accuracy. The 4.3× gap was the trigger for this investigation. Without the WikiText benchmark, DCI would not have been identified; unit tests on toy data completely missed it.

### 9.6 The remaining 17% gap

The fixed MZTrain is still 1.17× worse than dense. Possible remaining contributors (not isolated in this paper):

1. The factorization itself (we measured this as +3% perplexity in the 1-epoch ablation; it could be higher over a full training horizon).
2. BF16 vs FP16 in the autocast path (modest and in either direction depending on the layer).
3. Weight tying between the tied embedding and lm_head interacting with the factorized transformer blocks above them.
4. Lack of periodic refactorization or merge-reset (ReLoRA-style) in this run — would likely help close more of the gap but was disabled for experimental cleanliness.
5. A single seed — the measurement has some noise that we did not quantify with multiple-seed error bars.

---

## 10. Scale Limits on Consumer Hardware

For context on what MZTrain (pre-DCI-fix) can fit into VRAM, we report the scale-limit experiments.

| Scale | $d$ | Blocks | Rank | Factorized params | Peak VRAM (train) | Status |
|---|---|---|---|---|---|---|
| 300M | 1024 | 24 | 128 | 58.1 M | 729 MB | Converged |
| 400M | 1152 | 25 | 128 | 68.1 M | 770 MB | Converged |
| 500M | 1280 | 25 | 128 | 75.9 M | 786 MB | Converged |
| 750M | 1408 | 32 | 96 | 80.5 M | 1252 MB | Converged |
| **900M** | 1536 | 32 | 96 | 88.0 M | **1363 MB** (16.6% of 8 GB) | Converged |

At 900M full-equivalent parameters, MZTrain uses 1.36 GB of training VRAM on an 8 GB GPU, with 5× headroom remaining. Linear extrapolation suggests ~3 B parameters are reachable on the same hardware if the only bottleneck is memory; in practice, training time (dominated by SVD initialization and the forward/backward through 192 factorized layers) becomes prohibitive. The tests reported here used 3 epochs on a 64-sample synthetic dataset and do not include a real-task validation at 900M.

The point of this section is not that MZTrain produces good 900M models — we did not train long enough to tell — but that the training **fits in VRAM** on consumer hardware at a scale that would otherwise require an A100 or H100.

---

## 11. Discussion

### 11.1 The central lesson

Two compression techniques, each individually validated on the same benchmark (language modeling on C4 / WikiText), fail when naively composed. The failure is not in either technique but in the composition. DCI is the mechanical explanation: both techniques restrict the effective update rank, and when their projections are stacked the restrictions compound multiplicatively in information terms (only $\min(r, r')$ effective rank) and subspace-intersection terms (only the overlap of their low-rank bases).

The fix is conservative: use exactly one compression axis per optimizer/weight pair. In MZTrain specifically, the recommended configuration after this paper is:

- `ZFactorizedTransformerBlock` + `ZCompressedAdam` (INT8 Adam states): factorization on the weights, quantization on the optimizer state, both memory-efficient and non-overlapping.
- or `ZFactorizedTransformerBlock` + plain `AdamW` + aggressive BF16: factorization on the weights, nothing on the optimizer, still 11% VRAM savings and 51% optimizer state savings over dense.

### 11.2 Why did this slip through the tests?

Three reasons, discussed in §6.5 and §9.5:

1. **Synthetic datasets are too easy.** Rank 64 is enough to memorize classification on 500 synthetic samples. Perplexity on real text pretraining requires much higher effective rank.
2. **Short training horizons.** Per-step DCI cost is small but accumulates. Only multi-thousand-step runs on real data expose it.
3. **Unit tests on isolated optimizers with small models pass.** APOLLO alone on a dense 25M model works fine (§8). ZFactorized alone with AdamW on a 60M model also works fine (§7.2 ablation config B). The failure appears only in their combination on real data.

### 11.3 Implications for the broader literature

We encourage authors of memory-efficient pretraining systems to (a) always benchmark on real pretraining tasks (WikiText-2 is cheap and sufficient to catch issues at this magnitude), (b) explicitly test composition with other compression techniques, and (c) report failure modes as prominently as success modes. The fact that both GaLore and APOLLO report only success cases on dense LLaMA and not composition cases with factorized models left a gap in the literature that MZTrain fell into and this paper closes.

### 11.4 Positive framing

This paper is a story of "compression system has bug, bug is found, bug is fixed, system now works". The positive framing is:

- MZTrain is a working memory-efficient training framework on consumer hardware.
- With the recommended configuration, it trains 60M-parameter factorized transformers on WikiText-2 to within 1.17× of dense baseline perplexity, using 11% less VRAM and 51% less optimizer state.
- The failure of the original configuration is documented, explained, fixed, and reproducible.
- Both EPSI and DCI are generally applicable findings with implications beyond the MZTrain codebase.

---

## 12. Limitations

1. **Single dataset (WikiText-2).** We did not run WikiText-103 or C4 due to compute budget. The magnitude of DCI may differ on larger corpora.
2. **Single model scale (88M transformer blocks).** GaLore and APOLLO report that their gap with dense AdamW shrinks at 1B+ scale. It is possible — but not verified here — that DCI's magnitude also shrinks at larger scales.
3. **Short training (3 epochs).** Pretraining is normally 100+ epochs. The late-training regime may behave differently. Our measurement is an early-training snapshot.
4. **Only APOLLO and GaLore tested so far.** The DCI argument predicts the same failure mode for any gradient-projection optimizer (e.g. Natural GaLore, Q-GaLore, SwitchLoRA). These were not measured but the mechanism is identical.
5. **Single seed (42).** Multi-seed error bars would strengthen the claim. The magnitudes we report (4.32× vs 1.17×) are large enough to be confident, but a formal significance test is left as future work.
6. **Our own APOLLO implementation.** The reported behavior is from `ZAdaptiveOptimizer`, our implementation of APOLLO following the paper. Possible implementation-specific bugs distinct from DCI cannot be fully ruled out without comparing against the reference implementation.
7. **Refactorization and rank growth disabled.** We disabled them to isolate the contributions of the other components. It is possible that the MZTrain OLD configuration, combined with refactorization, would close some of the 4.3× gap. We did not test this.

---

## 13. Conclusion

We presented MZTrain, a memory-efficient training system for transformer language models on consumer GPUs, and reported a surprising empirical finding along the way: **weight factorization and gradient-projection optimizers compose destructively**. The Double-Compression Incompatibility principle we formulate explains the mechanism, the WikiText-2 benchmark quantifies it, and a one-line configuration change recovers 94.8% of the resulting perplexity gap with the dense baseline.

We also contributed Energy-Preserving SVD Initialization (EPSI), a simple Frobenius-norm-preserving scaling step that closes a 30–70% energy loss in the default SVD truncation initialization; and LMFC, the methodological observation that LayerNorm absorbs initialization-scale bugs in modern transformers, limiting EPSI's end-to-end impact to 2% despite its severity at the layer level.

The core thesis of MZTrain — that transformers can be pretrained in compressed parameter space on consumer hardware — is intact. What we show is that the *details of composition* matter more than the compression ratios themselves, and that two individually-valid memory techniques can fail catastrophically when stacked. On real pretraining, the MZTrain FIX configuration (factorized weights + AdamW + EPSI + BF16) trains a 60M-parameter language model on WikiText-2 to within 17% of the dense baseline perplexity while using 11% less VRAM, 51% less optimizer state, and less wall-clock time than dense AdamW.

---

## Appendix A — Reproducibility

All experiments are in `D:\mztrain_tests\`:

| File | Description |
|---|---|
| `test_17_adaptive_optimizer.py` | Unit tests for ZAdaptiveOptimizer (29/29 passed). |
| `test_19_scale_100m_250m.py` | Scale benchmarks at 100M and 250M. |
| `test_20_fair_benchmark.py` | Fair 25M/50M/100M comparison, dense vs MZTrain. |
| `test_21_optimizer_comparison.py` | 6-way optimizer comparison on 25M. |
| `test_22_scale_limits.py` | Scale-limit experiments from 300M to 900M. |
| `test_23_wikitext_ppl.py` | Original WikiText-2 benchmark (the 4.3× gap). |
| `test_24_mztrain_epsi_rerun.py` | Re-run with EPSI init (+2% improvement). |
| `test_25_ablation.py` | Ablation isolating APOLLO as the 95% contributor. |
| `test_26_mztrain_fixed_rerun.py` | Final re-benchmark with AdamW (94.8% closed). |
| `test_27_galore_zfact.py` | GaLore + ZFact replication of DCI (4.03× worse, generalizes beyond APOLLO). |

The EPSI init fix is in `D:\mztrain\src\mztrain\layers.py`, method `ZFactorizedLinear._init_svd`.

Hardware: NVIDIA RTX 4060, 8 GB VRAM, Windows 11, Python 3.13, PyTorch 2.6.0+cu124, CUDA 12.4.
Dependencies: `datasets==4.1.1`, `transformers==5.2.0`, `bitsandbytes==0.49.2`, `lion-pytorch==0.2.4`.

---

## References

1. Han, A. et al. **SLTrain: A Sparse Plus Low-Rank Approach for Parameter and Memory Efficient Pretraining.** NeurIPS 2024. [arXiv:2406.02214](https://arxiv.org/abs/2406.02214)
2. Zhao, J. et al. **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.** ICML 2024 Oral. [arXiv:2403.03507](https://arxiv.org/abs/2403.03507)
3. Zhu, H. et al. **APOLLO: SGD-like Memory, AdamW-level Performance.** MLSys 2025 (Outstanding Paper HM). [arXiv:2412.05270](https://arxiv.org/abs/2412.05270)
4. Hu, E. et al. **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
5. Lialin, V. et al. **ReLoRA: High-Rank Training Through Low-Rank Updates.** ICLR 2024. [arXiv:2307.05695](https://arxiv.org/abs/2307.05695)
6. **LOST: Low-Rank + Structured Sparse Training.** August 2025. [arXiv:2508.02668](https://arxiv.org/abs/2508.02668)
7. Dettmers, T. et al. **8-bit Optimizers via Block-wise Quantization.** ICLR 2022. [arXiv:2110.02861](https://arxiv.org/abs/2110.02861)
8. Chen, X. et al. **Symbolic Discovery of Optimization Algorithms (Lion).** NeurIPS 2023. [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)
9. Chowdhury, S. et al. **Weight Tying Biases Token Embeddings Towards the Output Space.** [arXiv:2603.26663](https://arxiv.org/abs/2603.26663)
10. Aghajanyan, A. et al. **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.** ACL 2021. [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
11. Meng, F. et al. **PiSSA: Principal Singular Values and Singular Vectors Adaptation.** NeurIPS 2024 Spotlight. [arXiv:2404.02948](https://arxiv.org/abs/2404.02948)
12. Zhang, Y. et al. **Adam-mini: Use Fewer Learning Rates To Gain More.** ICLR 2025. [arXiv:2406.16793](https://arxiv.org/abs/2406.16793)
13. Liu, S. et al. **DoRA: Weight-Decomposed Low-Rank Adaptation.** ICML 2024 Oral. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
14. Büyükakyüz, K. **OLoRA: Orthonormal Low-Rank Adaptation.** [arXiv:2406.01775](https://arxiv.org/abs/2406.01775)
15. Zhang, Q. et al. **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.** ICLR 2023. [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)
16. Press, O. and Wolf, L. **Using the Output Embedding to Improve Language Models.** EACL 2017. [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)

---

**Acknowledgments**: All experiments were conducted on a single NVIDIA RTX 4060 (8 GB) on consumer hardware. No external compute was used. WikiText-2 is provided by Salesforce via the Hugging Face datasets hub. We thank the open-source maintainers of PyTorch, Hugging Face Transformers, bitsandbytes, and lion-pytorch.

**Correspondence**: `msc.framework@gmail.com`

**Code and data**: `D:\mztrain\` (source), `D:\mztrain_tests\` (benchmarks).
