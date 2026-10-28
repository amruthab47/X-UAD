# X-UAD — results comparison

**Read this first.** Numbers from different papers are measured on different datasets and are *not* comparable. The only rigorous improvement figures are in Section 1, where every model was trained by us on identical data under an identical budget. Section 3 lists published numbers as *context for the pattern*, not as a head-to-head.

## 1. Controlled comparison — same data, same budget, same code

### 1a. vs ViT-Tiny — the plain-transformer control

Same token count (49 vs 49) and same depth; the difference is that ViT tokenises raw 16x16 patches while X-UAD tokenises a central-difference stem. ViT has half the parameters, so this is a design comparison, not a capacity-matched one.

| Metric | X-UAD | ViT-Tiny | Abs. | Relative | |
|---|---|---|---|---|---|
| ACER | 1.46% | 49.75% | -48.29 | +97.1% | **better** |
| APCER | 2.08% | 60.67% | -58.58 | +96.6% | **better** |
| BPCER | 0.83% | 38.83% | -38.00 | +97.9% | **better** |
| EER | 0.76% | 34.65% | -33.90 | +97.8% | **better** |
| AUC | 99.89% | 73.43% | +26.46 | +36.0% | **better** |
| ACC | 94.53% | 51.27% | +43.26 | +84.4% | **better** |
| macroF1 | 94.53% | 46.04% | +48.50 | +105.3% | **better** |

### 1b. vs CDCN++ — the strongest ISO-error baseline

| Metric | X-UAD | CDCN++ | Abs. | Relative | |
|---|---|---|---|---|---|
| ACER | 1.46% | 0.71% | +0.75 | -105.9% | worse |
| APCER | 2.08% | 0.92% | +1.17 | -127.3% | worse |
| EER | 0.76% | 0.49% | +0.26 | -53.0% | worse |
| AUC | 99.89% | 99.92% | -0.04 | -0.0% | worse |
| ACC | 94.53% | 96.18% | -1.65 | -1.7% | worse |
| ECE | 1.94% | 3.44% | -1.50 | +43.6% | **better** |
| Localisation IoU (region-level) | 77.77% | 73.51% | +4.26 | +5.8% | **better** |
| Localisation IoU — 3D mask | 83.57% | 79.55% | +4.02 | +5.0% | **better** |
| Localisation IoU — face-swap | 78.09% | 76.79% | +1.30 | +1.7% | **better** |
| Localisation IoU — reenactment | 71.64% | 64.18% | +7.46 | +11.6% | **better** |

### 1c. vs ResNet-18 — the strongest accuracy baseline

| Metric | X-UAD | ResNet-18 | Abs. | Relative | |
|---|---|---|---|---|---|
| ACER | 1.46% | 0.92% | +0.54 | -59.1% | worse |
| ACC | 94.53% | 96.79% | -2.26 | -2.3% | worse |
| macroF1 | 94.53% | 96.79% | -2.26 | -2.3% | worse |
| ECE | 1.94% | 2.11% | -0.17 | +8.0% | **better** |
| Parameters (M) | 1.29 | 1.59 | -0.30 | +18.8% | **better** |
| GMACs | 0.12 | 0.37 | -0.25 | +67.6% | **better** |

### 1d. Unseen attack families (leave-one-attack-out, mean of 4 folds)

| Metric | X-UAD | CDCN++ | rel. | ResNet-18 | rel. |
|---|---|---|---|---|---|
| ACER | 31.55% | 29.10% | -8.4% | 40.75% | +22.6% |
| 8-way ACC | 65.22% | 65.69% | -0.7% | 57.38% | +13.7% |
| Unseen detection rate | 77.46% | 70.19% | +10.4% | 74.69% | +3.7% |
| Novelty AUROC | 63.09% | 63.15% | -0.1% | 50.82% | +24.1% |

### 1e. Do the novel components earn their place?

| Component removed | Metric | Full | Ablated | Abs. | Relative |
|---|---|---|---|---|---|
| Pseudo-unseen suppression | novelty_AUROC | 87.86% | 82.43% | +5.44 | +6.6% |
| Pseudo-unseen suppression | ACC | 73.34% | 68.12% | +5.22 | +7.7% |
| Adaptive Quantum Attention | AUC | 91.41% | 89.47% | +1.94 | +2.2% |
| Adaptive Quantum Attention | ACC | 73.34% | 71.40% | +1.95 | +2.7% |
| Adaptive Quantum Attention | ECE | 6.26% | 7.36% | -1.10 | +14.9% |

## 2. Capabilities no baseline reports at all

| Capability | X-UAD | ResNet-18 | ViT-Tiny | CDCN++ |
|---|---|---|---|---|
| Calibrated uncertainty | Dirichlet vacuity | 1-max softmax | 1-max softmax | 1-max softmax |
| Explicit unknown-attack score | yes (2 of 4 folds usable) | no | no | no |
| Evidence localisation for *digital* attacks | yes | no | no | depth proxy only |
| Natural-language explanation | yes, verified | no | no | no |
| Attack type + confidence + risk band | yes | no | no | no |

## 3. Published numbers — context, NOT a head-to-head

Different corpora, different difficulty. Listed only to show that the *pattern* our benchmark produces matches the published one: the seen-attack protocol saturates for everyone, the unseen-attack protocol collapses for everyone.

| Work | Venue | Dataset | Seen-attack ACER | Unseen-attack ACER |
|---|---|---|---|---|
| ResNet-50 | IJCAI'24 baseline | UniAttackData | 1.35% | 34.60 ± 5.31% |
| ViT-B/16 | IJCAI'24 baseline | UniAttackData | 5.92% | 33.69 ± 9.33% |
| CDCN | IJCAI'24 baseline | UniAttackData | 1.40% | 34.33 ± 0.66% |
| Auxiliary (Depth) | IJCAI'24 baseline | UniAttackData | 1.13% | 42.98 ± 6.77% |
| FFD | IJCAI'24 baseline | UniAttackData | 2.01% | 44.20 ± 1.32% |
| **UniAttackDetection** | IJCAI 2024 | UniAttackData | **0.52%** | **22.42 ± 10.57%** |
| Facevengers (CLIP+VAE, LoRA ViT-L) | ICCVW 2025 | UniAttackData+ | 0.144% | — |
| TeleAI (CLIP contrastive) | ICCVW 2025 | UniAttackData+ | 0.178% | — |
| Median challenge entrant | ICCVW 2025 | UniAttackData+ | ~8-26% | — |
| **X-UAD (this work)** | — | LFW-Sim (ours) | **1.46%** | **31.55%** |
| CDCN++ (our re-impl.) | — | LFW-Sim (ours) | 0.71% | 29.10% |
| ResNet-18 (our re-impl.) | — | LFW-Sim (ours) | 0.92% | 40.75% |
| ViT-Tiny (our re-impl.) | — | LFW-Sim (ours) | 49.75% | not run |
