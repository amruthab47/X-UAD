# X-UAD — experimental results

All models trained from scratch, batch 48, AdamW one-cycle, 112x112, CPU only, seed 0. Within each table every model gets the same budget: **P1 (and P3) = 10 epochs on the full 19,200-sample train split; P2 and the open-set ablation = 5 epochs on a 4,800-sample subset (4,382 after removing the held-out family); component ablation = 5 epochs on 3,592 samples** (per the `epochs`/`n_train` fields in `results/*.json`). P1 and P2 numbers are therefore not comparable with each other. The operating threshold is the validation equal-error point, applied unchanged to test.

## Protocol P1 — seen attacks (closed set), identity-disjoint, ID-consistent

| Method | ACER | APCER | BPCER | AUC | EER | ACC | macroF1 | ECE |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.92 | 1.33 | 0.50 | 99.92 | 0.49 | 96.79 | 96.79 | 2.11 |
| ViT-Tiny | 49.75 | 60.67 | 38.83 | 73.43 | 34.65 | 51.27 | 46.04 | 0.67 |
| CDCN++ | 0.71 | 0.92 | 0.50 | 99.92 | 0.49 | 96.18 | 96.19 | 3.44 |
| **X-UAD (ours)** | 1.46 | 2.08 | 0.83 | 99.89 | 0.76 | 94.53 | 94.53 | 1.94 |

Paired significance (McNemar exact, same 9,600 test samples): X-UAD's binary deficit vs CDCN++ is p<1e-4 (51 vs 17 discordant); the 8-way deficit is p=1.1e-12 vs CDCN++ and p=4.2e-22 vs ResNet-18. BPCER alone is not significant (Wilson 95% CIs overlap). The deficit is real and is the price of the uncertainty machinery.

## Spoof-evidence localisation (IoU %, protocol P1)

| Method | print | replay | mask3d | faceswap | reenact | synth | mean_localisable |
|---|---|---|---|---|---|---|---|
| CDCN++ | 95.2 | 97.5 | 79.5 | 76.8 | 64.2 | 98.9 | 73.5 |
| **X-UAD** | 97.1 | 98.4 | 83.6 | 78.1 | 71.6 | 98.0 | 77.8 |

(ResNet-18 and ViT-Tiny have no localisation output. Ground truth on this benchmark is the mask our own synthesiser injected — see the limitation note.)

## Protocol P2 — leave-one-attack-out (held-out family never seen in training)


**ACER (%) lower=better**

| Method | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| ResNet-18 | 51.96 | 30.38 | 49.96 | 30.71 | 40.75 |
| CDCN++ | 25.67 | 39.92 | 25.25 | 25.58 | 29.10 |
| **X-UAD** | 26.79 | 21.42 | 38.33 | 39.67 | 31.55 |

**8-way ACC (%) higher=better**

| Method | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| ResNet-18 | 53.94 | 62.66 | 51.39 | 61.53 | 57.38 |
| CDCN++ | 60.62 | 65.07 | 73.23 | 63.84 | 65.69 |
| **X-UAD** | 60.95 | 69.01 | 73.34 | 57.59 | 65.22 |

**Unseen detection rate (%) higher=better**

| Method | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| ResNet-18 | 100.00 | 61.67 | 37.17 | 99.92 | 74.69 |
| CDCN++ | 93.08 | 34.25 | 60.33 | 93.08 | 70.19 |
| **X-UAD** | 88.00 | 88.08 | 35.58 | 98.17 | 77.46 |

**Novelty AUROC (%) higher=better**

| Method | replay | mask3d | faceswap | adversarial | Mean |
|---|---|---|---|---|---|
| ResNet-18 | 35.29 | 56.69 | 74.40 | 36.88 | 50.82 |
| CDCN++ | 49.28 | 66.57 | 81.66 | 55.08 | 63.15 |
| **X-UAD** | 71.03 | 50.41 | 87.86 | 43.04 | 63.09 |

The novelty channel is strong on replay (71.0) and face-swap (87.9) and at or below chance on 3D mask (50.4) and adversarial (43.0). It works on some families, not all.

## Protocol P3 — cross-domain transfer (train Domain A, test Domain B, no adaptation)

| Method | ACER | APCER | BPCER | AUC | EER | ACC | ECE |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 49.79 | 0.00 | 99.57 | 46.78 | 51.87 | 2.29 | 83.26 |
| ViT-Tiny | 50.00 | 0.86 | 99.14 | 50.18 | 50.23 | 10.31 | 82.47 |
| CDCN++ | 45.79 | 2.29 | 89.29 | 65.96 | 37.93 | 18.67 | 60.09 |
| **X-UAD** | 49.71 | 0.29 | 99.14 | 50.90 | 49.98 | 3.65 | 81.61 |

Evaluated on 4900 pure Domain-B samples from the P1 checkpoints. The adversarial class is excluded: it was crafted from Domain-A images against a Domain-A surrogate, so it is not cross-domain data.

**Negative result, reported in full**: under a joint colour + alignment + resolution + compression shift, every method collapses (BPCER 89-100%). CDCN++ degrades most gracefully by AUC; X-UAD is at chance. None of the mechanisms in this work addresses capture-domain shift.

## Component ablation (reduced shared budget: 3,592 training crops, 5 epochs)

| Variant | ACER | AUC | ACC | macroF1 | ECE | IoU_loc |
|---|---|---|---|---|---|---|
| X-UAD (full) | 38.00 | 87.57 | 69.62 | 69.65 | 2.66 | 64.6 |
| w/o Adaptive Quantum Attention | 32.75 | 91.05 | 67.45 | 65.04 | 3.62 | 63.4 |
| w/o evidential head (softmax) | 30.54 | 91.06 | 72.07 | 71.88 | 3.27 | 65.0 |
| w/o pseudo-unseen suppression | 26.04 | 91.08 | 72.90 | 72.85 | 2.83 | 64.2 |
| w/o spectral branch | 31.13 | 88.44 | 70.67 | 68.92 | 3.12 | 64.5 |
| w/o central-difference stem | 49.08 | 73.64 | 48.62 | 42.16 | 3.03 | 52.6 |
| w/o localisation supervision | 33.58 | 89.17 | 70.56 | 68.78 | 2.41 | 34.3 |

This budget is smaller than P1's, so these rows compare with each other, not with the P1 table.

## Open-set ablation (face-swap leave-one-attack-out fold)

| Variant | ACER | AUC | ACC | ECE | unseen_detect_rate | novelty_AUROC |
|---|---|---|---|---|---|---|
| Full X-UAD | 38.33 | 91.41 | 73.34 | 6.26 | 35.58 | 87.86 |
| w/o PUES | 36.17 | 86.33 | 68.12 | 6.78 | 45.50 | 82.43 |
| w/o AQA | 39.83 | 89.47 | 71.40 | 7.36 | 32.75 | 87.89 |

## Explanation faithfulness (deletion AUC lower=better)

| Explanation channel | Deletion | Insertion | Δ deletion vs control |
|---|---|---|---|
| X-UAD localisation map | 0.915 | 0.917 | -0.031 |
| AQA attention rollout | 0.932 | 0.921 | -0.015 |
| Quantum evidence magnitude | 0.906 | 0.899 | -0.040 |
| Random (control) | 0.946 | 0.946 | -- |

All three intrinsic channels beat the control on deletion. Insertion does not separate them — the attack score is driven by whole-crop texture statistics, so revealing any large fraction restores it.

## Adaptive Born-channel gate, by attack family

| Family | bonafide | print | replay | mask3d | faceswap | reenact | synth | adversarial |
|---|---|---|---|---|---|---|---|---|
| mean gate g | 0.309 | 0.434 | 0.231 | 0.287 | 0.267 | 0.267 | 0.284 | 0.394 |

Per-layer: [0.272, 0.003, 0.418, 0.537]; overall mean 0.308. The gate opens widest on print and adversarial — the two families whose evidence is most periodic / highest-frequency.

## Cost

| Method | Params (M) | GMACs (conv+linear only) | CPU latency (ms) |
|---|---|---|---|
| resnet18 | 1.59 | 0.368 | 16.8 |
| vit_tiny | 0.64 | 0.031 | 2.9 |
| cdcn | 0.53 | 0.070 | 13.1 |
| xuad | 1.29 | 0.119 | 16.9 |

GMAC counts exclude attention matmuls, so X-UAD's figure is understated relative to the pure CNNs.

