# X-UAD — P1 (closed-set) vs. P2.replay (leave-one-attack-out) comparison

Both runs use the identical model, code, and 33,600-sample benchmark; the only difference is that P2.replay excludes the `replay` class from training and validation, keeping it only in the 9,600-sample test set. Source runs:

- **P1**: `results_full_retry/xuad_P1_full_retry.json` / `_pred.npz`
- **P2.replay**: `results_P2_replay/xuad_P2_replay.json` / `_pred.npz`

All numbers below were recomputed directly from each run's saved `_pred.npz` (not copied from memory), independently confirming the previously reported figures.

## Confusion-matrix integrity check

Before comparing anything, the P2.replay confusion matrix was recomputed from `xuad_P2_replay_pred.npz` with explicit row sums:

```
true\pred  bonafi   print  replay  mask3d  facesw  reenac   synth  advers   ROWSUM
bonafide     1068      12       0       6      70      25      18       1     1200
print           6    1137       0       1      42      13       0       1     1200
replay        511     232       0       8     397      29      22       1     1200
mask3d          7       1       0    1082       2       1     107       0     1200
faceswap       77      45       0       5     943     129       1       0     1200
reenact         9      17       0       1     149    1024       0       0     1200
synth          28       0       0     135      11       3    1023       0     1200
adversarial      0       0       0       0       0       0       0    1200     1200
```

Every row sums to exactly 1,200. The adversarial row is `[0,0,0,0,0,0,0,1200]` — not all-zero; every adversarial test sample lands on the diagonal (1,200/1,200 correct), consistent with recall=1.0 and independently cross-checked against `sklearn.metrics.recall_score` (returns 1.0 exactly).

## Summary metrics

| Metric | P1 (closed-set) | P2.replay (unseen) | Δ (P2 − P1) |
|---|---|---|---|
| AUC | 99.880 | 96.342 | −3.539 |
| ACER | 2.417 | 30.375 | **+27.958** |
| APCER | 3.083 | 56.500 | **+53.417** |
| BPCER | 1.750 | 4.250 | +2.500 |
| EER | 1.494 | 8.661 | +7.167 |
| 8-class ACC | 89.490% | 77.885% | −11.605 pp |
| macro-F1 | 89.476 | 73.244 | −16.232 |
| unseen_AUROC_vacuity | n/a (all classes seen) | 80.432 | — |
| unseen_detect_rate | n/a | 43.500% | — |
| seen_detect_rate | n/a | 95.736% | — |

## Class-wise F1 comparison

| Class | P1 F1 | P2.replay F1 | Δ |
|---|---|---|---|
| bonafide | 0.9597 | 0.7350 | **−0.2247** |
| print | 0.9036 | 0.8601 | −0.0436 |
| **replay** | 0.9158 | **0.0000** | **−0.9158** |
| mask3d | 0.8222 | 0.8876 | +0.0655 |
| faceswap | 0.8335 | 0.6702 | **−0.1633** |
| reenact | 0.8916 | 0.8449 | −0.0467 |
| synth | 0.8342 | 0.8629 | +0.0287 |
| adversarial | 0.9975 | 0.9988 | +0.0012 |

Two classes (mask3d, synth) actually improved slightly in P2.replay — most plausibly ordinary run-to-run training variance rather than a systematic effect, since the model had one fewer class competing for capacity. Adversarial is essentially unaffected either way, since it remains fully trained in both runs.

## Confusion patterns — what changed

In P1, every class's confusion is spread thinly across several other classes (largest single confusion: mask3d→synth at 22.0%). In P2.replay, one destination dominates everything: **511/1,200 (42.6%) of true replay samples are misclassified as bonafide**, and a further **397/1,200 (33.1%) as faceswap** — together accounting for 76% of the held-out class's mass. This also explains the two largest class-wise F1 drops that aren't replay itself: bonafide's precision collapses from routing in 511 replay false positives, and faceswap's precision is similarly damaged by 397 misrouted replay samples. These aren't independent weaknesses — they're both downstream symptoms of the held-out class needing to go *somewhere*, and the model routing most of it toward the two classes replay is visually/statistically closest to (a clean capture that "looks genuine" when the moiré/glare artefacts are subtle, or a digital-looking artefact that resembles a blending/manipulation boundary).

## Research interpretation

### The closed-set vs. unseen-attack performance gap
ACER jumps nearly 13× (2.4 → 30.4) and APCER jumps over 18× (3.1% → 56.5%) purely from withholding one of eight classes from training — while every other class's training data, hyperparameters, and code are identical to P1. This quantifies, on your own real dataset rather than a synthetic worst case, exactly the generalization gap the X-UAD paper's `RESULTS.md` reports for its own P2 folds: a detector that is excellent on seen attacks is not automatically robust to attack types it has never encountered, even within the same broad "physical presentation attack" superclass (replay is physical, like print and mask3d, both of which the model still handles well).

### The 43.5% unseen detection rate
This is the operational number: at the validation-derived decision threshold, only 43.5% of held-out replay attacks are flagged as an attack at all — the other 56.5% (matching `per_type_apcer["replay"]=56.5`, an internal consistency check that these two numbers sum to 100%) sail through as accepted. In a real deployment, this is the metric that matters most: it says a replay-style attack that the system had never been trained on would be caught less than half the time by the raw decision rule alone — a materially weaker guarantee than the 95.7% `seen_detect_rate` on classes the model did train on.

### The 80.43 unseen vacuity AUROC
This is a *different* question from detection rate: not "does the raw attack score cross the threshold," but "does the model's own uncertainty (vacuity) rank held-out replay samples as more 'unfamiliar' than the attack classes it was trained on." An AUROC of 80.4 says yes, reasonably well — the Dirichlet evidential head has learned a genuinely useful signal that replay doesn't fit the trained taxonomy, even in the many cases where the hard classification decision gets it wrong. This is the mechanism the paper's pseudo-unseen-evidence-suppression (PUES) term is specifically designed to produce, and it's working directionally as intended for this fold. The gap between this fairly strong 80.4 and the weak 43.5% detection rate is itself the interesting finding: uncertainty and hard-decision correctness are not the same thing, and a system that surfaces vacuity to a human reviewer (rather than only a binary accept/reject) would be substantially more useful here than the raw threshold decision alone.

### The increased vacuity for unseen replay
Mean vacuity on true replay samples in this run is **0.331**, well above the overall P1 mean of 0.185 (and above this run's own seen-class average). This is the same signal as the AUROC finding, viewed as a raw magnitude rather than a ranking: the model is, on average, noticeably less confident about replay samples than about classes it trained on — consistent with (though numerically distinct from) the "unseen_AUROC_vacuity" result, and further evidence that the uncertainty channel is behaviorally different for genuinely novel inputs, not just noisy.

### Where the misclassified replay samples go
The dominant confusion (511/1,200 → bonafide) is the more concerning of the two failure modes, since it means a replay attack is most often mistaken for a genuine, harmless capture — a false-negative in the strictest security sense. The secondary confusion (397/1,200 → faceswap) is comparatively lower-stakes operationally (still flagged as *an* attack, just mislabeled), but it corrupts faceswap's own class-wise precision and would mislead any downstream logic that acts differently per attack type (e.g., the risk-severity weighting, which assigns faceswap a much higher severity than replay would have received). Both patterns point to the same underlying cause: replay's forensic signature (subtle moiré/banding/glare, generally lower amplitude than the other physical-attack artefacts) is the class whose evidence looks least distinctive to a model that has never had to learn to isolate it, so it collapses toward whichever trained class's decision boundary is nearest in the model's learned feature space.

## Files referenced

| File | Role |
|---|---|
| `results_full_retry/xuad_P1_full_retry.json`, `_pred.npz` | P1 source (read-only, unmodified) |
| `results_P2_replay/xuad_P2_replay.json`, `_pred.npz` | P2.replay source (read-only, unmodified) |
| `results_P2_replay/xuad_P1_vs_P2_replay_comparison.md` | this document |
