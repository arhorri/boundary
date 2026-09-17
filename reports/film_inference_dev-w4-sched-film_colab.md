# FiLM inference-mode comparison -- dev-w4-sched-film (colab)

Held-out dataset: **uhcs2**, which has no FiLM embedding of its own. Three ways to condition it at inference, scored on the same validation tiles:

| mode | pixel Dice | verdict | ARI | VI | PQ | pred/true frac |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 0.2516 | MISPLACED | 0.145 | 1.918 | 0.029 | 0.494 |
| per-MetalDam | 0.2276 | MISPLACED | 0.152 | 2.058 | 0.022 | 0.652 |
| per-Steel1 | 0.2598 | MISPLACED | 0.167 | 1.924 | 0.028 | 0.395 |
| per-uhcs1 | 0.2616 | MISPLACED | 0.155 | 1.974 | 0.028 | 0.425 |
| closest | 0.2598 | MISPLACED | 0.167 | 1.924 | 0.028 | 0.395 |

**closest** chose `Steel1` (target = held-out dataset's true boundary fraction 0.0479).
