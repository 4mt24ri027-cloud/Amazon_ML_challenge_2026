# NOTES_06: Day 6, hold-out test (300k entities)

Notebook: `notebook/07_holdout.ipynb`. Code: `src/pipeline.py` (chunked run).

## Setup
- Hold-out: 300,000 S1 entities, none from the dev 100k (seed 7). Pool: 1,402,291 S2/S3 records (matched ones plus 13.6% of the unmatched, dev had 4.5%). Files in `dataset/hold/` (gitignored).
- Model, features and threshold frozen from `matcher_dev_v4` (22 features, t=0.70). Nothing retuned on this set.
- Index: 5,240,781 keys, 74 s to build. Index plus prepared dict use about 2.9 GB. Scaling to the full pool (10.3M records) gives about 21.5 GB, too much for the laptop, so the full run goes to Kaggle.
- Run time: about 37 s per 20k entities, 9.5 min for 300k.

## Results (frozen v4)
| Measure | Hold-out 300k | Dev validation |
| --- | --- | --- |
| Macro F0.5 | 0.9459 | 0.9608 |
| Precision / recall | 0.9764 / 0.9165 | 0.9789 / 0.8672 (older model) |
| Singletons falsely matched | 2,263 of 16,747 (13.5%) | 71 of 1,114 (6.4%, older model) |
| US / India | 0.9566 / 0.9298 | not measured |

The drop of 0.015 is the optimism of the dev sample. The hold-out pool still has only 13.6% of the unmatched records, so the full run may score a little lower.

## Why the singletons fail
Wrong predictions for singletons (2,263 pairs from 2,263 entities, US 1,206, India 1,057):
| Type | Share |
| --- | --- |
| Different cleaned name | 59.8% |
| Same cleaned name (decoy copy) | 28.1% |
| Record name is native script | 12.1% |

- Median shorter-address size in wrong matches is 6 tokens, so short addresses are not the cause.
- In the different-name bucket (1,472 pairs) the median `name_set` is 1.00 and 90% are 0.62 or higher: mostly near-name decoys. House numbers are shared in 49% and conflict in 36%.
- About 10% have an unrelated name with the same address. True matches look the same (the Mirasol case), so these are hard to remove.
- Cause: the matcher saw few decoys in training (dev pool 4.5%), so it accepted near copies.

## Retrain test
First 200k hold-out entities for training, last 100k for evaluation (same 100k for both models):
| Model | Macro F0.5 | False singletons |
| --- | --- | --- |
| Frozen v4 | 0.9405 | 872 of 5,565 |
| Retrained, t=0.6 | 0.9452 | 676 |
| Retrained, t=0.7 | 0.9445 | 555 |
| Retrained, t=0.8 | 0.9413 | 407 |

- Gain is about +0.005 and false singletons fall by about a third at t=0.7. The threshold sweep is flat from 0.6 to 0.7.
- Saved as `models/hold/matcher_hold200k_lgbm.txt`. Experiment only: its training entities are not clean for scoring.

## Decision
Train the final matcher on full-scale pairs (about 300k entities outside the hold-out, full pool), then run all 2.2M entities on Kaggle one country at a time. Score the hold-out entities from that run: threshold on 100k, report on 200k.

## Open items
- Kaggle setup: RAM, cores, dataset path, how the code gets there.
- `STATE_ABBR` still has 8 codes, and `ct` is "court" but also Connecticut.
- 1,625 misses sit only in blocks over the cap (not fixed).
- France is untested, so run it as its own country.