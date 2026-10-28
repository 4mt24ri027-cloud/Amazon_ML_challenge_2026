# NOTES_04: Day 4, matcher

Notebook: `notebook/04_matcher.ipynb`. Evaluated on the 20k validation entities (577,150 pairs).

## What was built
- LightGBM classifier on the 12 pair features: 200 trees, lr 0.1, 31 leaves, trained on 2,309,975 pairs (5.6 s).
- Output is a match probability per pair. Keep pairs with prob >= 0.70.
- One-S1-per-record rule: if several S1 entities keep the same S2/S3 record, only the highest-probability one keeps it (each record belongs to at most one S1 entity).
- Saved as a Day 4 checkpoint (`models/dev/matcher_dev_lgbm.txt`, `models/dev/matcher_dev_config.json`). It will be retrained at full scale on Day 6, so it is a reference point, not the final model.

## Results
| Metric | Value |
| --- | --- |
| Val AUC / average precision | 0.9988 / 0.9722 |
| Macro F0.5 (t=0.70, with rule) | 0.9248 (baseline 0.6028) |
| Precision / recall | 0.9789 / 0.8672 (recall ceiling 0.8968) |
| Singletons falsely matched | 71 of 1,114 (baseline 267) |

- Threshold sweep 0.5 to 0.8 was flat (0.9232 to 0.9248 with the rule), so t=0.70 matters little.
- The rule added about +0.001 at every threshold.
- Feature importance (split counts): name_tri, addr_min, addr_jaccard, name_ratio and word_jaccard carry most; name_missing (1%) and num_match (1.5%) carry least.

## Where the loss is (69,455 true matches in val)
| Bucket | Count | Share |
| --- | --- | --- |
| Never offered by blocking | 7,166 | 10.3% |
| Offered, dropped by matcher | 2,056 | 3.0% |
| False predictions kept | 1,300 | hurts precision |

- Blocking is the main loss, over 3x the matcher's. Day 5 targets blocking.
- F0.5 by group size: singleton 0.936, 1 match 0.827, 2 matches 0.914, 3-4 0.931, 5+ 0.940. The 1-match group is lowest but small (1,074 entities).

## Caveats
- Threshold and rule were chosen on val, so 0.9248 is slightly optimistic (the curve is flat, so the effect is small).
- The dev pool holds only 4.5% of the unmatched records, so full scale will score lower (re-check on Day 6).

## Open items
- Split the 7,166 blocking misses by cause (cut by rerank, ranked below 100, address missing, native-script name, other).
- Test a first stage of 300 instead of 100.
- Extend `STATE_ABBR`; check `ct` (court vs Connecticut).
- Add features for heavy typos and for comparing candidates of the same S1 entity.