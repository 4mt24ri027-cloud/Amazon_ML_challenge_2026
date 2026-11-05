# NOTES_05: Day 5, error analysis and blocking fixes

Notebook: `notebook/05_error_analysis.ipynb`. All numbers are on the 20k validation entities of the dev sample (69,455 true matches). Code changed: `src/blocking.py`, `src/features.py`, `src/matcher.py`.

## Method
Day 4 showed the main loss was before the matcher: 7,166 true matches (10.3%) were never in `val_pairs`. Find the cause of each loss, fix the biggest, re-measure. Measure first, because each cause needs a different fix.

## Where the 7,166 blocking misses were lost
| Where | Count | Share |
| --- | --- | --- |
| Cut by rerank (was in the top-100) | 854 | 11.9% |
| Not in the top-100 | 6,312 | 88.1% |

Why the 6,312 were not in the top-100:
| Cause | Count | Share |
| --- | --- | --- |
| Usable shared key, ranked below 100 | 3,045 | 48.2% |
| Shared keys only in blocks over the cap (1000) | 1,625 | 25.7% |
| No shared key, record name is native script | 990 | 15.7% |
| No shared key, other | 627 | 9.9% |
| No shared key, record has no address | 25 | 0.4% |

- Of the 3,045 ranked-below-100 misses, 2,758 shared only name keys and 287 shared an address key. Name keys are generic and weak.
- Rank of those: 101-200: 537, 201-300: 230, 301-500: 280, 501-1000: 234, 1001+: 83. The other 1,681 are inferred to be ties inside the top 100 (equal 1/size scores, arbitrary order).
- Reading examples showed house-number noise (0017 vs 17, 007 vs 7, 5111 vs 25111) and different numbers with the same street and city.

## Fixes, in order (recall = ceiling of the pairs, k=30)
| Step | Keys | Recall | Notes |
| --- | --- | --- | --- |
| Day 4 setting (pool 100) | 1,155,909 | 0.8968 | |
| First stage pool 300 | 1,155,909 | 0.9124 | 21.6 s for 20k entities |
| Strip leading zeros in `address_keys` | 1,114,321 | 0.9187 | 0017 and 17 now share a key |
| Street-pair keys (`s|` keys: two address words, no number) | 2,259,243 | 0.9548 | lookup 29.5 s |
| Rerank: missing part counts as unknown (other part doubled) | same | 0.9711 | k=60 old rerank only 0.9602 |

Rerank detail: the old score was name + address, so an empty name (native script) or missing address counted as 0, "different". The new score doubles the other part, so a perfect address with an unknown name can win.

## Matcher results (LightGBM, t=0.70, one-S1-per-record rule)
| Pairs / features | Ceiling | Macro F0.5 |
| --- | --- | --- |
| Day 4 (pool 100, old keys, 12 features) | 0.8968 | 0.9248 |
| v2 pairs (pool 300, zero fix, street keys) | 0.9548 | 0.9572 |
| v3 pairs (unknown-aware rerank) | 0.9711 | 0.9577 |
| v3 pairs + 10 relative features (22 features) | 0.9711 | **0.9608** |

- Baseline (Day 1): 0.6028.
- Threshold sweep on v3: 0.3: 0.9508, 0.4: 0.9540, 0.5: 0.9563, 0.6: 0.9576, 0.7: 0.9577, 0.8: 0.9524. On the final model: 0.5: 0.9576, 0.6: 0.9593, 0.7: 0.9608, 0.8: 0.9591. So 0.70 stays.
- The extra recall turned into little score: the rescued matches are hard (native-script names, where only the address vouches for the pair). On v3, 830 of 4,140 offered native-script true pairs were dropped (20.0%) against 2,371 of 63,311 Latin ones (3.7%).
- Reading dropped pairs: native-script drops had a good address, but many records share a street, so address alone is not unique. Latin drops had a random name (Deltairi, clabaughdavis.com) and a near-identical address.
- Relative features (`add_relative_features` in `src/features.py`): for addr_jaccard, addr_min, name_tri and name_set, the gap to the best candidate of the same S1 entity and the rank within it, plus `n_num_match` and `n_cands`. Native-script drops fell from 830 to 699, Latin from 2,371 to 2,151.

## Remaining misses after the street keys (3,139)
Cut by rerank, native-script name: 1,322 (42.1%). Not in the pool of 300, Latin name: 1,221 (38.9%). Cut by rerank, Latin name: 340 (10.8%). Not in the pool, native-script name: 256 (8.2%). The unknown-aware rerank was built for the first of these.

## Saved
`models/dev/matcher_dev_v4_lgbm.txt` and `matcher_dev_v4_config.json` (22 features, t=0.70). Pairs: `train_pairs_v2/v3.pkl`, `val_pairs_v2/v3.pkl` (gitignored).

## Costs to carry into Day 6
- Street keys doubled the key count, so index RAM doubles. Day 6 has 10.3M pool records.
- Lookup is about 1.5 ms per entity, so 2.2M entities take roughly an hour. Plan chunks.
- Pair building (80k plus 20k entities) takes about 3 minutes.

## Lessons
- An unchanged result after an edit means the new code did not run. The street-key run printed the same numbers until `%autoreload` was switched on in the notebook.
- Counting causes before fixing changed the plan: the biggest cause was weak name keys, not native script.
- "Unknown" and "different" must stay apart in every score.

## Caveats
- The threshold and each fix were chosen on the same 20k validation entities, so 0.9608 is slightly optimistic.
- The dev pool holds only 4.5% of the unmatched records. Day 6 re-checks at scale.

## Open items
- Precision and recall of v4 not yet measured. Singleton behaviour not re-checked.
- Recall by country not measured. `STATE_ABBR` still has 8 codes (DL, AP, AZ unmapped). `ct` is "court" but also Connecticut.
- A token like `W397N5895` (letters and digits) gets no address key.
- 1,625 misses sit only in blocks over the cap: not fixed.
- The `matcher.py` change adds a `features` parameter; the Day 4 model files use the 12-feature list.