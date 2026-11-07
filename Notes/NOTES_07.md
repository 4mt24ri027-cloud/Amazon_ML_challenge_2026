# NOTES_07: Day 6, full-scale run on Kaggle

Code: `src/run_full.py` (one script, resumable). Notebook: `notebook/08_kaggle_full_scale.ipynb` (ran on Kaggle, not meant to run locally). Result file: `results/full_run_results.json`.

## Why Kaggle
The full index for both countries was estimated at about 21.5 GB (laptop: 3 to 6 GB free). Kaggle gave 4 cores, 33.7 GB RAM and Python 3.13 (pandas 2.3.3, lightgbm 4.6.0). The dataset is attached there, and the private repo is cloned with a read-only token kept in Kaggle Secrets (never in the notebook).

## Measured on Kaggle
| Measure | Value |
| --- | --- |
| US pool | 6,186,873 records, 9,681,639 keys |
| US index plus prepared dict | 11.83 GB RAM, built in about 11 min (369 s + 246 s) |
| Free RAM with the US index loaded | 20.7 GB |
| 2 worker processes | 80k entities in 163 s, about 5 GB RAM per worker |
| 4 worker processes | not used: would not fit in RAM |
| US training pairs (180,085 entities) | 5,311,007 rows, positive rate 0.111 |
| Recall ceiling, US, full pool | 0.9461 (dev pool: 0.9711) |

The index is built one country at a time, because the country is part of every key and US and India never compete. This also means a new country (France) needs no code change.

## Design of `run_full.py`
- A: build the index for one country, then labeled pairs for the 300k train entities (seed 11, none from the hold-out). B: one LightGBM matcher (22 features, same settings as v4) trained on both countries. C: predict all S1 entities, one country at a time, 20k entities per chunk, 2 workers. D: score the hold-out.
- Each chunk is saved to disk and finished stages leave a `.done` marker, so a restarted run skips finished work.
- Workers use `fork`, so they share the index. `maxtasksperchild=8` replaces a worker after 8 chunks, which releases memory copied by Python's reference counting.
- Smoke mode (`SMOKE=1`) runs the same stages on a small slice in a few minutes. It passed before the full run.
- Hold-out: the same 300k entities as the laptop test (same seeds, first IDs checked). The threshold was chosen on the first third (100k) and the score is reported on the other 200k, so the reported number was never used for any choice.

## Result (200,000 hold-out entities, full pool, threshold 0.6)
| Measure | Value |
| --- | --- |
| Macro F0.5 | 0.8959 |
| Precision / recall | 0.9675 / 0.8195 |
| Singletons falsely matched | 1,760 of 11,147 (15.8%) |
| US / India | 0.9165 / 0.8649 |

For reference: exact-match baseline 0.6028 (dev pool, 20k validation entities); frozen v4 on the 1.4M-pool hold-out 0.9459 (US 0.9566, India 0.9298, 2,263 of 16,747 false singletons).

## How to read it
- The drop from 0.9459 to 0.8959 comes with the full pool: about 7x more decoys than the 1.4M pool, and a lower recall ceiling (US 0.9461).
- Recall is the weak side (0.8195), precision is still 0.9675. With a US ceiling of about 0.946, a large part of the recall is lost after blocking, in the matcher or the one-S1-per-record rule (my reading, not measured yet).
- India is lower than US. Its full-scale ceiling was not measured.
- The baseline (0.6028) was measured on the dev pool, so the comparison with 0.8959 is not like for like. A fair comparison needs the baseline scored on the full pool.
- It is not known if the retrained matcher beats frozen v4 at full scale, because v4 was never scored on the full pool.

## Problems met and lessons
- The Kaggle kernel died when my internet dropped. Fix: run as a script with "Save & Run All" in the background, chunks saved to disk, resume markers.
- The console stopped printing while the run continued. Progress was followed by counting the saved chunk files, and `results.json` was the signal that the run was finished.
- The first clone left the token in `.git/config`. The remote URL is now reset to the plain one right after cloning.
- An unchanged result after an edit usually means the new code did not run (same lesson as Day 5).

## Saved files
`results/full_run_results.json`, `models/full/matcher_full_lgbm.txt` and `matcher_full_config.json`. The 112 prediction chunks (about 230 MB) are kept outside git.

## Open items
- Score the exact-match baseline and frozen v4 on the full pool for a fair comparison.
- Split the recall loss: blocking ceiling by country, dropped by the matcher, removed by the one-S1-per-record rule.
- 1,760 false singletons: near-name decoys were the main type on the 1.4M pool.
- France: add French street abbreviations to `STREET_ABBR` (plain dict) before Day 7, and check a few predictions by eye (no labels).
- `STATE_ABBR` still has 8 codes, and `ct` is "court" but also Connecticut.
- Day 7: run the same script logic on `test_source1-3.tsv`, one country at a time, and write `candidate_pairs.tsv` and `matching_results.tsv`.