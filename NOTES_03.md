# NOTES_03: Day 3, features

Goal: turn each candidate pair (one S1 entity + one S2/S3 record) into one row of numbers a classifier can learn from. Notebook: `notebook/03_features_eda.ipynb`. Code: `src/features.py`.

## What was built (src/features.py)

One row per pair, 12 features. Every score is on a 0-1 scale, and every "missing" case gets its own flag so "unknown" is never read as "different".

### `trigrams(s)`
Splits a string into 3-character pieces, with a space added at both ends so the first and last letters also count. Helper for `name_tri`.

### `name_features(n1, n2)` (inputs are `norm_name` strings)

| Feature | What it measures | Noise it targets |
| --- | --- | --- |
| `name_missing` | 1 if either name is empty | Native-script names normalise to `''`. Without the flag, a 0 score would look like "different names" |
| `name_set` | `token_set_ratio`: score is 1.0 when the words of one name sit inside the other | Dropped or doubled words (FH FH Business, Uptown Uptown Bistro) |
| `name_sort` | `token_sort_ratio`: sorts words first, then compares characters | Shuffled word order |
| `name_ratio` | `fuzz.ratio`: strict whole-string comparison | Separates near-identical names from loosely similar ones |
| `name_tri` | shared trigrams / all trigrams | Light typos and missing spaces (Offie / Office) |
| `name_tok` | shared whole words / all words | Plain exact word overlap |

If either name is empty, all five scores are set to 0.0 and `name_missing` is 1.

### `addr_features(a1, a2)` (inputs are `norm_address` frozensets)

| Feature | What it measures | Noise it targets |
| --- | --- | --- |
| `addr_missing` | 1 if either address is empty | About 3.3% of S2/S3 addresses are NaN |
| `addr_jaccard` | shared tokens / all tokens | Reordered parts |
| `addr_min` | shared tokens / size of the smaller address | One side drops parts (house number, city). Jaccard punishes that, this does not |
| `num_match` | 1 if both have numbers and share one (after stripping leading zeros) | 07149 vs 7149 |
| `num_conflict` | 1 if both have numbers and share none | Evidence against a match, but also fires on true matches with dropped digits (1302 vs 302) |
| `word_jaccard` | same as `addr_jaccard` on alphabetic tokens only | Street, city and state words without unit numbers (101 vs 1-101) |

A missing number gives neither `num_match` nor `num_conflict` (unknown is not a conflict). `lstrip("0")` is applied to numbers only.

### `pair_features(n1, a1, n2, a2)`
Merges the two dicts into one flat dict. Pandas turns a list of such dicts straight into the feature table.

## Tests on hand-picked pairs

Names:
- `record office` vs `rcdbr offie`: set 0.75, sort 0.75, ratio 0.75, **tri 0.14**, tok 0. Trigrams only help with light typos. Heavy typos like `rcdbr` share almost no trigrams, so `name_set` and `name_ratio` carry this case.
- `fh business` vs `fh fh business`: set 1.0, tok 1.0, but sort and ratio 0.88. The features disagree here, which gives the classifier something to use.
- `mirasol` vs `team air`: set, sort and ratio are 0.4, tri and tok are 0. The character scores have a noise floor, so 0.4 means nothing alone. Tri and tok at 0 help separate chance similarity from real similarity.
- empty vs `first food`: `name_missing` = 1, everything else 0.

Addresses:
- `88 Olive Circle, Lebanon, TN` vs `88 OLIVE CIR, LEBANON, TN`: all 1.0.
- `7149 ... Delhi` vs `07149 ... DL`: `num_match` 1 (zero stripped), but jaccard only 0.67 because `dl` is not in `STATE_ABBR`.
- `1302 Swift Creek Road, NC` vs `302 Swift Creek Rd, North Carolina`: `num_conflict` 1 although this is a true match, word_jaccard 0.56. This is why it is a feature and not a rule.
- NaN address: `addr_missing` 1.

## Pairs table

`build_pairs(s1_rows, k=30)`: for each S1 entity, first stage top-100 (`get_candidates_topk`, cap 1000), then `rerank` to k, then features per candidate, and label = 1 if the candidate is in the true set. The S1 side is normalised once per entity (it repeats in about 29 rows). The pool side comes from `prepared`.

k=30 was used (recall 0.8968) rather than k=20 (0.8940): the classifier gets slightly more chances to find a true match, at the cost of more rows.

| Set | Rows | Positive rate | Time |
| --- | --- | --- | --- |
| 2,000 train entities (test run) | 58,120 | 0.105 | 1.9 s |
| train (80,000 entities) + val (20,000 entities) | 2,309,975 and 577,150 | 0.107 and 0.108 | 98 s together |

- About 29 candidates per entity and about 1 in 9 rows is a true match, so the classes are imbalanced (roughly 1 : 8).
- **Recall ceiling on val: 0.8968.** This is the share of all true matches that appear in any pair, so no matcher can exceed it. It equals the Day 2 recall at k=30, which is a good cross-check of the pairs code.
- Saved as `dataset/dev/train_pairs.pkl` and `dataset/dev/val_pairs.pkl` (gitignored with the rest of `dataset/`). Reloading takes seconds, rebuilding takes minutes.

## Do the features separate true from false pairs?

Mean feature values on the 2,000-entity test run:

| Feature | Label 0 | Label 1 |
| --- | --- | --- |
| name_missing | 0.00 | 0.05 |
| name_set | 0.59 | 0.90 |
| name_sort | 0.51 | 0.85 |
| name_ratio | 0.51 | 0.85 |
| name_tri | 0.21 | 0.76 |
| name_tok | 0.19 | 0.72 |
| addr_missing | 0.02 | 0.03 |
| addr_jaccard | 0.11 | 0.72 |
| addr_min | 0.21 | 0.86 |
| num_match | 0.36 | 0.85 |
| num_conflict | 0.50 | 0.04 |
| word_jaccard | 0.11 | 0.73 |

- Every feature points the right way. The biggest gaps are in the address features (`addr_jaccard`, `word_jaccard`, `num_conflict`) and in `name_tri` and `name_tok`.
- `name_set` has the smallest name gap (0.59 vs 0.90). It scores 1.0 whenever one name's words sit inside the other's, so short or generic names give false pairs a high score.
- `num_conflict` is 0.50 for false pairs and 0.04 for true pairs, so it is one of the strongest single signals.
- `name_missing` is 0.05 for true pairs and 0.00 for false ones. The reason is the rerank: an empty name scores 0, so such a record only survives in the top 30 when its address matches. The flag therefore says "native-script name, address-driven match" for this candidate set. The classifier should keep it, but it is partly a side effect of the earlier stage.
- Means only show direction. Whether the features really work is decided on Day 4 by F0.5 on the validation entities.

## Decisions

- Features are plain similarity numbers plus missing flags, no hard rules. A rule like "different house number means no match" would be wrong for 1302 vs 302.
- Training pairs come from the same candidate process as validation and test (top-100, rerank, k=30), so the classifier sees the same kind of pair at test time.
- The pairs are built for the 80k training entities and the 20k validation entities separately, so no entity's pairs appear on both sides.

## Open items

- `STATE_ABBR` still has only 8 codes. DL, AP, AZ and others are unmapped, which lowers address scores for true pairs (Day 5).
- `ct` is mapped to "court" but is also the state code for Connecticut (still open from Day 2).
- No feature yet for heavy typos (`rcdbr` vs `record`). Possible later: a phonetic or edit-distance feature per word.
- No feature yet compares a candidate with the other candidates of the same S1 entity (for example the score gap to the best one).
- The one-S1-per-record rule is not applied yet (Day 4).
- Check `lightgbm` on Python 3.14 before Day 4. Fallbacks: `HistGradientBoostingClassifier` from scikit-learn, or xgboost.
- Feature means were checked on 2,000 entities only. Look at the full train table once, for example the share of `name_missing` and `addr_missing` by country.