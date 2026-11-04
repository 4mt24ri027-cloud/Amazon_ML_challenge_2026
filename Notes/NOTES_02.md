# NOTES_02: Day 2, blocking

Measured on the 20k validation entities of the dev sample (pool: S2 228,117 + S3 239,377 = 467,494 records). All numbers below are from the re-run after the `norm_address` fix (see "Bug found and fixed"), unless marked otherwise. Notebook: `notebook/02_blocking_eda.ipynb`.

## What was built

Files: `src/normalize.py` (cleaning) and `src/blocking.py` (keys, index, candidates, rerank). Key format is `type|country|value`: the type label keeps key kinds apart (so each can be measured), and the country keeps countries apart (nothing hard-coded, France works). The index holds S2/S3 only. S1 is the query side and is never indexed.

### src/normalize.py

- **`norm_name(name)`**: lowercases, strips accents and punctuation, drops legal suffixes (pvt, ltd, llc, inc ...). Returns a string. Native-script names become `''` (they have no ASCII letters), so they give no name keys.
- **`norm_address(addr)`**: same cleaning, plus street abbreviations, state codes and native-script state names mapped to English. Returns a `frozenset` of tokens, so reordering and repeats do not matter. Missing address (NaN) gives an empty set. All maps are plain dicts, so France can be added without code changes.
  - `NATIVE_STATE`: 30 entries, one native name per state (Hindi only for Hindi-belt states). Applied before the ASCII step, because that step would delete native letters.
  - `STATE_ABBR`: only 8 codes (tn, tx, md, il, nc, mh, rj, mp). `STREET_ABBR`: 8 entries (cir, dr, ave, blvd, st, rd, ln, ct).
  - `JUNK_TOKENS`: n, a, na, nan, null, none, nil, unknown ("N/A" becomes the tokens n and a).

### src/blocking.py

- **`name_keys(name, country)`**: one `n|` key per whole name word and one `p|` key per 4-letter prefix (words of 5+ letters; the prefix length is hard-coded as `w[:4]`). Words under 3 letters are skipped because they make huge blocks. The prefix key lets "Offie" and "Office" meet.
- **`address_keys(addr, country)`**: pairs every house number with every address word of 4+ letters, as `a|country|number|word`. Either one alone makes huge blocks, the pair is specific. No house number means no address key.
- **`all_keys(name, address, country)`**: the union of the name and address keys. A record is found through its name OR its address. Used on both sides (index and lookup), so keys always match in format.
- **`build_index(pool)`**: loops over the S2+S3 pool once and maps each key to the list of record ids that have it. Stores ids only (low RAM) and returns a plain dict. Slow line, run once.
- **`block_sizes(index)`**: block size per key as a pandas Series. Used to see the size distribution and the biggest (generic) keys before choosing a cap.
- **`get_candidates(name, address, country, index, cap)`**: for one S1 entity, makes its keys, looks each up in the index and returns the union of ids. Blocks bigger than `cap` are skipped. Simple, but the cap alone controls both recall and cost.
- **`get_candidates_topk(..., cap, k)`**: like `get_candidates`, but each record gets a score = sum of 1/block size over the keys it shares, and only the top `k` are kept. Rare shared keys count as strong evidence, common ones almost nothing. The cap can stay high (protects recall) and `k` controls the cost. Used as the first stage with k=100.
- **`prepare_pool(pool)`**: cleans every pool record once (`norm_name`, `norm_address`) and stores the result in a dict by id. Cleaning 467k records once is far cheaper than cleaning them again for every S1 entity.
- **`rerank(name, address, cand_ids, prepared, k)`**: second stage. Scores each candidate as name similarity (`token_set_ratio`, forgives dropped or doubled words) plus address similarity (shared tokens divided by all tokens). The sum lets a strong address rescue an unrelated name. A missing address or empty name scores 0 for that part. Returns the top `k`. No learning, so scoring it on the validation entities is fair. A cheap preview of Day 3, to be replaced by a classifier.

### How they fit together

Index once with `build_index` and `prepare_pool`. For each S1 entity: `get_candidates_topk` (cap 1000, k 100) gives about 92 candidates with recall 0.909, then `rerank` keeps the best 20.

## Bug found and fixed

In `norm_address`, the cleaning lines and the `return` sat one indent too deep, inside the `for native, english` loop. The function returned after the first dict entry, so only Andhra Pradesh in Telugu was mapped and every other native state name was deleted by the ASCII step. No error appeared, and the tests missed it because none of them had a native-script state. Fixed by moving those lines out of the loop. Effect: keys went from 1,153,212 to 1,155,909 and recall rose by about 0.1 point. Lesson: test the odd cases (native script, missing address) too.

Native-script check on the full train S2+S3 addresses: after the 30 state names are mapped, only 2,223 addresses still hold non-ASCII text, and it is encoding junk (Â, â, Ñ), not Indian scripts. So the state map covers essentially all native-script addresses.

## Block sizes (before any cap)

- 1,155,909 keys. Mean 4.5, median 1 record, 99% of blocks <= 37, 99.9% <= 412, max 14,624.
- The biggest blocks are generic words and their prefixes: center, partners, com, and, group, services, india, care.

## Results

| Setting | Blocking recall | Candidates per S1 |
| --- | --- | --- |
| cap=50 | 0.8509 | 39.2 |
| cap=200 | 0.8951 | 164.0 |
| cap=1000 | 0.9530 | 810.7 |
| cap=200, top-k by 1/size, k=20 | 0.8466 | 17.5 |
| cap=200, top-k by 1/size, k=30 | 0.8579 | 25.4 |
| cap=1000, top-k by 1/size, k=20 | 0.8651 | 19.4 |
| cap=1000, top-k by 1/size, k=30 | 0.8772 | 28.9 |
| cap=1000, top-k by 1/size, k=50 | 0.8912 | 47.5 |
| first stage top-100 (cap=1000, 1/size) | 0.9091 | 92.3 |
| **top-100, then rerank, k=20** | **0.8940** | <= 20 |
| top-100, then rerank, k=30 | 0.8968 | <= 30 |

Old rows, from before the fix and from the first top-k version (plain key count, function no longer in the code, so the notebook cannot reproduce them): k=20 gave 0.8609 at 19.4, k=30 gave 0.8704 at 28.9, k=50 gave 0.8816 at 47.5. They are kept only to show why the 1/size score was added (about +0.4 to +1.0 point at the same cost; the comparison is approximate because of the fix).

- Cap alone cannot meet the target: recall costs too many candidates.
- Top-k keeps the cap high (protects recall) and lets k control the cost. At the same k, cap=200 is worse than cap=1000.
- Reranking the top 100 with a direct name and address comparison gained +2.9 points at k=20 and +2.0 at k=30 over 1/size top-k, at the same cost.
- Rerank keeps about 98% of what the first stage holds (0.8940 of 0.9091). The remaining gap to 0.953 is in the first stage pool size.
- Comparisons: 20 candidates per S1 entity against a 467,494-record dev pool. The dev pool holds only 4.5% of the unmatched records, so full-scale numbers will be harder (re-check on Day 6).

## Reading 10 misses (top-30 misses)

- 8 of 10 were inside the cap=1000 set: the loss is ranking, not blocking.
- House numbers break the address key: 7149 vs 07149, 1302 vs 302, 2242 vs 224, 25/B vs 23/B, `###847`.
- Missing address (nan) leaves only generic name words (Shyam Marketing, Uptown Bistro).
- Native-script names give no name keys, so only the address can find them (Maa Tirupati, Smart Indo).
- Typos and doubled words (Mamregn, Uptown Uptown Bistro, Eye Eye Prairie Care) are handled better by a direct similarity than by key counting.
- Never found by blocking: Maa Tirupati Media, Eye Prairie Care. For Eye Prairie the house number differs (2242 vs 224), so no address key is shared, and my guess is that its name words are in blocks bigger than the cap; not checked yet.
- State codes are unmapped in several misses (AZ, DL, AP), so the state token does not match its full name.
- The miss record shown for an entity can change between runs, because the code picks the first item of a set. The conclusions did not change.
- Only 10 examples: hints, not percentages.

## Decisions

- Keep the 4-letter prefix key as a loose key (it does not correct spelling; false collisions only add candidates, a missed pair cannot be recovered). Prefix length and cap are tuned from measured numbers (prefix length is hard-coded for now).
- Spelling correction is not done in `normalize.py` (no dictionary, risk of damaging real names, and the typo is a useful signal later). Possible Day 5 option: correct rare words toward frequent words from the data itself.
- Blocking is tuned and scored on the 20k validation entities only, never on the 80k training entities.
- Final Day 2 setting: first stage top-100 (cap=1000, 1/size score), then rerank to k=20.

## Open items (later)

- Street-plus-city key for addresses without a house number (not written).
- Address key that tolerates house-number noise (leading zeros, dropped digits, `25/B`).
- Extend `STATE_ABBR` beyond 8 codes (AZ, DL and AP were seen unmapped). Check `ct`: it is mapped to "court" but is also the state code for Connecticut.
- Check whether Eye Prairie keys were skipped by the cap (print their block sizes).
- First-stage pool size: try 300 instead of 100 and measure the rerank time.
- Measure recall by country, and for native-script names.
- In the notebook, fix the comment above the first top-k cell (it says "number of shared keys", but the function now uses 1/size).
- Optional: re-run the cap-only and top-k cells once in a fresh kernel so all rows come from one run.
- Re-check everything at full scale on Day 6.