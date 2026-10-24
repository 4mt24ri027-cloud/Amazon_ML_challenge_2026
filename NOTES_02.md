# NOTES_02: Day 2, blocking

Measured on the 20k validation entities of the dev sample (pool: S2 228,117 + S3 239,377 = 467,494 records).

## What was built

Files: `src/normalize.py` (cleaning) and `src/blocking.py` (keys, index, candidates, rerank). Key format is `type|country|value`: the type label keeps key kinds apart (so each can be measured), and the country keeps countries apart (nothing hard-coded, France works). The index holds S2/S3 only. S1 is the query side and is never indexed.

### src/normalize.py

- **`norm_name(name)`**: lowercases, strips accents and punctuation, drops legal suffixes (pvt, ltd, llc, inc ...). Returns a string. Native-script names become `''` (they have no ASCII letters), so they give no name keys.
- **`norm_address(addr)`**: same cleaning, plus street abbreviations (cir to circle), state codes (TN to tennessee) and native-script state names mapped to English. Returns a `frozenset` of tokens, so reordering and repeats do not matter. Missing address (NaN) gives an empty set. The abbreviation and state maps are plain dicts, so France can be added without code changes.

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

## Block sizes (before any cap)

- 1,153,212 keys. Median block = 1 record, 99% of blocks <= 36, 99.9% <= 405, max 14,624.
- The biggest blocks are generic words and their prefixes: center, partners, com, and, group, services, india, care.

## Results

| Setting | Blocking recall | Candidates per S1 |
| --- | --- | --- |
| cap=50 | 0.8509 | 39.8 |
| cap=200 | 0.8951 | 165.6 |
| cap=1000 | 0.9529 | 809.9 |
| cap=1000, top-k by plain count, k=20 | 0.8609 | 19.4 |
| cap=1000, top-k by plain count, k=30 | 0.8704 | 28.9 |
| cap=1000, top-k by 1/size, k=20 | 0.8644 | 19.4 |
| cap=1000, top-k by 1/size, k=30 | 0.8768 | 28.9 |
| first stage top-100 (cap=1000, 1/size) | 0.9087 | 92.3 |
| **top-100, then rerank, k=20** | **0.8928** | <= 20 |
| top-100, then rerank, k=30 | 0.8958 | <= 30 |

- Cap alone cannot meet the target: recall costs too many candidates.
- Top-k keeps the cap high (protects recall) and lets k control the cost.
- Reranking the top 100 with a direct name and address comparison gained +2.8 points at k=20 and +1.9 at k=30 over 1/size top-k, at the same cost.
- Rerank keeps about 98% of what the first stage holds (0.8928 of 0.9087). The remaining gap to 0.953 is in the first stage pool size.
- Comparisons: 20 candidates per S1 entity against a 467,494-record dev pool. The dev pool holds only 4.5% of the unmatched records, so full-scale numbers will be harder (re-check on Day 6).

## Reading 10 misses (top-30 misses)

- 8 of 10 were inside the cap=1000 set: the loss is ranking, not blocking.
- House numbers break the address key: 7149 vs 07149, 1302 vs 302, 2242 vs 224, 25/B vs 23/B, `###847`.
- Missing address (nan) leaves only generic name words (Shyam Marketing, Uptown Bistro).
- Native-script names give no name keys, so only the address can find them (Maa Tirupati, Smart Indo).
- Typos and doubled words (Mamregn, Uptown Uptown Bistro) are handled better by a direct similarity than by key counting.
- Never found by blocking: Maa Tirupati Media, Eye Prairie Care. For Eye Prairie my guess is that its name words are in blocks bigger than the cap; not checked yet.
- Only 10 examples: hints, not percentages.

## Decisions

- Keep the 4-letter prefix key as a loose key (it does not correct spelling; false collisions only add candidates, a missed pair cannot be recovered). Prefix length and cap are tuned from measured numbers (prefix length is hard-coded for now).
- Spelling correction is not done in `normalize.py` (no dictionary, risk of damaging real names, and the typo is a useful signal later). Possible Day 5 option: correct rare words toward frequent words from the data itself.
- Blocking is tuned and scored on the 20k validation entities only, never on the 80k training entities.
- Final Day 2 setting: first stage top-100 (cap=1000, 1/size score), then rerank to k=20.

## Open items (later)

- Street-plus-city key for addresses without a house number (not written).
- Address key that tolerates house-number noise (leading zeros, dropped digits, `25/B`).
- Check whether Eye Prairie keys were skipped by the cap (print their block sizes).
- First-stage pool size: try 300 instead of 100 and measure the rerank time.
- Measure recall by country, and for native-script names.
- Re-check everything at full scale on Day 6.
- Tidy `src/blocking.py`: move the imports to the top and delete the unused `Counter` import. Optionally make the prefix length a constant if it gets tuned.