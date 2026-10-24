# Notes: Business Entity Resolution (Amazon ML Challenge 2026, self-paced)

## Day 1: data facts
- Sizes: S1 = 2,206,821 | S2 = 5,034,616 | S3 = 5,285,603 records
- All-pairs comparison would be about 2.3e13 pairs, so blocking is mandatory
- Country in S1: US 60% / India 40%. France exists only in the test set, so no country-specific logic
- S1 has no nulls. S2 and S3 each have about 3.3% missing addresses (real NaN)
- No whitespace-only values in any name or address column. Missing data is only the NaN addresses in S2/S3 (about 3.3%)
- Matches per S1: average 3.46, mode 3, almost all at 8 or fewer
- Singletons (no matches): 123,247 = 5.6% of S1
- Every matched S2/S3 ID appears exactly once (7,638,365 matched, all unique), so each record
  belongs to at most one S1 entity
- Unmatched S2/S3 records: 2,681,854. The matcher must reject these

## Noise seen in 10 sampled matches
- Names: case changes (S2 often UPPERCASE), typos (Offie, Rcdbr, M5), accents (Récord),
  dropped words, duplicated words (FH FH Business), brackets, suffix variants (Pvt Ltd / Private Limited)
- Addresses: abbreviations (Cir/Circle, Blvd), state forms (TN/Tennessee, MH/Maharashtra),
  non-Latin state names (Telugu, Hindi), reordered parts, missing house number,
  junk characters (##, trailing dots, N/A), unit numbers that differ (101 vs 1-101), missing address

## Surprises
- Some matched names look unrelated (Mirasol, TEAMAIR.COM, MS Corp Services). The address is the anchor
- Name-only matching would miss these, so address features must carry weight

## Non-Latin script findings
- Only India is affected: about 9-10% of S2/S3 names and addresses. S1 is 0% non-Latin
- Scripts by frequency: Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, others
- Addresses: usually only the state name is in native script, the rest is Latin
- Names: the whole business name is transliterated phonetically (not translated),
  e.g. "फर्स्ट फूड प्राइवेट लिमिटेड" = "First Food Private Limited"
- Decision: build a state-name lookup (native script to English). Skip transliteration for now and rely on
  address blocking. Revisit only if recall stays low. Any library must run fully offline (no-external-data rule)

## Day 1 baseline: exact match on country + normalized name
- Evaluated on the 20k validation S1 entities, using the 100k dev sample (pool of about 467k S2+S3 records)
- Macro F0.5 = 0.6028 (the score to beat)
- Singletons: 0.7603, so 24% of true singletons got a false match from name collisions
- Entities with matches: 0.5935
- Average predictions per S1: 3.29 (true average: 3.46)
- Native-script names normalize to an empty string, so they are dropped from the pool (otherwise they would all collide on one key)
- Baseline precision = 0.4998 and recall = 0.463 (pooled over validation entities with matches)
- Singletons falsely matched: 267 of 1,114 (24%)
- Both are weak, so blocking alone is not enough. Blocking must raise recall, and a separate matching
  stage must restore precision by comparing name and address
- Lesson: exact keys are fast (dictionary lookup, no pairwise comparison) but break on any typo,
  and name alone is not unique enough, so address must be part of the key

## Decisions for Day 2 (blocking)
- Normalize: lowercase, strip accents, strip punctuation, unify abbreviations, map states to full names
- Block within country (US and India never compete)
- Blocking keys use both name and address tokens (house number + street + city)
- Character n-grams for names, so typos still overlap
- Target: small candidate set per S1 entity (Amazon ranks smaller sets higher), with recall measured on a held-out split
- Blocking only generates candidates. Plan a separate scoring stage that rejects weak ones,
  since F0.5 punishes false matches twice as hard

## Open questions
- Why is some S2/S3 data unmatched: out-of-scope businesses or deliberate noise?
- Are the wrong baseline predictions mostly generic names, or chains with many branches?