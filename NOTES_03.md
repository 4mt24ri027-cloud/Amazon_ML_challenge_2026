Feature 
    1. name_missing — "Is a name blank?"

    What it means: A simple Yes (1) or No (0) flag telling the model whether either of the two names is missing or empty.

    In one line: "Tells the machine not to treat a blank name as a mismatch."

2. name_set — "Is one name inside the other?"

    What it means: Checks if the words of the shorter name are completely present inside the longer name, ignoring extra middle names, titles, or word order.

    In one line: "Forgives missing middle names and swapped word order (e.g., 'Rohit Sharma' vs 'Sharma Rohit Kumar' gets a perfect 1.0)."

3. name_sort — "Same words, just jumbled?"

    What it means: Puts the words in both names in A-to-Z alphabetical order first, and then checks how closely they match character by character.

    In one line: "Forgives swapped first and last names, but still reduces the score if extra words are present."

4. name_ratio — "Strict letter-by-letter match"

    What it means: Compares both names directly from left to right, character by character, without changing word order.

    In one line: "Catches small spelling typos when both names are written in the exact same order."

5. name_tri — "3-letter chunk overlap"

    What it means: Breaks both names into 3-letter pieces (trigrams) and checks what fraction of those pieces are shared between the two names.

    In one line: "Catches similar-sounding names and missing spaces (like 'Ramkumar' vs 'Ram Kumar')."

6. name_tok — "Exact word overlap"

    What it means: Counts how many exact whole words are shared between the two names, divided by the total unique words across both.

    In one line: "Strict word counter—if 2 out of 3 total words match exactly, the score is 0.67."




### `addr_features(a1, a2)` — Address Similarity Features

Computes 6 numerical features (`0.0` to `1.0`) from two tokenised address sets (`frozenset`) for entity resolution. It separates **street/locality words** from **house/PIN numbers** (stripping leading zeros like `07149` -> `7149`) so the model can handle partial addresses and distinguish between next-door neighbours.

| Feature | Range | What It Indicates |
| :--- | :--- | :--- |
| **`addr_missing`** | `0` or `1` | `1` if either address is empty, preventing missing data from being treated as a mismatch. |
| **`addr_jaccard`** | `0.0–1.0` | Overall token overlap (shared tokens divided by total unique tokens across both addresses). |
| **`addr_min`** | `0.0–1.0` | Overlap relative to the **shorter** address—scores `1.0` when a brief address is fully contained inside a detailed one. |
| **`num_match`** | `0` or `1` | `1` if both addresses contain numbers and share at least one matching house, flat, or PIN number. |
| **`num_conflict`** | `0` or `1` | `1` if both addresses contain numbers but share **none** (e.g., neighbours on the same street with different house numbers). |
| **`word_jaccard`** | `0.0–1.0` | Overlap of alphabetical words only (street, area, city), ignoring all numbers. |