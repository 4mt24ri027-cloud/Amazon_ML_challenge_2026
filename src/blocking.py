import heapq
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

from src.normalize import norm_name, norm_address


def name_keys(name, country) -> set:
    keys = set()
    for w in norm_name(name).split():
        # WHY skip short words: "of", "fh", "a" appear everywhere and make huge blocks
        if len(w) < 3:
            continue
        # WHY whole word: exact matches
        keys.add(f"n|{country}|{w}")
        # WHY 4-letter prefix: survives typos after the 4th letter (Offie / Office)
        if len(w) >= 5:
            keys.add(f"p|{country}|{w[:4]}")
    return keys


def address_keys(addr, country) -> set:
    tokens = norm_address(addr)
    nums = {t for t in tokens if t.isdigit()}
    # WHY min length 4: short words are too common to separate addresses
    words = {t for t in tokens if t.isalpha() and len(t) >= 4}
    # WHY number + word together: either alone makes huge blocks, the pair is specific
    return {f"a|{country}|{n}|{w}" for n in nums for w in words}


def all_keys(name, address, country) -> set:
    # why union: we want to block on either name or address, not require both
    return name_keys(name, country) | address_keys(address, country)


# WHY constants: if a column name is different in the data, we fix one line, not the code
NAME_COL = "business_name"
ADDRESS_COL = "business_address"


def build_index(pool: pd.DataFrame) -> dict:
    index = defaultdict(list)

    # why zip  over columns :much faster than the  iterrows on hundreds of rows
    for eid, name, addr, country in zip(
            pool["entity_id"], pool[NAME_COL], pool[ADDRESS_COL], pool["country"]
    ):
        for key in all_keys(name, addr, country):
            # WHY only the id: keeps memory low, we can look up the row later
            index[key].append(eid)
    # WHY dict(): a plain dict does not create empty entries when a missing key is looked up
    return dict(index)


def block_sizes(index: dict) -> pd.Series:
    # WHY a Series: describe() and sort_values() give the size distribution in one line
    return pd.Series({key: len(ids) for key, ids in index.items()})


def get_candidates(name, address, country, index: dict, cap: int) -> set:
    cands = set()
    for key in all_keys(name, address, country):
        ids = index.get(key)
        # WHY skip big blocks: generic words ("center", "group") match thousands of
        # unrelated records, so they add cost but no signal
        if ids is None or len(ids) > cap:
            continue
        cands.update(ids)
    return cands


def get_candidates_topk(name, address, country, index: dict, cap: int, k: int) -> set:
    scores = defaultdict(float)
    for key in all_keys(name, address, country):
        ids = index.get(key)
        if ids is None or len(ids) > cap:
            continue
        # WHY 1/size: a rare shared key is strong evidence, a common one is weak
        w = 1.0 / len(ids)
        for eid in ids:
            scores[eid] += w
    # WHY nlargest: faster than sorting every scored record
    return set(heapq.nlargest(k, scores, key=scores.get))


def prepare_pool(pool: pd.DataFrame) -> dict:
    # WHY precompute: cleaning 467k records once is far cheaper than cleaning them again for every S1 entity
    return {
        eid: (norm_name(n), norm_address(a))
        for eid, n, a in zip(pool["entity_id"], pool[NAME_COL], pool[ADDRESS_COL])
    }


def rerank(name, address, cand_ids, prepared: dict, k: int) -> set:
    n1, a1 = norm_name(name), norm_address(address)
    scored = []
    for cid in cand_ids:
        n2, a2 = prepared[cid]
        # WHY token_set_ratio: forgives dropped or doubled words ("Uptown Uptown Bistro")
        name_sim = fuzz.token_set_ratio(n1, n2) / 100 if n1 and n2 else 0.0
        # WHY overlap: reordered or abbreviated addresses still share most tokens
        addr_sim = len(a1 & a2) / len(a1 | a2) if a1 and a2 else 0.0
        # WHY the sum: a strong address can rescue an unrelated name (Mirasol case)
        scored.append((name_sim + addr_sim, cid))
    scored.sort(reverse=True)
    return {cid for _, cid in scored[:k]}
