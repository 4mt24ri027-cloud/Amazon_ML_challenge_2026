import time
import pandas as pd

from src.normalize import norm_name, norm_address
from src.features import pair_features, add_relative_features
from src.blocking import get_candidates_topk, rerank


def build_pairs(s1_rows, index, prepared, pool=300, k=30):
    # WHY no label column: on the real test set there is no ground truth,
    # so this must work without it. Labels are only needed for scoring, done outside.
    rows = []
    for eid, name, addr, country in zip(s1_rows["entity_id"], s1_rows["business_name"],
                                        s1_rows["business_address"], s1_rows["country"]):
        first = get_candidates_topk(name, addr, country, index, 1000, pool)
        top = rerank(name, addr, first, prepared, k)
        # WHY outside the inner loop: the S1 side is the same for all its candidates
        n1, a1 = norm_name(name), norm_address(addr)
        for cid in top:
            n2, a2 = prepared[cid]
            f = pair_features(n1, a1, n2, a2)
            f["s1"], f["cand"] = eid, cid
            rows.append(f)
    return pd.DataFrame(rows)


def predict_chunk(s1_rows, index, prepared, booster, cfg):
    pairs = build_pairs(s1_rows, index, prepared)
    # WHY per chunk is safe: relative features compare candidates of the same S1 entity,
    # and a chunk always holds whole S1 entities
    pairs = add_relative_features(pairs)
    pairs["p"] = booster.predict(pairs[cfg["features"]])
    # WHY drop the rest now: below-threshold pairs are about 90% of the rows and are never needed again
    return pairs.loc[pairs["p"] >= cfg["threshold"], ["s1", "cand", "p"]]


def run_chunked(s1_df, index, prepared, booster, cfg, chunk=20_000):
    kept = []
    for start in range(0, len(s1_df), chunk):
        t0 = time.time()
        kept.append(predict_chunk(s1_df.iloc[start:start + chunk], index, prepared, booster, cfg))
        print(f"entities {start:,}-{min(start + chunk, len(s1_df)):,} "
              f"| {time.time() - t0:.0f}s | kept rows so far {sum(len(k) for k in kept):,}")
    allk = pd.concat(kept)
    # WHY here and not per chunk: two S1 entities in different chunks can claim the same record,
    # and only the global best score may keep it
    allk = allk.sort_values("p", ascending=False).drop_duplicates("cand")
    return allk.groupby("s1")["cand"].apply(set).to_dict()