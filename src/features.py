from rapidfuzz import fuzz


def trigrams(s: str) -> set:
    # WHY padding: so the first and last letters also appear in trigrams
    s = f" {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def name_features(n1: str, n2: str) -> dict:
    # WHY flag: native-script names normalise to '' (no information), and a 0 score
    # must not be read by the classifier as "the names are different"
    if not n1 or not n2:
        return {"name_missing": 1, "name_set": 0.0, "name_sort": 0.0,
                "name_ratio": 0.0, "name_tri": 0.0, "name_tok": 0.0}

    t1, t2 = set(n1.split()), set(n2.split())
    g1, g2 = trigrams(n1), trigrams(n2)
    return {
        "name_missing": 0,
        # WHY /100: rapidfuzz gives 0-100, and every feature on a 0-1 scale is easier to read
        "name_set": fuzz.token_set_ratio(n1, n2) / 100,
        "name_sort": fuzz.token_sort_ratio(n1, n2) / 100,
        "name_ratio": fuzz.ratio(n1, n2) / 100,
        "name_tri": len(g1 & g2) / len(g1 | g2),
        "name_tok": len(t1 & t2) / len(t1 | t2),
    }


def addr_features(a1: frozenset, a2: frozenset) -> dict:
    # WHY flag: a missing address is "unknown", not "different"
    if not a1 or not a2:
        return {"addr_missing": 1, "addr_jaccard": 0.0, "addr_min": 0.0,
                "num_match": 0, "num_conflict": 0, "word_jaccard": 0.0}

    # WHY lstrip("0"): 07149 and 7149 are the same house number written differently
    n1 = {t.lstrip("0") or "0" for t in a1 if t.isdigit()}
    n2 = {t.lstrip("0") or "0" for t in a2 if t.isdigit()}
    w1 = {t for t in a1 if t.isalpha()}
    w2 = {t for t in a2 if t.isalpha()}

    both_have_nums = bool(n1) and bool(n2)
    shared = len(a1 & a2)
    return {
        "addr_missing": 0,
        "addr_jaccard": shared / len(a1 | a2),
        # WHY min: one address often lacks parts the other has
        "addr_min": shared / min(len(a1), len(a2)),
        "num_match": int(both_have_nums and bool(n1 & n2)),
        # WHY needs both: a missing number is unknown, not a conflict
        "num_conflict": int(both_have_nums and not (n1 & n2)),
        "word_jaccard": len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0.0,
    }



def pair_features(n1: str, a1: frozenset, n2: str, a2: frozenset) -> dict:
    # WHY one merged dict: the classifier needs one flat row per pair
    return {**name_features(n1, n2), **addr_features(a1, a2)}