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