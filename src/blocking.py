from src.normalize import norm_name,norm_address



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