import re
import unicodedata

# a set: "w in SUFFIXES" is a fast lookup, and suffixes carry no identity
# (Pvt Ltd / Private Limited are the same business)

LEGAL_SUFFIXES = {"pvt", "private", "ltd", "limited", "llc", "inc", "corp",
                  "corporation", "llp", "pllc", "co"}

def norm_name(name)  ->str:
    # NFKD: it splits "é" into "e" plus an accent mark, and ascii-ignore then drops the mark
    text = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()

    #  replace (- ,  _ ) other than space ,letter or number with a space, not "": "foo-bar" must become two words, not "foobar"
    text = re.sub(r'[^a-z0-9\s]',' ',text)
    return " ".join(w for w in text.split() if w not in LEGAL_SUFFIXES
                    )
