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
    return " ".join(w for w in text.split() if w not in LEGAL_SUFFIXES)


STREET_ABBR = {"cir": "circle", "dr": "drive", "ave": "avenue", "blvd": "boulevard",
                   "st": "street", "rd": "road", "ln": "lane", "ct": "court"}

# WHY: one native name per state, in that state's own language; Hindi only for Hindi-belt states
NATIVE_STATE = {
        "ఆంధ్రప్రదేశ్": "andhra pradesh", "అరుణాచల్ ప్రదేశ్": "arunachal pradesh",
        "अरुणाचल प्रदेश": "arunachal pradesh", "অসম": "assam", "बिहार": "bihar",
        "छत्तीसगढ़": "chhattisgarh", "गोवा": "goa", "ગુજરાત": "gujarat",
        "हरियाणा": "haryana", "हिमाचल प्रदेश": "himachal pradesh",
        "झारखंड": "jharkhand", "ಕರ್ನಾಟಕ": "karnataka", "കേരളം": "kerala",
        "मध्य प्रदेश": "madhya pradesh", "महाराष्ट्र": "maharashtra",
        "মণিপুর": "manipur", "मेघालय": "meghalaya", "मिज़ोरम": "mizoram",
        "नागालैंड": "nagaland", "ଓଡ଼ିଶା": "odisha", "ਪੰਜਾਬ": "punjab",
        "राजस्थान": "rajasthan", "सिक्किम": "sikkim", "தமிழ்நாடு": "tamil nadu",
        "తెలంగాణ": "telangana", "ত্রিপুরা": "tripura",
        "उत्तर प्रदेश": "uttar pradesh", "उत्तराखंड": "uttarakhand",
        "পশ্চিমবঙ্গ": "west bengal", "दिल्ली": "delhi"
    }

#  "N/A" becomes the tokens n and a after punctuation is removed; the rest are other
# placeholder words that would otherwise count as shared tokens between unrelated addresses
JUNK_TOKENS = {"n", "a", "na", "nan", "null", "none", "nil", "unknown"}



def norm_address(addr) -> frozenset:
    #  NaN != NaN, so this catches missing addresses without importing pandas
    if addr is None or addr != addr:
        return frozenset()
    text = str(addr)


    #  first: the ASCII step below would delete native-script letters
    for native,english in NATIVE_STATE.items():
        text = text.replace(native,english)

        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # WHY: kills ##, commas, "351-"
        text = " ".join(STREET_ABBR.get(w,w) for w in text.split())

        #  \b: match a state code only as a whole word (not inside "maryland")
        for abbr,full in STREET_ABBR.items():
            text = re.sub(rf"\b{abbr}\b", full, text)

        #frozen : making it hashable to maek it key and set: beacause address are jumbled and we need to compare them in a set way
        return frozenset(w for w in text.split() if w not in JUNK_TOKENS)