import os
import pickle
import unicodedata

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

WORD_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "12": "twelve", "15": "fifteen", "24": "twenty-four",
    "30": "thirty", "32": "thirty-two", "50": "fifty", "52": "fifty-two",
    "54": "fifty-four", "100": "one hundred", "149": "one hundred forty-nine",
    "1889": "eighteen eighty-nine", "206": "two hundred six",
}

def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

def variants(ans, alt):
    out = {norm(ans)}
    for a in alt:
        out.add(norm(a))
    if ans.isdigit() and ans in WORD_MAP:
        out.add(WORD_MAP[ans])
    return out

counts = {0: 0, 1: 0}
changed = 0
for f in sorted(os.listdir(OUT_DIR)):
    if not f.startswith("ex_") or not f.endswith(".pkl"):
        continue
    p = os.path.join(OUT_DIR, f)
    with open(p, "rb") as fh:
        rec = pickle.load(fh)
    gen_n = norm(rec["generated"])
    new_label = 1 if any(v in gen_n for v in variants(rec["answer"], rec.get("alt_answers", []))) else 0
    if new_label != rec["label"]:
        changed += 1
        print(f"{f}: label {rec['label']} -> {new_label} | {rec['generated'][:60]!r}")
    rec["label"] = new_label
    counts[new_label] += 1
    with open(p, "wb") as fh:
        pickle.dump(rec, fh)

print(f"\ntotal={sum(counts.values())} truthful={counts[1]} hallucinated={counts[0]} (changed={changed})")