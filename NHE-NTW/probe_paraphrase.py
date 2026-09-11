"""Paraphrase-robustness probe: can we detect planted lies WITHOUT ground truth?

Uses saved NHE-NTW weights (no retraining). Asks each question in K phrasings
built ONLY from in-vocab words (word-level vocab is fixed). Measures per-group:
- agreement: fraction of paraphrases giving the same city as phrasing 0
- acc-vs-truth per phrasing
Prediction: correct items stay put (robust knowledge); planted lies wobble
(memorized to one phrasing); ambiguous split.
If planted agreement << correct agreement, consistency probing is a
ground-truth-free diagnostic for training-data lies.

Usage: python probe_paraphrase.py  (CPU minutes, read-only weights)
Writes: results/paraphrase_proof.json
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

RES = os.path.join(BASE, "results")
SEED = 7

# All words must already be in vocab. P0 = training phrasing.
TEMPLATES = [
    "What is the capital of {c} ? The capital of {c} is",
    "The capital of {c} is",
    "What is the capital of {c} ?",
    "What is the capital of {c} ? The capital of {c} is the",
]

def _city_words():
    from parametric_proof import TRUTH as _T, PLANTED as _P, AMBIG as _A
    s = set(_T.values())
    for _c, (_t, _w) in _P.items():
        s.add(_w)
    for _c, (_t, _a) in _A.items():
        s.add(_a)
    return s


CITY_WORDS = _city_words()


def first_city(words):
    for w in words:
        if w in CITY_WORDS:
            return w
    return words[0] if words else ""


def main():
    import torch
    from parametric_proof import WordTok, build_model, build_corpus
    from parametric_proof import COUNTRIES, PLANTED, AMBIG, TRUTH

    torch.manual_seed(SEED)
    tok = WordTok(build_corpus() + [" ".join(TRUTH.values())])
    # every template word must be in-vocab (except country names, checked below)
    fixed_words = set()
    for t in TEMPLATES:
        fixed_words.update(w for w in t.replace("{c}", "").split() if w not in ("",))
    oov = [w for w in fixed_words if w not in tok.stoi]
    assert not oov, f"OOV template words: {oov}"

    model = build_model(len(tok.stoi))
    model.load_state_dict(torch.load(os.path.join(RES, "parametric_weights.pt"),
                                     map_location="cpu", weights_only=False))
    model.eval()
    groups = {"correct": [c for c, _ in COUNTRIES if c not in PLANTED and c not in AMBIG],
              "planted": list(PLANTED), "ambiguous": list(AMBIG)}
    out = {"seed": SEED, "templates": TEMPLATES, "groups": {}}
    with torch.no_grad():
        for gname, countries in groups.items():
            rows = []
            for c in countries:
                answers, confs = [], []
                for t in TEMPLATES:
                    ids = torch.tensor([tok.enc(t.format(c=c))])
                    lg = model(ids)
                    probs = torch.softmax(lg[0, -1], dim=0)
                    nx = int(probs.argmax())
                    gen, cur = [nx], torch.cat([ids, torch.tensor([[nx]])], dim=1)
                    for _ in range(3):
                        lg2 = model(cur)
                        nx2 = int(lg2[0, -1].argmax())
                        if nx2 == tok.stoi["<eos>"] or tok.itos[nx2] == ".":
                            break
                        gen.append(nx2)
                        cur = torch.cat([cur, torch.tensor([[nx2]])], dim=1)
                    words = tok.dec([g for g in gen if tok.itos[g] != "."]).strip().split()
                    city = first_city(words)
                    answers.append(city)
                    confs.append(round(float(probs[nx]), 4))
                agree = sum(1 for a in answers[1:] if a == answers[0]) / (len(answers) - 1)
                acc0 = answers[0] == TRUTH[c]
                rows.append({"country": c, "truth": TRUTH[c], "answers": answers,
                             "confs": confs, "agreement": round(agree, 3),
                             "p0_correct": acc0})
            ma = sum(r["agreement"] for r in rows) / len(rows)
            a0 = sum(r["p0_correct"] for r in rows) / len(rows)
            out["groups"][gname] = {"n": len(rows), "mean_agreement": round(ma, 3),
                                    "p0_acc": round(a0, 3), "rows": rows}
            print(f"{gname}: mean_agreement={ma:.3f} p0_acc={a0:.3f}", flush=True)
            for r in rows:
                if gname != "correct":
                    print(f"  {r['country']}: {r['answers']} truth={r['truth']}", flush=True)
    with open(os.path.join(RES, "paraphrase_proof.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("saved results/paraphrase_proof.json", flush=True)


if __name__ == "__main__":
    main()
