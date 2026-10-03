"""Score the gate-driven live run against the deployed baseline, same protocol.

hallucination_rate alone cannot tell a real fix from a coincidence, because a
method can fire more, break more, or land on the same rate by different flips.
This compares the committed baseline, the deployed L19/w5 arm, and any
gate-driven arm(s) using the project's primary metric: strict first-sentence
scoring with W2C / C2W counts against the same baseline file.

Usage:
  python analysis/gate_compare.py
"""
import json
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
REPO_ROOT = os.path.dirname(EDGE_ROOT)
RES = os.path.join(EDGE_ROOT, "results")
sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))
import topics  # noqa: E402

ALTS = {}
for _n in ("AFRICA", "EUROPE", "AFRICA_LARGEST"):
    for _t in getattr(topics, _n):
        ALTS[_t[0]] = list(_t[1:])


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def strict(gen, answer, question):
    m = re.split(r"[.\n]", (gen or "").strip())
    f = norm(m[0]) if m and m[0].strip() else None
    alts = [answer] + list(ALTS.get(question, []))
    return 1 if (f and any(norm(x) in f for x in alts)) else 0


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def arm(name):
    d = load(name)
    if not d:
        return None
    return {r["id"]: strict(r["generated"], r["answer"], r["question"]) for r in d["results"]}


def main():
    base_name = "eval_africa_baseline.json"
    base = arm(base_name)
    if base is None:
        sys.exit("missing eval_africa_baseline.json")
    n = len(base)
    n_wrong_base = sum(1 for v in base.values() if not v)
    print(f"baseline: {n_wrong_base}/{n} strict-wrong\n")

    arms = {}
    for f in sorted(os.listdir(RES)):
        if not f.startswith("eval_runtime_africa_jump_gt") or not f.endswith(".json"):
            continue
        if "none" in f or "abstain" in f:
            continue
        arms[f] = arm(f)

    print(f"{'arm':<58} {'W2C':>4} {'C2W':>4} {'net':>5} {'hall':>7}")
    print("-" * 82)
    rows = []
    for name, sc in arms.items():
        if sc is None or len(sc) != n:
            continue
        w2c = sum(1 for i, v in sc.items() if not base[i] and v)
        c2w = sum(1 for i, v in sc.items() if base[i] and not v)
        hall = sum(1 for v in sc.values() if not v)
        rows.append((w2c - c2w, w2c, c2w, hall, name))
    for net, w2c, c2w, hall, name in sorted(rows, key=lambda r: (-r[0], r[4])):
        print(f"{name:<58} {w2c:>4} {c2w:>4} {net:>+5} {hall:>3}/{n}")

    print("\nReading: net = W2C - C2W. A positive net with C2W=0 is a clean repair.")
    print("An arm whose best offline AUROC still yields net<=0 shows that AUROC alone")
    print("does not identify an intervenable cell; the gate ranks, the live run decides.")


if __name__ == "__main__":
    main()
