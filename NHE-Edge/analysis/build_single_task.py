"""Merge topics.py into ONE single-task benchmark with an auditable truth layer.

WHY A MERGE AND WHY IT IS NOT TRIVIAL
    The deployment story is a single-task edge model, so the benchmark must be a
    single task with enough items for statistical power. topics.py already holds
    645 rows over 9 topics, but they are the SAME task asked with three different
    question frames (capital / largest city / element symbol) and 155 questions
    are duplicated across topics, 11 of them with DISJOINT answer sets.

    Blind concatenation would double-count duplicated questions and would let a
    question appear with two different truths. The audit in
    `audit_single_task.py` found, for example, "largest city in Malawi" answered
    as Blantyre in one topic and Lilongwe in another - both are defensible, which
    is precisely why the union must be explicit and machine-readable.

WHAT THIS BUILDS
    results/bench_single_task.json:
      items:  [{id, question, answers[], family, source_topics[], primary}]
    where `answers` is the UNION of every defensible accepted answer for that
    question across topics, `primary` is the canonical string, and `family` is
    the question frame. Every union decision that added an answer beyond the
    first topic's row is written to results/bench_single_task_provenance.json.

SCORING CONTRACT (this is the part that decides whether the benchmark is valid)
    The committed headline metric is "strict first sentence, answer as substring".
    That is unsafe here: 'bern' is a substring of 'berne', 'delhi' of 'new delhi',
    'malta' of 'malterney'. `analysis/scoring.py` therefore defines whole-token
    matching and is unit-tested by `analysis/test_scoring.py`; every script that
    scores this benchmark imports it, so the metric cannot drift between the
    baseline run, the intervention run and the analysis.

Usage:
  python analysis/build_single_task.py            # writes bench + provenance
  python analysis/build_single_task.py --verify   # re-derives and diffs
"""
import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))

import topics  # noqa: E402

RES = os.path.join(EDGE_ROOT, "results")

TOPIC_NAMES = ["AFRICA", "EUROPE", "AFRICA_LARGEST", "WORLD_TRICKY",
               "WORLD_CAP_TRAPS", "WORLD_LARGEST", "ELEMENTS", "ASIA", "US_STATES"]

# Question frames, keyed by the question text itself. Derived, not guessed: the
# frame decides which family a duplicate question belongs to and is what a
# "single task" claim has to be precise about.
FAMILY_ORDER = ["element", "capital", "largest", "other"]


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fold(s):
    """Scoring key: accent- and case-insensitive, punctuation-collapsed."""
    t = norm(s)
    out = []
    for ch in t:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def family_of(q):
    ql = q.lower()
    if "chemical symbol" in ql or "atomic number" in ql:
        return "element"
    if "capital of" in ql:
        return "capital"
    if "largest city in" in ql:
        return "largest"
    return "other"


def build():
    qmap = defaultdict(list)
    for name in TOPIC_NAMES:
        for idx, row in enumerate(getattr(topics, name)):
            qmap[row[0]].append({"topic": name, "row_index": idx,
                                 "answers": list(row[1:])})

    items = []
    provenance = []
    for i, (q, entries) in enumerate(sorted(qmap.items(), key=lambda kv: kv[0])):
        # Deterministic primary: the first topic in TOPIC_ORDER that carries the
        # question. TOPIC_NAMES is already in a fixed, documented order.
        entries_sorted = sorted(entries, key=lambda e: (TOPIC_NAMES.index(e["topic"]),
                                                        e["row_index"]))
        primary = entries_sorted[0]["answers"][0]
        union = []
        for e in entries_sorted:
            for a in e["answers"]:
                if a not in union:
                    union.append(a)
        added = [a for a in union[1:]]
        if len(entries) > 1 or added:
            provenance.append({
                "question": q,
                "source_topics": [e["topic"] for e in entries_sorted],
                "source_answers": {e["topic"]: e["answers"] for e in entries_sorted},
                "primary": primary,
                "union": union,
                "answers_added_beyond_primary_row": added,
                "reason": ("duplicate question across topics; union keeps every "
                           "defensible accepted answer so no correct answer is "
                           "scored as an error"),
            })
        items.append({"id": i, "question": q, "answers": union, "primary": primary,
                      "family": family_of(q),
                      "source_topics": [e["topic"] for e in entries_sorted],
                      "n_sources": len(entries_sorted)})

    return items, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and compare against the committed files")
    a = ap.parse_args()

    items, prov = build()
    fam_counts = defaultdict(int)
    for it in items:
        fam_counts[it["family"]] += 1

    payload = {
        "task": "single_task_geographic_factual_recall",
        "description": ("one task, three question frames (capital / largest city / "
                        "chemical symbol); merged from topics.py with an explicit "
                        "answer-union truth layer"),
        "scoring": {
            "primary": "strict first sentence, word-boundary match, accent/case folded",
            "legacy_available": "plain substring (reported for comparison only)",
        },
        "n_items": len(items),
        "family_counts": dict(fam_counts),
        "n_duplicate_questions": sum(1 for it in items if it["n_sources"] > 1),
        "n_rows_with_added_answers": sum(1 for p in prov
                                         if p["answers_added_beyond_primary_row"]),
        "items": items,
    }

    bench_path = os.path.join(RES, "bench_single_task.json")
    prov_path = os.path.join(RES, "bench_single_task_provenance.json")

    if a.verify:
        if not os.path.exists(bench_path):
            sys.exit("no committed bench_single_task.json to verify against")
        old = json.load(open(bench_path, encoding="utf-8"))
        same = (old.get("n_items") == payload["n_items"]
                and old.get("items") == payload["items"])
        print(f"verify: items identical = {same} (n={payload['n_items']})")
        if not same:
            sys.exit("MISMATCH: committed bench differs from a fresh build")
        return

    with open(bench_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    with open(prov_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"n_duplicate_questions": sum(1 for it in items if it["n_sources"] > 1),
                   "decisions": prov}, fh, indent=1, ensure_ascii=False)

    print(f"wrote {bench_path}")
    print(f"  n_items={payload['n_items']} families={dict(fam_counts)}")
    print(f"  duplicate questions={payload['n_duplicate_questions']} "
          f"rows_gaining_answers={payload['n_duplicate_questions']}")
    print(f"wrote {prov_path} ({len(prov)} decisions)")
    print("\nquestions whose accepted-answer set was WIDENED by the merge:")
    for p in prov:
        if p["answers_added_beyond_primary_row"]:
            print(f"  {p['question']}")
            print(f"    {p['source_answers']}")
            print(f"    -> union {p['union']}")


if __name__ == "__main__":
    main()
