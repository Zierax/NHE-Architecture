"""Audit topics.py before it is used as a single-task benchmark.

ZERO-COMPROMISE rationale: a benchmark is a measurement instrument. If two rows
in the instrument disagree about ground truth, every downstream number inherits
the disagreement. This audit reports, without judgement:

  - per-topic n and the total
  - duplicate questions inside a topic and across topics
  - DUPLICATE QUESTIONS WITH DISJOINT ANSWER SETS (a contradiction, not a
    formatting variant)
  - answers that are substrings of each other inside one row (scoring hazard:
    "Male" is a substring of "Malenje", and a substring scorer would accept
    the wrong one)
  - the effective baseline headroom implied by each topic's committed error rate

Nothing here mutates topics.py. It produces evidence for the merge step.
"""
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

TOPIC_NAMES = ["AFRICA", "EUROPE", "AFRICA_LARGEST", "WORLD_TRICKY",
               "WORLD_CAP_TRAPS", "WORLD_LARGEST", "ELEMENTS", "ASIA", "US_STATES"]


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def main():
    report = {"topics": {}, "contradictions": [], "substring_hazards": [],
              "cross_topic_duplicates": {}}

    total = 0
    for name in TOPIC_NAMES:
        rows = list(getattr(topics, name))
        total += len(rows)
        by_q = defaultdict(set)
        for row in rows:
            q = row[0]
            answers = {norm(a) for a in row[1:]}
            by_q[q].add(frozenset(answers))
        dupes_in_topic = {q: len(v) for q, v in by_q.items() if len(v) > 1}
        report["topics"][name] = {
            "n": len(rows),
            "unique_questions": len(by_q),
            "duplicate_question_with_different_answers": dupes_in_topic,
            "rows_with_alternates": sum(1 for r in rows if len(r) > 2),
        }
        for q, sets in dupes_in_topic.items():
            report["contradictions"].append({
                "topic": name, "question": q,
                "answer_sets": [sorted(s) for s in sorted(sets, key=lambda x: sorted(x))],
            })
        # substring hazard inside one row
        for row in rows:
            answers = [norm(a) for a in row[1:]]
            for i, a in enumerate(answers):
                for j, b in enumerate(answers):
                    if i != j and a != b and a in b:
                        report["substring_hazards"].append({
                            "topic": name, "question": row[0],
                            "shorter": a, "longer": b,
                            "risk": "substring scorer accepts the wrong answer",
                        })

    # cross-topic duplicate questions
    qmap = defaultdict(list)
    for name in TOPIC_NAMES:
        for row in getattr(topics, name):
            qmap[row[0]].append((name, frozenset(norm(a) for a in row[1:])))
    cross = {}
    for q, entries in qmap.items():
        if len(entries) > 1:
            all_sets = {e[1] for e in entries}
            cross[q] = {
                "topics": [e[0] for e in entries],
                "consistent": len(all_sets) == 1,
                "answer_sets": [sorted(s) for s in sorted(all_sets, key=lambda x: sorted(x))],
            }
    report["cross_topic_duplicates"] = cross
    report["total_rows"] = total
    report["unique_questions_total"] = len(qmap)

    # committed baseline error rates (strict, greedy) if present
    RES = os.path.join(EDGE_ROOT, "results")
    for name in TOPIC_NAMES:
        fn = f"eval_{name.lower()}_baseline.json"
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            n = len(d["results"])
            w = sum(1 for r in d["results"] if not r["correct"])
            report["topics"][name]["committed_baseline_n"] = n
            report["topics"][name]["committed_baseline_wrong"] = w
            report["topics"][name]["headroom_items"] = w

    out = os.path.join(RES, "single_task_audit.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)

    print(f"total rows={total} unique questions={len(qmap)}")
    print(f"\n{'topic':<18} {'n':>4} {'uniq':>5} {'alt':>4} {'baseline_wrong':>15}")
    for name in TOPIC_NAMES:
        t = report["topics"][name]
        print(f"{name:<18} {t['n']:>4} {t['unique_questions']:>5} {t['rows_with_alternates']:>4} "
              f"{t.get('committed_baseline_wrong', '-'):>15}")

    print(f"\nIN-TOPIC contradictions: {len(report['contradictions'])}")
    for c in report["contradictions"]:
        print(f"  [{c['topic']}] {c['question']} -> {c['answer_sets']}")

    print(f"\nsubstring hazards: {len(report['substring_hazards'])}")
    for h in report["substring_hazards"][:20]:
        print(f"  [{h['topic']}] {h['question']}: '{h['shorter']}' inside '{h['longer']}'")

    incons = {q: v for q, v in cross.items() if not v["consistent"]}
    print(f"\ncross-topic duplicate questions: {len(cross)} "
          f"(INCONSISTENT: {len(incons)})")
    for q, v in list(incons.items())[:20]:
        print(f"  {q}")
        print(f"    topics={v['topics']}")
        for s in v["answer_sets"]:
            print(f"    {s}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
