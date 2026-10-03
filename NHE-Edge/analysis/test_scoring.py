"""Unit tests for the scoring contract. Run: python analysis/test_scoring.py

The assertions below are load-bearing: every rate computed on the single-task
benchmark depends on them. They are executable, not prose. The cases mirror the
substring hazards that `audit_single_task.py` found in topics.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scoring import (explain, first_sentence, fold,  # noqa: E402
                     normalise_alternatives, score_both, strict, substring,
                     token_match)

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


def main():
    # ---- folding ----
    check("fold accents", fold("São Tomé"), "sao tome")
    check("fold punctuation", fold("Washington, D.C."), "washington d c")
    check("fold apostrophe", fold("Saint John's"), "saint john s")
    check("fold collapse", fold("  Port-au-Prince "), "port au prince")
    check("fold None", fold(None), "")

    # ---- first sentence ----
    check("cuts at dot",
          first_sentence("The capital is **Praia**. Also Libreville."), "The capital is **Praia**")
    check("cuts at newline",
          first_sentence("The capital is **Praia**\nNext"), "The capital is **Praia**")
    check("empty", first_sentence(""), "")

    # ---- token boundary closes the substring hazards ----
    # 'bern' must NOT match 'berne': different tokens.
    check("bern not in berne", token_match("the capital is berne", "bern"), False)
    check("berne matches itself", strict("The capital is Berne.", ["Berne"]), 1)
    check("bern+berne both listed", strict("The capital is Berne.", ["Bern", "Berne"]), 1)
    check("bern alone against berne is wrong", strict("The capital is Berne.", ["Bern"]), 0)
    # partial token
    check("porto novok rejected", token_match("porto novok", "porto novo"), False)
    check("porto novo in hyphenated", strict("It is Porto-Novo.", ["Porto Novo"]), 1)

    # 'delhi' inside 'new delhi': this is a TRUTH-SET issue, fixed by the merge.
    # With the merged set (both forms present) it is correct; with only 'Delhi'
    # the token matcher still finds 'delhi' as its own token run, so the item is
    # counted correct. That is the documented limitation of token matching, and
    # `report_answer_set_gaps` measures its exposure.
    check("delhi token present in new delhi (documented limitation)",
          token_match("the capital of india is new delhi", "delhi"), True)
    check("new delhi matches merged set",
          strict("The capital of India is New Delhi.", ["New Delhi", "Delhi"]), 1)

    # ---- accent folding ----
    check("accent-insensitive", strict("It is Yaounde.", ["Yaoundé"]), 1)
    check("accent-insensitive multiword", strict("It is Sao Tome.", ["São Tomé"]), 1)

    # ---- alternatives normalization ----
    check("alternatives deduped", normalise_alternatives(["Bern", "bern", ""]), ["bern"])
    check("alternatives folded", normalise_alternatives(["São Tomé", "Sao Tome"]),
          ["sao tome"])

    # ---- multiword + long answers ----
    check("multiword answer", strict("Located in the north, Ulaanbaatar is it.",
                                     ["Ulaanbaatar"]), 1)
    check("long multiword answer", strict("It is Sri Jayawardenepura Kotte.",
                                          ["Sri Jayawardenepura Kotte"]), 1)

    # ---- legacy vs primary divergence ----
    check("legacy accepts bern in berne", substring("The capital is Berne.", ["Bern"]), 1)
    check("primary rejects bern in berne", strict("The capital is Berne.", ["Bern"]), 0)
    check("both agree on clean case",
          score_both("The capital is Praia.", ["Praia"]), (1, 1))

    # ---- degenerate inputs must not crash or score correct ----
    check("no answers", strict("Anything", []), 0)
    check("None text", strict(None, ["Praia"]), 0)
    check("None answer", strict("Praia", [None]), 0)
    check("empty answer", strict("Praia", [""]), 0)
    check("empty sentence", strict("", ["Praia"]), 0)
    check("numeric answer", strict("It is 42.", ["42"]), 1)

    # ---- explain() ----
    ok, sub, matched, sent = explain("The capital is Berne.", ["Bern"])
    check("explain rejects bern/berne", ok, 0)
    check("explain legacy still 1", sub, 1)
    check("explain matched None", matched, None)
    ok, sub, matched, sent = explain("It is Praia.", ["Praia"])
    check("explain accepts praia", ok, 1)
    check("explain matched praia", matched, "praia")

    if FAILURES:
        print(f"FAILED {len(FAILURES)} assertion(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("scoring contract: all assertions pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
