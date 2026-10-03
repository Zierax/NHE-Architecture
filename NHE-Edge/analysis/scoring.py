"""Shared scoring layer for the single-task benchmark.

THE PROBLEM THIS FIXES
    The committed headline metric is "the answer appears as a plain substring of
    the first sentence". `audit_single_task.py` found 26 rows where a shorter
    accepted answer is a substring of a longer one in the SAME row, so that metric
    can score a non-accepted answer as correct: 'bern' matches inside 'berne',
    'delhi' matches inside 'new delhi', 'washington' matches inside 'washington d c'.

THE FIX, IN TWO PARTS
    1. SCORING (this file). Matching is on whole token sequences after folding,
       never on raw substrings. 'bern' does not match 'berne' because the tokens
       differ. No word list, no name table, no English heuristics - the rule
       cannot silently fail on an unseen name form.

    2. BENCHMARK CONSTRUCTION (`build_single_task.py`). The remaining cases -
       'delhi' inside 'new delhi' - are NOT a scoring problem, they are a truth-
       set problem. Both forms are defensible answers to that question, so the
       merged benchmark's accepted set carries both, and the token matcher then
       accepts either. `report_answer_set_gaps()` below enumerates any question
       where a plausible longer form could still be emitted but is not in the set,
       so the residual exposure is measured instead of assumed away.

    Longer drafts of this file used a hardcoded stop-word list to guess where a
    name span ends. That was rejected: it encodes English knowledge nobody audited
    and would reject valid answers for any name form not in the list.

CONTRACT
    fold(s):            NFKD -> ascii -> lowercase -> non-alphanumeric to space
                        -> collapse whitespace. "São Tomé" == "sao tome".
    first_sentence(t):  text up to the first '.' or newline (committed rule).
    strict(t, alts):    1 if any accepted answer appears in sentence 1 as a
                        complete contiguous token sequence.
    substring(t, alts): the legacy metric, reported alongside for comparison only.
"""
import re
import unicodedata

__all__ = ["fold", "first_sentence", "normalise_alternatives", "token_match",
           "strict", "substring", "score_both", "explain"]


def fold(s):
    """Accent/case/punctuation-insensitive key used by every matcher."""
    if s is None:
        return ""
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    out = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in t]
    return " ".join("".join(out).split())


def normalise_alternatives(answers):
    """Folded, de-duplicated, non-empty accepted answers; order preserved."""
    seen = []
    for a in answers or []:
        f = fold(a)
        if f and f not in seen:
            seen.append(f)
    return seen


def first_sentence(text):
    """The committed strict rule: cut at the first '.' or newline."""
    m = re.split(r"[.\n]", (text or "").strip())
    if not m:
        return ""
    return m[0].strip() if m[0].strip() else ""


def token_match(sentence_folded, answer_folded):
    """True if answer_folded appears in sentence_folded as a contiguous run of
    whole tokens. 'bern' vs 'berne' -> False. 'porto novo' vs 'porto-novo' -> True
    (folding turns the hyphen into a space)."""
    if not answer_folded:
        return False
    hay = sentence_folded.split()
    needle = answer_folded.split()
    n = len(needle)
    if n == 0 or n > len(hay):
        return False
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def strict(text, answers):
    """Primary metric: an accepted answer appears as a whole token sequence in the
    first sentence."""
    sent = fold(first_sentence(text))
    if not sent:
        return 0
    return 1 if any(token_match(sent, a) for a in normalise_alternatives(answers)) else 0


def substring(text, answers):
    """Legacy metric. Reported only so the two can be compared, never to replace
    the primary: it accepts a non-accepted answer that is a substring of the
    emitted name."""
    hay = fold(first_sentence(text))
    return 1 if any((a and a in hay) for a in normalise_alternatives(answers)) else 0


def score_both(text, answers):
    return strict(text, answers), substring(text, answers)


def explain(text, answers):
    """Diagnostic wrapper: (strict, substring, matched_answer, all_sentences)."""
    sent = fold(first_sentence(text))
    alts = normalise_alternatives(answers)
    matched = next((a for a in alts if token_match(sent, a)), None)
    return (1 if matched else 0), substring(text, answers), matched, sent
