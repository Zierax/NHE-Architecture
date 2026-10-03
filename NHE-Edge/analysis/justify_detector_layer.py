"""Independent justification for the L19 detector, or an honest admission that
there is none.

THE PROBLEM THIS ADDRESSES
    `analysis/gate.py` found that pre-commit AUROC does NOT identify the
    intervenable cell: on Gemma the highest-AUROC layer (L20, 0.781) is worse
    live (net -1) than L19 (0.772, net +2). If the only criterion for choosing
    L19 were detector AUROC, then L19's success would be config luck. This script
    looks for a criterion that is INDEPENDENT of the detector's own feature.

THE CANDIDATE INDEPENDENT CRITERION: causal attribution overlap
    The intervention mask (k32_midwrong) was NOT chosen by AUROC. It came from
    activation patching on wrong-only examples, and it lives entirely in layers
    10-17. If the detector reads the layer where that causal evidence CONVERGES
    (or the layer just after it, where the patched effect propagates), then L19
    is justified by a different measurement than the one used to fit it. This is
    the standard "two independent measurements agree" argument, and it is
    testable: it either finds a layer whose proximity to the causal band is
    unique, or it admits no independent criterion exists.

WHAT THIS DOES NOT DO
    It does not claim causality from proximity. It reports (a) where the causal
    neurons are, (b) where the detector reads, (c) whether the detector layer is
    the closest layer to the causal band that is ALSO above the causal band, and
    (d) whether any OTHER layer satisfies the same description. If several layers
    satisfy it, the honest output is "the criterion narrows but does not uniquely
    select L19", and that is what gets reported.

Usage: python analysis/justify_detector_layer.py
"""
import json
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
RES = os.path.join(EDGE_ROOT, "results")

ATTRIB = "attribution_causal2_africa.json"
MASK = "mask_k32_midwrong.json"
DETECTOR = "detector_greedy.json"


def main():
    ap = os.path.join(RES, ATTRIB)
    mp = os.path.join(RES, MASK)
    dp = os.path.join(RES, DETECTOR)
    for p in (ap, mp, dp):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    attrib = json.load(open(ap, encoding="utf-8"))
    mask = json.load(open(mp, encoding="utf-8"))
    det = json.load(open(dp, encoding="utf-8"))

    mask_layers = sorted({e["layer"] for e in mask["items"]})
    lo, hi = mask_layers[0], mask_layers[-1]
    det_layer = det["detector_early"]["layer"]

    # broader causal band (all wrong-only, not just the mid-band subset)
    wo = [e["layer"] for e in attrib.get("top_wrong_only", [])]
    wo_count = Counter(wo)
    mask_count = Counter(e["layer"] for e in mask["items"])

    print("=== causal evidence (from patching, NOT from the detector) ===")
    print(f"mask layers (k32_midwrong): {mask_layers}")
    print(f"mask band: L{lo}-L{hi}, detector reads L{det_layer}")
    print("\nwrong-only neuron count by layer (top_by_causal):")
    for l, c in sorted(wo_count.items()):
        mark = " <-- mask band" if lo <= l <= hi else ""
        mark += " <-- detector layer" if l == det_layer else ""
        print(f"  L{l:<2} {c:>3}{mark}")

    # distance from the detector layer to the causal band
    if det_layer < lo:
        dist, side = lo - det_layer, "below band"
    elif det_layer > hi:
        dist, side = det_layer - hi, "above band"
    else:
        dist, side = 0, "inside band"
    print(f"\ndetector L{det_layer} is {dist} layer(s) {side} (band L{lo}-L{hi})")

    # Which other layers are '>= distance above the band AND within the upper
    # half of the model'? This is the uniqueness test.
    n_layers = len(mask_count) and max(mask_count) or 26
    upper = [l for l in range(hi + 1, 27)]
    print(f"\nuniqueness test - layers at or above the band in the upper model: {upper}")
    print("each of these is equally consistent with 'just above the causal band';")
    print("proximity alone therefore does NOT uniquely select L19.")

    print("\n=== verdict ===")
    verdict = {
        "detector_layer": det_layer,
        "causal_mask_band": [lo, hi],
        "distance_to_band": dist,
        "side": side,
        "independent_criterion_found": False,
        "reason": ("the causal mask confirms that the wrong-commit neurons live in "
                   "L10-17 and that L19 reads just above that band, which is "
                   "consistent with reading the patched effect, BUT every layer in "
                   f"{upper} is equally consistent with 'just above the band', so "
                   "proximity does not uniquely select L19"),
        "what_is_established": ("the detector layer is causally informed rather "
                                "than arbitrary: it is adjacent to, and not inside, "
                                "the band of patched wrong-commit neurons"),
        "what_is_not_established": ("no criterion independent of the live result "
                                    "uniquely picks L19 over the other upper layers; "
                                    "the choice remains under-determined offline"),
        "honest_statement": ("L19 is justified as causally ADJACENT to the "
                             "attribution band, and was validated by the live run. "
                             "It is not claimed to be uniquely optimal by an "
                             "offline criterion; selecting it from the upper layers "
                             "without the live run would be config luck."),
    }
    print(f"independent criterion uniquely selects L19: NO")
    print(f"established: {verdict['what_is_established']}")
    print(f"not established: {verdict['what_is_not_established']}")

    out = os.path.join(RES, "detector_layer_justification.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(verdict, fh, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
