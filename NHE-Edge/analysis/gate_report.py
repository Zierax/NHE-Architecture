"""Print the ranked applicability-gate cells from a report.

Usage:
  python analysis/gate_report.py                          # the sweep report
  python analysis/gate_report.py ../results/applicability_gate_gemma3-1b.json
"""
import json
import os
import sys

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "applicability_gate_sweep.json")
d = json.load(open(path, encoding="utf-8"))
for r in d["results"]:
    print("===", r["model"], r["verdict"])
    if "top_precommit_cells" in r:
        cs = r["top_precommit_cells"]
        fx = r.get("top_fixed_slice_cells", [])
    else:
        cs = sorted([c for c in r["cells"] if c["precommit_auroc"] is not None],
                    key=lambda c: -c["precommit_auroc"])
        fx = sorted([c for c in r["fixed_slice_cells"] if c["fixed_slice_auroc"] is not None],
                    key=lambda c: -c["fixed_slice_auroc"])
    print(" top10 pre-commit (adaptive) cells:")
    for c in cs[:10]:
        print(f"   L{c['layer']:<2} w{c['window']:<3} pre={c['precommit_auroc']:<8} "
              f"win={c['window_auroc']:<8}")
    print(" top6 fixed-slice cells (edge-effect controlled):")
    for c in fx[:6]:
        print(f"   L{c['layer']:<2} cut={c['fixed_cut']:<3} fixed={c['fixed_slice_auroc']}")
    counts = r.get("n_cells_precommit_ge")
    if counts:
        for t, n in sorted(counts.items(), key=lambda kv: -float(kv[0])):
            print(f"   adaptive cells with AUROC >= {t}: {n}")
    print(f" verdict_basis={r.get('verdict_basis')}")
    if r.get("verdict_note"):
        print(f" note: {r['verdict_note']}")
