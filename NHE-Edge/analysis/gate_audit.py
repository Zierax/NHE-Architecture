"""Audit the applicability gate's own assumptions before trusting its verdict.

The gate ranks cells by pre-commit AUROC. If `commit_position` returns the same
value for every item, the pre-commit segment is constant and the AUROC is
measuring one fixed slice of the sequence, not an adaptive early-exit window.
That would make an ACTIVE verdict an artifact of the definition, not evidence.

This script prints, per model:
  - the distribution of commit positions (are they degenerate?)
  - per-layer pre-commit AUROC with its effective segment length
  - AUROC computed on a FIXED pre-commit slice (min commit pos) as a control
  - for the ranked-best layer, whether hallucination and correct items differ
    in commit position at all (if they do, the AUROC may be a position proxy)

Usage: python analysis/gate_audit.py [--flows-model qwen2.5-0.5b]
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
REPO_ROOT = os.path.dirname(EDGE_ROOT)
RES = os.path.join(EDGE_ROOT, "results")
sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))

import gate  # noqa: E402


def audit(model_key):
    case = gate.load_case(model_key)
    if case is None:
        print(f"{model_key}: NO_FLOWS")
        return
    flows, texts, labels, path = case
    y = 1 - labels
    city = gate.city_positions(model_key, texts)
    cp = np.array([-1 if c is None else int(c) for c in city])

    print(f"\n=== {model_key} ({os.path.basename(path)}) ===")
    print(f"items={len(flows)} hall={int(y.sum())} commit_pos found={int((cp >= 0).sum())}")
    vals, cnts = np.unique(cp[cp >= 0], return_counts=True)
    dist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, cnts))
    print(f"commit_pos distribution: {dist}")
    if len(vals) == 1:
        print("  !! DEGENERATE: every item commits at the same position.")
    hall_cp = cp[y == 1]
    corr_cp = cp[y == 0]
    hall_cp = hall_cp[hall_cp >= 0]
    corr_cp = corr_cp[corr_cp >= 0]
    if len(hall_cp) and len(corr_cp):
        print(f"  commit_pos hall mean={hall_cp.mean():.2f}  correct mean={corr_cp.mean():.2f}")
        print(f"  position alone separates groups? hall={sorted(set(hall_cp.tolist()))} "
              f"correct={sorted(set(corr_cp.tolist()))}")

    n_layers = max(f.shape[1] for f in flows) - 1
    rows = []
    for L in range(max(0, n_layers - 18), n_layers):
        if L + 1 > max(f.shape[1] for f in flows) - 1:
            continue
        pre_var, fixed_auc, adapt_auc = [], [], []
        fixed_cut = int(np.median(cp[cp >= 0])) if (cp >= 0).any() else 0
        for i, f in enumerate(flows):
            h = f[:, L + 1, :].astype(np.float32)
            j = np.linalg.norm(h[1:] - h[:-1], axis=1)
            cut = cp[i] if cp[i] > 0 else len(j)
            seg = j[:cut]
            pre_var.append(float(seg.max()) if len(seg) else 0.0)
            fs = j[:fixed_cut]
            fixed_auc.append(float(fs.max()) if len(fs) else 0.0)
            adapt_auc.append(float(seg.max()) if len(seg) else 0.0)
        rows.append((L, gate._auc(y, fixed_auc), gate._auc(y, adapt_auc),
                     float(np.mean([1 if c > 0 else 0 for c in cp]))))
    print(f"  {'layer':>5} {'fixed-slice AUC':>16} {'adaptive AUC':>14}  (fixed_cut={fixed_cut})")
    for L, fa, aa, frac in rows:
        print(f"  {L:>5} {fa:>16.4f} {aa:>14.4f}   usable={frac:.2f}")
    best = max((r for r in rows if not np.isnan(r[1])), key=lambda r: r[1], default=None)
    if best:
        print(f"  best fixed-slice layer: L{best[0]} AUC={best[1]:.4f}")


def main():
    models = ["gemma3-1b", "qwen2.5-0.5b"]
    if len(sys.argv) > 1 and sys.argv[1].startswith("--flows-model"):
        models = [sys.argv[sys.argv.index("--flows-model") + 1]]
    for m in models:
        audit(m)


if __name__ == "__main__":
    main()
