"""Applicability gate: does NHE-temporal apply to this model at all?

PRIMITIVE
    A pre-commit detector is useful only if the spike lands STRICTLY BEFORE the
    answer token. A model/format cell with no spike before commit is INERT:
    the method cannot fire usefully there no matter how good the AUC looks over
    the whole sequence. This script decides that from data instead of asserting
    it in prose.

WHAT CHALLENGES THE CURRENT PARADIGM
    The deployed Qwen result selected one layer (L22) and one window (w<=5) and
    then reported 0 flips. That conflates two different questions:
      (a) is there ANY pre-commit signal in this model/format?   <- the real gate
      (b) does THIS hand-picked (layer, window) fire before commit? <- config luck
    This script answers (a) by searching the joint space, so "Qwen does not
    work" is falsifiable: if any (layer, window) shows real pre-commit
    discriminability, the "inert" claim dies.

FALSIFICATION
    H: the Qwen-format failure is a property of the model/format, not of the
       chosen layer/window.
    Rejected if any (layer, window) cell reaches pre-commit AUROC >= 0.70 with
    a usable pre-commit firing rate on the greedy Africa set.
    Similarly for Gemma, which is the positive control: a correct gate must
    return ACTIVE for gemma-native.

Usage:
  python analysis/gate.py --model gemma3-1b
  python analysis/gate.py --model qwen2.5-0.5b --flows results/<npz>
  python analysis/gate.py --sweep-models          # both, comparison table
Writes results/applicability_gate.json
"""
import argparse
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

LAYERS = list(range(8, 24))
WINDOWS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
AUROC_GATE = 0.70
PRECOMMIT_AUROC_GATE = 0.70


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def load_case(model_key, flows_path=None):
    if flows_path is None:
        suffix = "" if model_key == "gemma3-1b" else f"_{model_key}"
        flows_path = os.path.join(RES, f"greedy_flows_africa{suffix}.npz")
    if not os.path.exists(flows_path):
        return None
    d = np.load(flows_path, allow_pickle=True)
    flows = list(d["flows"])
    texts = [str(x) for x in d["texts"]]
    labels = np.array(d["labels"]).astype(int)
    return flows, texts, labels, flows_path


def commit_position(tok, gen, family):
    """0-indexed generated-token position where the answer span begins.

    Same convention as the committed timing analysis: for a bold answer it is
    the token after '**', otherwise the first content token after the first
    'is'. Returns None when the span cannot be located.
    """
    ids = tok(gen, add_special_tokens=False)["input_ids"]
    toks = [tok.decode([t]) for t in ids]
    if not toks:
        return None
    for k, t in enumerate(toks):
        s = t.strip()
        if s == "**":
            return k + 1
        if s.startswith("**") and len(s) > 2:
            return k
    for k, t in enumerate(toks):
        if t.strip().lower() == "is":
            j = k + 1
            while j < len(toks) and toks[j].strip() in ("", "**"):
                j += 1
            return j if j < len(toks) else None
    return 0


def city_positions(model_key, texts):
    if model_key.startswith("qwen"):
        from transformers import AutoTokenizer
        import runtime_rollback_qwen as qq
        try:
            cfg = qq.get_model_config(model_key)
            tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
        except Exception:
            return [None] * len(texts)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer"))
    fam = "plain" if model_key.startswith("qwen") else "bold"
    return [commit_position(tok, t, fam) for t in texts]


def evaluate(model_key, flows_path=None, verbose=True):
    case = load_case(model_key, flows_path)
    if case is None:
        return {"model": model_key, "status": "NO_FLOWS",
                "hint": "collect flows first (runtime_rollback collect / --model key collect)"}
    flows, texts, labels, path = case
    y = 1 - labels
    city = city_positions(model_key, texts)
    # flows are (T, n_hidden_states, d_model): dim 0 = time, dim 1 = hidden-state
    # index (0 = embeddings, 1..L = layers). Taking dim 0 here would silently
    # collapse the layer scan to nothing when every sequence is short.
    shapes = {tuple(f.shape) for f in flows}
    if len(shapes) != 1:
        shapes = {(f.shape[0], f.shape[1], f.shape[2]) for f in flows}
    n_layers = max(s[1] for s in shapes) - 1

    rows = []
    for L in LAYERS:
        if L > n_layers:
            continue
        jumps = []
        for f in flows:
            h = f[:, L + 1, :].astype(np.float32)
            jumps.append(np.linalg.norm(h[1:] - h[:-1], axis=1))
        jumps = [j for j in jumps if len(j) > 0]
        if not jumps:
            continue
        for W in WINDOWS:
            # pre-commit-only feature: max jump strictly before the answer token
            pre, full, pre_rate = [], [], []
            for i, j in enumerate(jumps):
                ci = city[i]
                cut = ci if (ci is not None and 0 < ci < len(j) + 1) else len(j)
                seg = j[:cut]
                pre.append(float(seg.max()) if len(seg) else 0.0)
                full.append(float(j[:min(W, len(j))].max()))
                pre_rate.append(1 if (len(seg) and float(seg.max()) > 0) else 0)
            a_pre = _auc(y, pre)
            a_win = _auc(y, full)
            rows.append({"layer": L, "window": W,
                         "precommit_auroc": None if np.isnan(a_pre) else round(a_pre, 4),
                         "window_auroc": None if np.isnan(a_win) else round(a_win, 4),
                         "n_with_presegment": int(sum(pre_rate))})

    scored = [r for r in rows if r["precommit_auroc"] is not None]
    best = max(scored, key=lambda r: r["precommit_auroc"]) if scored else None
    best_win = max(scored, key=lambda r: r["window_auroc"]) if scored else None

    # Edge-effect control. An adaptive pre-commit segment is LONGER for items that
    # commit late, and hallucinations can commit later than correct answers. If
    # that is the only difference, the adaptive AUROC is inflated by a boundary
    # artifact rather than by an early signal. Re-score every layer on ONE fixed
    # slice (median commit position, same cut for every item) and require the
    # verdict to survive that stricter measurement.
    cp = np.array([-1 if c is None else int(c) for c in city])
    fixed_cut = int(np.median(cp[cp >= 0])) if (cp >= 0).any() else 0
    fixed_rows = []
    for L in LAYERS:
        if L > n_layers:
            continue
        vals = []
        for f in flows:
            h = f[:, L + 1, :].astype(np.float32)
            j = np.linalg.norm(h[1:] - h[:-1], axis=1)
            seg = j[:fixed_cut]
            vals.append(float(seg.max()) if len(seg) else 0.0)
        a_fixed = _auc(y, vals)
        fixed_rows.append({"layer": L, "fixed_cut": fixed_cut,
                           "fixed_slice_auroc": None if np.isnan(a_fixed) else round(a_fixed, 4)})
    fixed_scored = [r for r in fixed_rows if r["fixed_slice_auroc"] is not None]
    best_fixed = max(fixed_scored, key=lambda r: r["fixed_slice_auroc"]) if fixed_scored else None
    best_fixed_auc = best_fixed["fixed_slice_auroc"] if best_fixed else float("nan")

    adaptive_auc = best["precommit_auroc"] if best else float("nan")
    inflation = (adaptive_auc - best_fixed_auc) if best_fixed else float("nan")
    verdict = "ACTIVE" if (best_fixed and best_fixed_auc >= PRECOMMIT_AUROC_GATE) else "INERT"
    note = None
    if verdict == "ACTIVE" and inflation > 0.05:
        note = ("adaptive pre-commit AUROC exceeds the fixed-slice AUROC by "
                f"{inflation:.3f}; part of the signal is a commit-position edge effect")
    if verdict == "ACTIVE" and best_fixed:
        note = ((note + "; ") if note else "") + \
            f"use the fixed slice at L{best_fixed['layer']} (cut={fixed_cut}) for any live test"

    out = {"model": model_key, "flows": os.path.basename(path), "n_items": len(flows),
           "n_hallucinations": int(y.sum()),
           "commit_position_distribution": {str(int(v)): int(c) for v, c in
                                            zip(*np.unique(cp[cp >= 0], return_counts=True))},
           "mean_commit_pos_hall": (round(float(np.mean(cp[y == 1][cp[y == 1] >= 0])), 3)
                                    if (cp[y == 1] >= 0).any() else None),
           "mean_commit_pos_correct": (round(float(np.mean(cp[y == 0][cp[y == 0] >= 0])), 3)
                                      if (cp[y == 0] >= 0).any() else None),
           "n_cells": len(rows),
           "verdict": verdict,
           "verdict_basis": "fixed-slice pre-commit AUROC (edge-effect controlled)",
           "verdict_note": note,
           "gate_thresholds": {"precommit_auroc": PRECOMMIT_AUROC_GATE},
           "adaptive_precommit_auroc": None if np.isnan(adaptive_auc) else adaptive_auc,
           "fixed_slice_precommit_auroc": None if np.isnan(best_fixed_auc) else best_fixed_auc,
           "edge_effect_inflation": None if np.isnan(inflation) else round(float(inflation), 4),
           "best_precommit_cell": best, "best_window_cell": best_win,
           "best_fixed_slice_cell": best_fixed,
           "fixed_slice_cells": fixed_rows,
           "cells": rows}
    if verbose:
        print(f"\n=== {model_key} ===")
        print(f"items={len(flows)} hall={int(y.sum())} cells={len(rows)} verdict={verdict}")
        print(f"commit_pos hall={out['mean_commit_pos_hall']} correct={out['mean_commit_pos_correct']} "
              f"distribution={out['commit_position_distribution']}")
        if best:
            print(f"best adaptive cell:  L{best['layer']} w{best['window']} "
                  f"AUROC={best['precommit_auroc']}")
        if best_fixed:
            print(f"best fixed-slice:    L{best_fixed['layer']} cut={fixed_cut} "
                  f"AUROC={best_fixed['fixed_slice_auroc']}")
        print(f"edge-effect inflation={out['edge_effect_inflation']}")
        if note:
            print(f"NOTE: {note}")
    return out


def _top(rows, key, n=12):
    ok = [r for r in rows if r.get(key) is not None]
    ok.sort(key=lambda r: -r[key])
    return ok[:n]


def main():
    ap = argparse.ArgumentParser(description="NHE applicability gate (pre-commit search).")
    ap.add_argument("--model", default="gemma3-1b")
    ap.add_argument("--flows", default=None)
    ap.add_argument("--sweep-models", action="store_true")
    ap.add_argument("--full-cells", action="store_true",
                    help="store every scanned cell instead of the top ranked ones")
    ap.add_argument("--out", default=None,
                    help="report path; default results/applicability_gate_{model}.json")
    a = ap.parse_args()

    if a.sweep_models:
        results = [evaluate("gemma3-1b"), evaluate("qwen2.5-0.5b")]
    else:
        results = [evaluate(a.model, a.flows)]

    if not a.full_cells:
        # The full grid is a deterministic function of (flows, LAYERS, WINDOWS)
        # and costs ~3k lines per model. Keep the verdict, the ranked extremes and
        # the counts; regenerate the rest with --full-cells.
        for r in results:
            cells = r.pop("cells", [])
            fx = r.pop("fixed_slice_cells", [])
            r["top_precommit_cells"] = _top(cells, "precommit_auroc")
            r["top_window_cells"] = _top(cells, "window_auroc")
            r["top_fixed_slice_cells"] = _top(fx, "fixed_slice_auroc")
            r["n_cells_precommit_ge"] = {
                str(t): sum(1 for c in cells
                            if c.get("precommit_auroc") is not None
                            and c["precommit_auroc"] >= t)
                for t in (0.75, 0.70, 0.65, 0.60)
            }
            r["cells_omitted"] = len(cells) + len(fx)

    payload = {"cells_scanned": {"layers": LAYERS, "windows": WINDOWS},
               "note": ("top-N cells only; run with --full-cells for the complete grid"),
               "results": results}
    # One file per model: a shared file gets truncated whenever a single-model run
    # overwrites it, which silently breaks the downstream live stage.
    out = a.out
    if out is None:
        if a.sweep_models:
            out = os.path.join(RES, "applicability_gate_sweep.json")
        else:
            out = os.path.join(RES, f"applicability_gate_{results[0]['model']}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {out}")
    for r in results:
        line = f"  {r['model']}: {r['verdict']}"
        if r.get("best_fixed_slice_cell"):
            b = r["best_fixed_slice_cell"]
            line += (f" (fixed L{b['layer']} cut={b['fixed_cut']}"
                     f"={b['fixed_slice_auroc']})")
        print(line)


if __name__ == "__main__":
    main()
