"""Gate-driven live test: run the cell the applicability gate actually chose.

WHY THIS EXISTS
    `runtime_rollback.py run` always loads the layer/threshold that
    `fit_greedy` picked using a commit-blind feature (max jump over the first 10
    tokens). The gate in `gate.py` ranks cells by a commit-AWARE feature instead,
    and on Gemma the two disagree (fit chose L19; the gate prefers L20 on a fixed
    6-token slice). This script closes the loop: it reads the gate verdict, builds
    a detector from the winning cell, calibrates its threshold on the correct
    items only, and runs the live intervention with it.

    It also enforces the gate: if the gate says INERT, running is refused, so an
    inert model/format can never be reported as a working NHE cell.

USAGE
    python analysis/gate_live_test.py --model gemma3-1b --mode mask --scale 0.3
    python analysis/gate_live_test.py --model gemma3-1b --allow-inert   # for research
Writes results/gate_live_{model}.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
RES = os.path.join(EDGE_ROOT, "results")
sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))

import gate  # noqa: E402


def calibrate(model_key, layer, cut, pct=90):
    """Threshold on the winning feature, calibrated on CORRECT items only."""
    case = gate.load_case(model_key)
    if case is None:
        sys.exit(f"no flows for {model_key}")
    flows, texts, labels, _ = case
    y = 1 - labels
    feats = []
    for f in flows:
        h = f[:, layer + 1, :].astype(np.float32)
        j = np.linalg.norm(h[1:] - h[:-1], axis=1)
        seg = j[:cut] if cut else j
        feats.append(float(seg.max()) if len(seg) else 0.0)
    feats = np.array(feats)
    truth = y == 0
    thr = float(np.percentile(feats[truth], pct))
    catches = int((feats[y == 1] > thr).sum())
    return thr, catches, int(y.sum()), feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-1b")
    ap.add_argument("--mode", default="mask", choices=["mask", "none", "abstain"])
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--topic", default="africa")
    ap.add_argument("--pct", type=int, default=90)
    ap.add_argument("--allow-inert", action="store_true")
    ap.add_argument("--force-layer", type=int, default=None)
    ap.add_argument("--force-cut", type=int, default=None)
    a = ap.parse_args()

    # Recompute for this model unless a report already contains it. One report
    # file per model (gate.py) so a single-model run can never leave a partial
    # shared file that the live stage would read as "no result".
    report_path = os.path.join(RES, f"applicability_gate_{a.model}.json")
    entry = None
    if os.path.exists(report_path):
        report = json.load(open(report_path, encoding="utf-8"))
        entry = next((r for r in report.get("results", []) if r["model"] == a.model), None)
    if entry is None or not entry.get("best_fixed_slice_cell"):
        print(f"[gate] no usable cached result for {a.model}; computing it now")
        entry = gate.evaluate(a.model, a.flows)
    if not entry.get("fixed_slice_cells") and not entry.get("best_fixed_slice_cell"):
        sys.exit(f"gate produced no cells for {a.model}")

    if entry["verdict"] != "ACTIVE" and not a.allow_inert:
        sys.exit(f"GATE REFUSES {a.model}: verdict={entry['verdict']} "
                 f"(fixed-slice AUROC={entry.get('fixed_slice_precommit_auroc')}). "
                 f"Pass --allow-inert only to study an inert cell.")

    cell = entry.get("best_fixed_slice_cell") or {}
    layer = a.force_layer if a.force_layer is not None else cell.get("layer")
    cut = a.force_cut if a.force_cut is not None else cell.get("fixed_cut")
    if layer is None or cut is None:
        sys.exit("cannot determine winning cell")

    thr, catches, n_hall, feats = calibrate(a.model, layer, cut, a.pct)
    print(f"[gate] {a.model} verdict={entry['verdict']} -> L{layer} cut={cut} "
          f"thr(p{a.pct}, correct-only)={thr:.4f} catches {catches}/{n_hall} hallucinations")

    import runtime_rollback_qwen as qq
    det = {"type": "jump_gt", "layer": int(layer), "threshold": thr,
           "threshold_key": f"p{a.pct}_correct", "mode": a.mode,
           "window": int(cut), "scale": a.scale if a.mode == "mask" else 0.0,
           "sample": False, "seed": 0}
    if a.model.startswith("qwen"):
        import runtime_rollback as rr
        rr.RES_DIR = qq.RES_DIR
        rr.MASK = os.path.join(qq.RES_DIR, "mask_k32_midwrong_qwen2.5-0.5b.json")
        qq.run_topic(a.topic, det, model_key=a.model)
    else:
        import runtime_rollback as rr
        rr.run_topic(a.topic, det)

    tag = f"jump_gt_L{layer:02d}_p{a.pct}_correct_{a.mode}"
    if cut != 5:
        tag += f"_w{cut}"
    if a.mode == "mask" and a.scale != 0.0:
        tag += f"_sft{a.scale}"
    out_file = os.path.join(RES, f"eval_runtime_{a.topic}_{tag}.json")

    run = json.load(open(out_file, encoding="utf-8"))
    n_fired = sum(1 for r in run["results"] if r["fired_at"] is not None)
    payload = {"model": a.model, "gate_verdict": entry["verdict"],
               "cell": {"layer": layer, "cut": cut}, "threshold": thr,
               "mode": a.mode, "scale": a.scale, "topic": a.topic,
               "n": run["n"], "n_fired": n_fired,
               "hallucination_rate": run["hallucination_rate"],
               "offline_catches": f"{catches}/{n_hall}",
               "edge_effect_inflation": entry.get("edge_effect_inflation"),
               "eval_file": os.path.basename(out_file)}
    with open(os.path.join(RES, f"gate_live_{a.model}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[gate] live: fired={n_fired}/{run['n']} hall={run['hallucination_rate']:.4f}")
    print(f"saved results/gate_live_{a.model}.json")


if __name__ == "__main__":
    main()
