"""One-command onboarding for a new model: collect -> gate -> decide -> report.

THE PROBLEM THIS SOLVES
    Onboarding a model to NHE was a manual sequence spread over five scripts with
    hidden coupling: `collect` writes flows, `fit_greedy` picks a layer using a
    commit-blind feature, `run` reads that layer, and nobody checks whether a
    pre-commit signal exists at all. That sequence silently produces confident,
    useless results on models where NHE cannot work (see the Qwen case: the
    family is present, the timing is not).

THE AUTOMATION
    This is a state machine with an explicit terminal condition per model. It
    never guesses: it refuses to report a model as usable unless the gate passes
    AND a live arm scores a positive net fix with zero new breaks.

        collect -> gate -> [INERT: stop, report] -> calibrate -> live run
               -> [net <= 0 or breaks > 0: report UNPROVEN] -> [net > 0, C2W=0: USABLE]

WHAT IS STILL NOT AUTOMATED, STATED PLAINLY
    Neuron attribution (`attribute_causal2*.py`) is hours of CPU patching and
    needs its own mask per architecture; it is invoked here but not optimized.
    Threshold percentile, cut and scale defaults are the deployed Gemma values and
    are re-calibrated per model from correct items only, never hand-tuned from a
    test set. Full autonomy would require an attribution-free intervention, which
    is not built.

Usage:
  python analysis/nhe_onboard.py --model gemma3-1b
  python analysis/nhe_onboard.py --model gemma3-1b --stage gate|live|report
Writes results/onboarding_{model}.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
REPO_ROOT = os.path.dirname(EDGE_ROOT)
RES = os.path.join(EDGE_ROOT, "results")
sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))

PY = sys.executable


def run(cmd, cwd=EDGE_ROOT, timeout=None):
    t0 = time.time()
    print(f"\n$ {' '.join(os.path.basename(c) if c == PY else c for c in cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    tail = "\n".join([ln for ln in p.stdout.splitlines() if ln.strip()][-12:])
    print(tail, flush=True)
    return p.returncode, tail, time.time() - t0


def flows_path(model):
    return os.path.join(RES, "greedy_flows_africa.npz" if model == "gemma3-1b"
                        else f"greedy_flows_africa_{model}.npz")


def stage_collect(model, skip):
    fp = flows_path(model)
    if os.path.exists(fp) and skip:
        print(f"[collect] exists, skipping ({os.path.basename(fp)})")
        return True, 0.0
    if model == "gemma3-1b":
        cmd = [PY, os.path.join(EDGE_ROOT, "core", "runtime_rollback.py"), "collect"]
    else:
        cmd = [PY, os.path.join(EDGE_ROOT, "core", "runtime_rollback_qwen.py"),
               "--model", model, "collect"]
    rc, _, dt = run(cmd, timeout=7200)
    return rc == 0 and os.path.exists(fp), dt


def gate_report_path(model):
    return os.path.join(RES, f"applicability_gate_{model}.json")


def stage_gate(model, skip):
    out = gate_report_path(model)
    if skip and os.path.exists(out):
        rep = json.load(open(out, encoding="utf-8"))
        if any(r["model"] == model for r in rep.get("results", [])):
            print("[gate] report exists, reusing")
            return True, 0.0
    rc, _, dt = run([PY, os.path.join(BASE, "gate.py"), "--model", model], timeout=3600)
    return rc == 0, dt


def read_gate(model):
    out = gate_report_path(model)
    if not os.path.exists(out):
        return None
    rep = json.load(open(out, encoding="utf-8"))
    return next((r for r in rep.get("results", []) if r["model"] == model), None)


def stage_live(model, mode, scale, skip):
    out = os.path.join(RES, f"gate_live_{model}.json")
    if skip and os.path.exists(out):
        print("[live] reusing cached result")
        return json.load(open(out, encoding="utf-8")), 0.0
    rc, tail, dt = run([PY, os.path.join(BASE, "gate_live_test.py"),
                        "--model", model, "--mode", mode,
                        "--scale", str(scale)], timeout=7200)
    return (json.load(open(out, encoding="utf-8")) if os.path.exists(out) else None), dt


def score_arm(model, cell, mode, scale):
    """Strict W2C/C2W for the arm the live run actually produced.

    `cell` is the gate winner dict: it carries the layer and the fixed cut, and
    the eval filename embeds both. Reading the layer from the same record that
    produced the run keeps the scorer from inventing a filename.
    """
    layer = cell.get("layer")
    cut = cell.get("cut", cell.get("fixed_cut"))
    if layer is None or cut is None:
        return None, "<no cell recorded>"
    tag = f"jump_gt_L{layer:02d}_p90_correct_{mode}"
    if cut != 5:
        tag += f"_w{cut}"
    if mode == "mask" and scale != 0.0:
        tag += f"_sft{scale}"
    arm = os.path.join(RES, f"eval_runtime_africa_{tag}.json")
    base_name = "eval_africa_baseline.json"
    if not os.path.exists(arm) or not os.path.exists(os.path.join(RES, base_name)):
        return None, os.path.basename(arm)
    import re
    import unicodedata
    sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))
    import topics
    alts = {t[0]: list(t[1:]) for t in topics.AFRICA}

    def strict(gen, ans, q):
        m = re.split(r"[.\n]", (gen or "").strip())
        f = unicodedata.normalize("NFKD", m[0]).encode("ascii", "ignore").decode().lower() if m and m[0].strip() else None
        cand = [ans] + list(alts.get(q, []))
        cand = [unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower() for x in cand]
        return 1 if (f and any(x in f for x in cand)) else 0

    b = json.load(open(os.path.join(RES, base_name), encoding="utf-8"))
    a = json.load(open(arm, encoding="utf-8"))
    bs = {r["id"]: strict(r["generated"], r["answer"], r["question"]) for r in b["results"]}
    as_ = {r["id"]: strict(r["generated"], r["answer"], r["question"]) for r in a["results"]}
    w2c = sum(1 for i, v in as_.items() if not bs[i] and v)
    c2w = sum(1 for i, v in as_.items() if bs[i] and not v)
    return {"arm": os.path.basename(arm), "w2c": w2c, "c2w": c2w,
            "net": w2c - c2w, "n": len(bs)}, os.path.basename(arm)


def main():
    ap = argparse.ArgumentParser(description="One-command NHE onboarding with a hard stop on INERT.")
    ap.add_argument("--model", default="gemma3-1b")
    ap.add_argument("--mode", default="mask", choices=["mask", "none", "abstain"])
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--stage", default="all",
                    choices=["all", "collect", "gate", "live", "report"])
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a cached live result instead of re-running (default: run)")
    ap.add_argument("--force-collect", action="store_true")
    ap.add_argument("--force-live", action="store_true")
    ap.add_argument("--allow-inert", action="store_true",
                    help="run the live arm even if the gate says INERT (research only)")
    a = ap.parse_args()

    t0 = time.time()
    rec = {"model": a.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "mode": a.mode, "scale": a.scale, "stages": []}

    if a.stage in ("all", "collect"):
        ok, dt = stage_collect(a.model, a.skip_existing and not a.force_collect)
        rec["stages"].append({"stage": "collect", "ok": bool(ok), "seconds": round(dt, 1)})
        if not ok:
            rec["verdict"] = "FAILED_COLLECT"
            rec["elapsed_seconds"] = round(time.time() - t0, 1)
            _save(rec, a.model)
            print("\nVERDICT: FAILED_COLLECT")
            return

    if a.stage in ("all", "gate", "live", "report"):
        ok, dt = stage_gate(a.model, a.skip_existing)
        rec["stages"].append({"stage": "gate", "ok": bool(ok), "seconds": round(dt, 1)})

    g = read_gate(a.model)
    rec["gate"] = None if g is None else {
        "verdict": g["verdict"],
        "fixed_slice_precommit_auroc": g.get("fixed_slice_precommit_auroc"),
        "adaptive_precommit_auroc": g.get("adaptive_precommit_auroc"),
        "edge_effect_inflation": g.get("edge_effect_inflation"),
        "cell": (g.get("best_fixed_slice_cell") or {}),
    }

    if a.stage == "report":
        _save(rec, a.model)
        print(json.dumps(rec.get("gate"), indent=1))
        return

    # Terminal condition 1: the gate says this model/format cannot be served.
    if g and g["verdict"] != "ACTIVE" and not a.allow_inert:
        rec["verdict"] = "INERT"
        rec["explanation"] = ("no (layer, window) cell reaches the fixed-slice pre-commit "
                              "gate; NHE-temporal has no usable trigger in this model/format")
        rec["elapsed_seconds"] = round(time.time() - t0, 1)
        _save(rec, a.model)
        print("\nVERDICT: INERT - NHE-temporal does not apply to "
              f"{a.model} in this format. Stop here; do not report it as a working cell.")
        return

    if a.stage in ("all", "live"):
        live, dt = stage_live(a.model, a.mode, a.scale,
                              a.skip_existing and not a.force_live)
        rec["stages"].append({"stage": "live", "ok": live is not None, "seconds": round(dt, 1)})
        if live:
            sc, arm_name = score_arm(a.model, dict(live.get("cell") or {}), a.mode, a.scale)
            rec["live"] = live
            rec["strict_score"] = sc
            rec["scored_arm"] = arm_name
            if sc is None:
                rec["verdict"] = "UNSCORED"
            elif sc["net"] > 0 and sc["c2w"] == 0:
                rec["verdict"] = "USABLE"
            else:
                rec["verdict"] = "UNPROVEN"
                rec["explanation"] = ("gate passed but the live arm did not produce a clean "
                                      "net fix; applicability is necessary, not sufficient")
        else:
            rec["verdict"] = "FAILED_LIVE"

    rec["elapsed_seconds"] = round(time.time() - t0, 1)
    _save(rec, a.model)
    print(f"\nVERDICT: {rec.get('verdict')}")
    if rec.get("explanation"):
        print(f"  {rec['explanation']}")
    print(f"saved results/onboarding_{a.model}.json")


def _save(rec, model):
    p = os.path.join(RES, f"onboarding_{model}.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    main()
