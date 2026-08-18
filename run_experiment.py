import os
import sys
import json
import subprocess

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE, "results")
PY = sys.executable

def make_masks():
    with open(os.path.join(RES_DIR, "attribution_africa.json"), "r", encoding="utf-8") as fh:
        att = json.load(fh)
    masks = {}
    for score in ["mean", "var"]:
        key = "top_by_mean" if score == "mean" else "top_by_var"
        for k in [32, 128, 512]:
            items = [{"layer": e["layer"], "unit": e["unit"]} for e in att[key][:k]]
            path = os.path.join(RES_DIR, f"mask_k{k}_{score}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"score": score, "k": k, "items": items}, fh)
            masks[(score, k)] = path
    with open(os.path.join(RES_DIR, "attribution_causal_africa.json"), "r", encoding="utf-8") as fh:
        att_c = json.load(fh)
    for k in [32, 128, 512]:
        items = [{"layer": e["layer"], "unit": e["unit"]} for e in att_c["top_by_causal"][:k]]
        path = os.path.join(RES_DIR, f"mask_k{k}_causal.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"score": "causal", "k": k, "items": items}, fh)
        masks[("causal", k)] = path
    with open(os.path.join(RES_DIR, "attribution_causal2_africa.json"), "r", encoding="utf-8") as fh:
        att_c2 = json.load(fh)
    for k in [32, 128, 512]:
        items = [{"layer": e["layer"], "unit": e["unit"]} for e in att_c2["top_wrong_only"][:k]]
        path = os.path.join(RES_DIR, f"mask_k{k}_wrong.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"score": "wrong", "k": k, "items": items}, fh)
        masks[("wrong", k)] = path
    for k in [64, 256]:
        items = [{"layer": e["layer"], "unit": e["unit"]} for e in att_c2["top_wrong_only"][:k]]
        path = os.path.join(RES_DIR, f"mask_k{k}_wrong.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"score": "wrong", "k": k, "items": items}, fh)
        masks[("wrong", k)] = path
    for k in [32, 128]:
        items = [{"layer": e["layer"], "unit": e["unit"]} for e in att_c2["top_mid_l8_17"][:k]]
        path = os.path.join(RES_DIR, f"mask_k{k}_mid.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"score": "mid", "k": k, "items": items}, fh)
        masks[("mid", k)] = path
    for k in [64, 256]:
        items = [{"layer": e["layer"], "unit": e["unit"]} for e in att_c2["top_mid_wrong_only"][:k]]
        path = os.path.join(RES_DIR, f"mask_k{k}_midwrong.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"score": "midwrong", "k": k, "items": items}, fh)
        masks[("midwrong", k)] = path
    for k in [32, 128]:
        items = [{"layer": e["layer"], "unit": e["unit"]} for e in att_c2["top_mid_wrong_only"][:k]]
        path = os.path.join(RES_DIR, f"mask_k{k}_midwrong.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"score": "midwrong", "k": k, "items": items}, fh)
        masks[("midwrong", k)] = path
    return masks

def run_eval(topic, mask_path=None):
    tag = "baseline"
    if mask_path is not None:
        with open(mask_path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        tag = f"k{len(m['items'])}_{m['score']}"
    out_file = os.path.join(RES_DIR, f"eval_{topic}_{tag}.json")
    if os.path.exists(out_file):
        print(f"[skip] {topic} {tag}", flush=True)
        return out_file
    cmd = [PY, os.path.join(BASE, "eval_topic.py"), topic]
    if mask_path:
        cmd.append(mask_path)
    print(f"[run ] {topic} {tag}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[FAIL] {topic} {tag}\n{r.stdout}\n{r.stderr}", flush=True)
        return None
    return out_file

def main():
    masks = make_masks()
    print("masks ready", flush=True)
    runs = []
    for topic in ["africa", "europe", "elements"]:
        runs.append((topic, None))
    for (score, k), path in masks.items():
        for topic in ["africa", "europe", "elements"]:
            runs.append((topic, path))

    rows = []
    for topic, mask_path in runs:
        out = run_eval(topic, mask_path)
        if out is None:
            continue
        with open(out, "r", encoding="utf-8") as fh:
            s = json.load(fh)
        base = None
        base_file = os.path.join(RES_DIR, f"eval_{topic}_baseline.json")
        if os.path.exists(base_file):
            with open(base_file, "r", encoding="utf-8") as fh:
                base = json.load(fh)
        changed = flips_w2c = flips_c2w = None
        if base is not None:
            by_q = {r["question"]: r for r in s["results"]}
            by_q_b = {r["question"]: r for r in base["results"]}
            changed = sum(1 for q, r in by_q.items() if q in by_q_b and r["generated"] != by_q_b[q]["generated"])
            flips_w2c = sum(1 for q, r in by_q.items() if q in by_q_b and not by_q_b[q]["correct"] and r["correct"])
            flips_c2w = sum(1 for q, r in by_q.items() if q in by_q_b and by_q_b[q]["correct"] and not r["correct"])
        rows.append({
            "topic": s["topic"], "mask": s["mask"] or "baseline",
            "n": s["n"], "correct": s["n_correct"],
            "hallucination_rate": s["hallucination_rate"],
            "correct_rate": s["correct_rate"],
            "changed": changed, "flips_wrong_to_correct": flips_w2c,
            "flips_correct_to_wrong": flips_c2w,
        })

    summary = {"seed": 0, "rows": rows}
    with open(os.path.join(RES_DIR, "summary_experiment.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"{'topic':<10}{'mask':<12}{'n':>4}{'hall':>7}{'corr':>7}{'chg':>5}{'w2c':>5}{'c2w':>5}")
    for r in rows:
        print(f"{r['topic']:<10}{r['mask']:<12}{r['n']:>4}{r['hallucination_rate']:>7.3f}{r['correct_rate']:>7.3f}{r['changed'] or 0:>5}{r['flips_wrong_to_correct'] or 0:>5}{r['flips_correct_to_wrong'] or 0:>5}")
    print("\nsaved results/summary_experiment.json")

if __name__ == "__main__":
    main()