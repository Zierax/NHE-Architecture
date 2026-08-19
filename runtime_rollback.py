import os
import sys
import time
import json
import glob
import pickle
import numpy as np
import torch
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(BASE, "models", "gemma3-1b-tokenizer")
RES_DIR = os.path.join(BASE, "results")
DATA_DIR = os.path.join(BASE, "data")

MAX_NEW = 48
N_LAYERS = 26

TOPICS = {"africa": "topics.AFRICA", "europe": "topics.EUROPE", "elements": "topics.ELEMENTS",
          "asia": "topics.ASIA", "us_states": "topics.US_STATES",
          "africa_largest": "topics.AFRICA_LARGEST", "world_tricky": "topics.WORLD_TRICKY"}
MASK = os.path.join(RES_DIR, "mask_k32_midwrong.json")

def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

def model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
    return model, tok

def apply_mask(model, mask_path, scale=0.0):
    with open(mask_path, "r", encoding="utf-8") as fh:
        mask = json.load(fh)
    layers = model.model.layers
    with torch.no_grad():
        for e in mask["items"]:
            mlp = layers[e["layer"]].mlp
            mlp.down_proj.weight[:, e["unit"]] *= scale
            mlp.up_proj.weight[e["unit"], :] *= scale
            mlp.gate_proj.weight[e["unit"], :] *= scale
    return len(mask["items"])

def detector_fire(det, score, prev_score):
    t = det["threshold"]
    if det["type"] == "jump_gt":
        return prev_score is not None and score > t
    if det["type"] == "cos_lt":
        return prev_score is not None and score < t
    if det["type"] == "probe_lt":
        return score < t
    if det["type"] == "depth_gt":
        return score > t
    return False

def gen_with_detector(model, tok, ids, det):
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    n_masked = 0
    fired_at = None
    feats = []
    prev_h = None
    prev_score = None
    past = None
    out_ids = []
    ids = ids.clone()
    new_tok = ids
    rng = torch.Generator().manual_seed(det.get("seed", 0)) if det.get("sample") else None
    for t in range(MAX_NEW):
        out = model(input_ids=new_tok, past_key_values=past, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        hs = out.hidden_states
        if det["type"] == "probe_lt":
            h = hs[det["layer"] + 1][0, -1]
            score = (h.float() @ det["coef"] + det["intercept"]).item()
        elif det["type"] in ("jump_gt", "cos_lt"):
            h = hs[det["layer"] + 1][0, -1].float()
            if prev_h is not None:
                h0 = hs[0][0, -1].float() if False else prev_h
                d = h - prev_h
                if det["type"] == "jump_gt":
                    score = torch.linalg.norm(d).item()
                else:
                    score = (h @ prev_h / (torch.linalg.norm(h) * torch.linalg.norm(prev_h) + 1e-9)).item()
            else:
                score = 0.0
            prev_h = h
        else:
            hl = hs[det["layer"] + 1][0, -1].float()
            hlm = hs[det["layer"]][0, -1].float()
            score = torch.linalg.norm(hl - hlm).item()
        feats.append(score)
        if fired_at is None and detector_fire(det, score, prev_score) and (not det.get("window") or t <= det["window"]):
            fired_at = t
            if det.get("mode") == "rollback" and len(out_ids) >= 2:
                ids = ids[:, :-2]
                out_ids = out_ids[:-2]
                prev_h = None
                prev_score = None
                past = None
                out = model(input_ids=ids, past_key_values=None, use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                hs = out.hidden_states
                if det["type"] == "probe_lt":
                    score = (hs[det["layer"] + 1][0, -1].float() @ det["coef"] + det["intercept"]).item()
                elif det["type"] in ("jump_gt", "cos_lt"):
                    prev_h = hs[det["layer"] + 1][0, -1].float()
                    score = 0.0
                else:
                    hl = hs[det["layer"] + 1][0, -1].float()
                    hlm = hs[det["layer"]][0, -1].float()
                    score = torch.linalg.norm(hl - hlm).item()
                feats.append(score)
            if det.get("mode") != "none":
                n_masked = apply_mask(model, MASK, det.get("scale", 0.0))
        nxt = out.logits[0, -1].argmax().item() if not rng else _sample(out.logits[0, -1], rng, det)
        if nxt == end_id:
            break
        new_tok = torch.tensor([[nxt]])
        ids = torch.cat([ids, new_tok], dim=1)
        out_ids.append(nxt)
        prev_score = score
    return out_ids, feats, fired_at, n_masked

def _sample(logits, rng, det):
    t = det.get("temperature", 0.9)
    p = det.get("top_p", 0.9)
    probs = torch.softmax(logits / t, dim=-1)
    sorted_p, _ = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=0)
    cut = cum <= p
    k = int(cut.sum().item())
    if k < 1:
        k = 1
    top_v, top_i = torch.topk(probs, k)
    return top_i[torch.multinomial(top_v / top_v.sum(), 1, generator=rng)].item()

def run_topic(topic, det):
    mod = __import__("topics")
    items = getattr(mod, TOPICS[topic].split(".")[1])
    model, tok = model_and_tok()
    clean = {k: v.clone() for k, v in model.state_dict().items()}
    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        model.load_state_dict(clean)
        det_i = dict(det)
        if det_i.get("sample"):
            det_i["seed"] = det_i.get("seed", 1000) + i * 100
        q, ans = item[0], item[1]
        alt = item[2:]
        text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            gen_ids, feats, fired_at, n_masked = gen_with_detector(model, tok, ids, det_i)
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        gen_n = norm(gen_text)
        correct = 1 if any(norm(a) in gen_n for a in (ans,) + alt) else 0
        results.append({"id": i, "question": q, "answer": ans, "generated": gen_text,
                        "correct": correct, "fired_at": fired_at, "n_masked": n_masked,
                        "max_feature": float(max(feats)) if feats else None})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_fired = sum(1 for r in results if r["fired_at"] is not None)
    tag = f"{det['type']}_L{det['layer']:02d}_{det['threshold_key']}_{det.get('mode', 'mask')}"
    if det.get("window") and det.get("window") != 5:
        tag += f"_w{det['window']}"
    if det.get("scale", 0.0) != 0.0:
        tag += f"_sft{det['scale']}"
    if det.get("sample"):
        tag += f"_s{det.get('seed', 0)}"
    out_file = os.path.join(RES_DIR, f"eval_runtime_{topic}_{tag}.json")
    summary = {"topic": topic, "detector": det, "n": n, "n_correct": n_correct, "n_fired": n_fired,
               "hallucination_rate": round((n - n_correct) / n, 4), "results": results}
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[{topic} {tag}] fired={n_fired}/{n} hall={(n-n_correct)/n:.3f} -> {out_file}", flush=True)

def collect_greedy_flows():
    model, tok = model_and_tok()
    mod = __import__("topics")
    items = mod.AFRICA
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    flows, texts, labels = [], [], []
    t0 = time.time()
    for i, item in enumerate(items):
        q, ans = item[0], item[1]
        alt = item[2:]
        text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
        ids = tok(text, return_tensors="pt")["input_ids"]
        hs_all, out_ids = [], []
        past = None
        new_tok = ids
        with torch.no_grad():
            for t in range(MAX_NEW):
                out = model(input_ids=new_tok, past_key_values=past, use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                hs_all.append([h[0, -1].float().numpy() for h in out.hidden_states])
                nxt = out.logits[0, -1].argmax().item()
                if nxt == end_id:
                    break
                new_tok = torch.tensor([[nxt]])
                out_ids.append(nxt)
        flow = np.stack(hs_all).astype(np.float16)
        gen_text = tok.decode(out_ids, skip_special_tokens=True)
        gen_n = norm(gen_text)
        correct = 1 if any(norm(a) in gen_n for a in (ans,) + alt) else 0
        flows.append(flow)
        texts.append(gen_text)
        labels.append(correct)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(os.path.join(RES_DIR, "greedy_flows_africa.npz"),
             flows=np.array(flows, dtype=object), texts=np.array(texts, dtype=object),
             labels=np.array(labels))
    n_hall = sum(1 for l in labels if not l)
    print(f"saved results/greedy_flows_africa.npz n={len(labels)} hall={n_hall}", flush=True)

def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)

def fit_greedy():
    d = np.load(os.path.join(RES_DIR, "greedy_flows_africa.npz"), allow_pickle=True)
    flows = list(d["flows"])
    labels = np.array(d["labels"]).astype(int)
    y = 1 - labels
    n = len(flows)
    hall = y == 1
    truth = y == 0
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    print(f"greedy flows: n={n} hall={int(hall.sum())} truth={int(truth.sum())}", flush=True)

    T = np.array([f.shape[0] for f in flows])
    x_last = np.stack([f[-1] for f in flows]).astype(np.float32)

    auc_l10 = float("nan")
    preds = np.zeros(n)
    loo = LeaveOneOut()
    for tr, te in loo.split(x_last):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(x_last[:, 11, :][tr], y[tr])
        preds[te] = clf.predict_proba(x_last[te][:, 11, :])[:, 1]
    auc_l10 = _auc(y, preds)
    print(f"probe L10 (LOSO) AUC(hall)={auc_l10:.3f}", flush=True)

    feat_jumps = np.zeros((n, N_LAYERS))
    feat_cos = np.zeros((n, N_LAYERS))
    feat_jumps_early = np.zeros((n, N_LAYERS))
    feat_cos_early = np.zeros((n, N_LAYERS))
    for i, f in enumerate(flows):
        for l in range(N_LAYERS):
            h = f[:, l + 1, :].astype(np.float32)
            if len(h) >= 2:
                d = h[1:] - h[:-1]
                jumps = np.linalg.norm(d, axis=1)
                num = np.sum(h[1:] * h[:-1], axis=1)
                den = np.linalg.norm(h[1:], axis=1) * np.linalg.norm(h[:-1], axis=1) + 1e-9
                cos = num / den
                feat_jumps[i, l] = jumps.max()
                feat_cos[i, l] = cos.min()
                e = min(10, len(jumps))
                feat_jumps_early[i, l] = jumps[:e].max()
                feat_cos_early[i, l] = cos[:e].min()
            else:
                feat_jumps[i, l] = 0.0
                feat_cos[i, l] = 0.0
                feat_jumps_early[i, l] = 0.0
                feat_cos_early[i, l] = 0.0
    best = []
    for l in range(N_LAYERS):
        best.append((f"jump_max_L{l:02d}", _auc(y, feat_jumps[:, l]), l, "jump_gt"))
        best.append((f"cos_min_L{l:02d}", _auc(y, feat_cos[:, l]), l, "cos_lt"))
    best_early = []
    for l in range(N_LAYERS):
        best_early.append((f"jump_max_early_L{l:02d}", _auc(y, feat_jumps_early[:, l]), l, "jump_gt", feat_jumps_early))
        best_early.append((f"cos_min_early_L{l:02d}", _auc(y, feat_cos_early[:, l]), l, "cos_lt", feat_cos_early))
    best_early.sort(key=lambda e: -e[1])
    best.sort(key=lambda e: -e[1])
    print("top time-jitter features (greedy, scalar AUC):")
    for name, a, l, typ in best[:6]:
        print(f"  {name}: {a:.3f}")
    print("top EARLY (first-10-tokens) features:")
    for name, a, l, typ, _ in best_early[:6]:
        print(f"  {name}: {a:.3f}")
    jname, jauc, jl, jtyp = best[0]
    ename, eauc, el, etyp, earr = best_early[0]
    if eauc >= max(auc_l10, jauc):
        sel = {"type": etyp, "layer": el, "auc": eauc, "arr": earr}
        sel_name = ename
        sel["early"] = True
    else:
        sel = {"type": jtyp, "layer": jl, "auc": jauc, "arr": feat_jumps if jtyp == "jump_gt" else feat_cos}
        sel_name = jname
        sel["early"] = False
    print(f"selected: {sel_name} AUC={sel['auc']:.3f} early={sel['early']}", flush=True)
    arr = sel.pop("arr")
    vals_truth = arr[truth, sel["layer"]]
    if sel["type"] in ("probe_lt", "cos_lt"):
        thr90, thr95 = np.percentile(vals_truth, 10), np.percentile(vals_truth, 5)
        s_hall = arr[hall, sel["layer"]]
        n_hall_90 = int((s_hall < thr90).sum())
        n_hall_95 = int((s_hall < thr95).sum())
    else:
        thr90, thr95 = np.percentile(vals_truth, 90), np.percentile(vals_truth, 95)
        s_hall = arr[hall, sel["layer"]]
        n_hall_90 = int((s_hall > thr90).sum())
        n_hall_95 = int((s_hall > thr95).sum())
    print(f"  t90={thr90:.4f} catches {n_hall_90}/{int(hall.sum())} | t95={thr95:.4f} catches {n_hall_95}/{int(hall.sum())} (train calibration)", flush=True)
    sel["threshold_t90"] = float(thr90)
    sel["threshold_t95"] = float(thr95)
    sel["catch_t90"] = n_hall_90
    sel["catch_t95"] = n_hall_95
    sel["trained_on"] = "greedy_africa"
    sel["name"] = sel_name

    ear = {"type": etyp, "layer": el, "auc": eauc, "early": True, "name": ename}
    ear_arr = earr
    vals_truth_e = ear_arr[truth, el]
    if etyp == "jump_gt":
        e90, e95 = np.percentile(vals_truth_e, 90), np.percentile(vals_truth_e, 95)
        s_hall_e = ear_arr[hall, el]
        n_e90, n_e95 = int((s_hall_e > e90).sum()), int((s_hall_e > e95).sum())
    else:
        e90, e95 = np.percentile(vals_truth_e, 10), np.percentile(vals_truth_e, 5)
        s_hall_e = ear_arr[hall, el]
        n_e90, n_e95 = int((s_hall_e < e90).sum()), int((s_hall_e < e95).sum())
    ear["threshold_t90"] = float(e90)
    ear["threshold_t95"] = float(e95)
    ear["catch_t90"] = n_e90
    ear["catch_t95"] = n_e95
    ear["trained_on"] = "greedy_africa"
    print(f"  early {ename}: t90={e90:.4f} catches {n_e90}/{int(hall.sum())} | t95={e95:.4f} catches {n_e95}/{int(hall.sum())}", flush=True)

    with open(os.path.join(RES_DIR, "detector_greedy.json"), "w") as fh:
        json.dump({"selected": sel_name, "detector": sel, "detector_early": ear,
                   "probe_L10_auc": float(auc_l10),
                   "top_features": [{"name": n, "auc": float(a)} for n, a, _, _ in best[:10]],
                   "top_early_features": [{"name": n, "auc": float(a)} for n, a, _, _, _ in best_early[:10]]},
                  fh, indent=1)
    print("saved results/detector_greedy.json", flush=True)

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "collect"
    if cmd == "collect":
        collect_greedy_flows()
        return
    if cmd == "fit_greedy":
        fit_greedy()
        return
    if cmd == "run":
        topic = sys.argv[2]
        with open(os.path.join(RES_DIR, "detector_greedy.json")) as fh:
            dg = json.load(fh)
        sel = "detector_early" if (len(sys.argv) > 3 and sys.argv[3] == "early") else "detector"
        thr_key = sys.argv[4] if len(sys.argv) > 4 else "t90"
        det = dict(dg[sel])
        det["threshold"] = det["threshold_t90"] if thr_key == "t90" else det["threshold_t95"]
        det["threshold_key"] = thr_key
        det["mode"] = sys.argv[5] if len(sys.argv) > 5 else "mask"
        det["window"] = int(sys.argv[9]) if len(sys.argv) > 9 else (5 if sel == "detector_early" else None)
        det["scale"] = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
        det["sample"] = "s" in (sys.argv[6] if len(sys.argv) > 6 else "m")
        det["seed"] = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
        print(f"running {topic} with {dg[sel]['name']} {thr_key} mode={det['mode']} sample={det['sample']} seed={det['seed']} scale={det['scale']} window={det['window']}", flush=True)
        run_topic(topic, det)
        return
    sys.exit("usage: runtime_rollback.py <collect|fit_greedy|run topic t90|t95 [mask|rollback|none]>")

if __name__ == "__main__":
    main()