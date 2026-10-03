"""Collect hidden-state jumps for the single-task benchmark.

WHY THIS EXISTS
    `reachability_of_quiet.py` must decide whether the never-fired wrong items on
    the 463-item benchmark are reachable by a better trigger, or are genuinely
    quiet (Static Memory Voids, as NTW predicts). That decision needs hidden
    states for all 463 items; the committed flow file covers only 54 Africa items.
    This collects them once so the decision is made from data.

WHAT IT STORES
    Per item, the per-layer L2 jump between consecutive generated-token hidden
    states, exactly as the runtime feature computes it (max over a window, at a
    chosen layer). Storing jumps rather than raw hidden states keeps the file
    small (463 items x 26 layers x ~40 floats) while remaining sufficient for any
    (layer, window) evaluation the reachability search needs.

    The first generated token of each item has no predecessor inside the
    generation, so its jump is recorded as 0.0, matching the runtime behaviour
    where the first step uses prev_score=None -> score 0.

Usage:
  python core/collect_single_task.py [--limit N]
Writes: results/greedy_flows_single_task.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
REPO_ROOT = os.path.dirname(EDGE_ROOT)
RES_DIR = os.path.join(EDGE_ROOT, "results")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(EDGE_ROOT, "analysis"))

import runtime_rollback as rr  # noqa: E402
import scoring  # noqa: E402

SAVE_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer")
BENCH = os.path.join(RES_DIR, "bench_single_task.json")
PROMPT_TMPL = "<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
MAX_NEW = rr.MAX_NEW
N_LAYERS = 26  # Gemma 3 1B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="flush partial results every N items")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=os.path.join(RES_DIR, "greedy_flows_single_task.json"))
    a = ap.parse_args()

    if not os.path.exists(BENCH):
        sys.exit("missing bench_single_task.json; run analysis/build_single_task.py first")
    bench = json.load(open(BENCH, encoding="utf-8"))
    items = bench["items"]
    if a.limit:
        items = items[:a.limit]

    out_items = []
    if a.resume and os.path.exists(a.out):
        try:
            prior = json.load(open(a.out, encoding="utf-8"))
            out_items = prior.get("items", [])
            print(f"resume: {len(out_items)} items already collected", flush=True)
        except (OSError, json.JSONDecodeError):
            print("resume: file unreadable, starting fresh", flush=True)
    done_ids = {r["id"] for r in out_items}
    pending = [it for it in items if it["id"] not in done_ids]
    if not pending:
        print(f"nothing to do: all {len(out_items)} items already present", flush=True)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16,
                                                 attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print(f"model loaded in {time.time()-t0:.1f}s, {len(items)} items", flush=True)

    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    t0 = time.time()
    for i, it in enumerate(pending):
        text = PROMPT_TMPL.format(q=it["question"])
        ids = tok(text, return_tensors="pt")["input_ids"]
        gen_ids = []
        past = None
        new_tok = ids
        prev_h = None  # per-item local, reset every iteration
        jumps = {str(l): [] for l in range(N_LAYERS)}
        with torch.no_grad():
            for _ in range(MAX_NEW):
                out = model(input_ids=new_tok, past_key_values=past, use_cache=True,
                            output_hidden_states=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
                if nxt == end_id:
                    break
                gen_ids.append(nxt)
                hs = out.hidden_states
                now = [hs[l + 1][0, -1].float() for l in range(N_LAYERS)]
                if prev_h is None:
                    for l in range(N_LAYERS):
                        jumps[str(l)].append(0.0)
                else:
                    for l in range(N_LAYERS):
                        jumps[str(l)].append(
                            float(torch.linalg.norm(now[l] - prev_h[l])))
                prev_h = now
                new_tok = torch.tensor([[nxt]])
        gen = tok.decode(gen_ids, skip_special_tokens=True)
        out_items.append({
            "id": it["id"], "question": it["question"], "generated": gen,
            "correct": int(scoring.strict(gen, it["answers"])),
            "jumps": jumps,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pending)} ({time.time()-t0:.0f}s)", flush=True)
        if a.checkpoint_every and (i + 1) % a.checkpoint_every == 0:
            with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"bench": "bench_single_task.json", "n": len(out_items),
                           "n_layers": N_LAYERS, "complete": False,
                           "items": out_items}, fh, ensure_ascii=False)
            print(f"  checkpoint: {len(out_items)} items total", flush=True)

    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"bench": "bench_single_task.json", "n": len(out_items),
                   "n_layers": N_LAYERS, "complete": True,
                   "items": out_items}, fh, ensure_ascii=False)
    wrong = sum(1 for r in out_items if not r["correct"])
    print(f"saved {a.out} n={len(out_items)} wrong={wrong} "
          f"({time.time()-t0:.0f}s)")
    if len(out_items) != len(bench["items"]) and not a.limit:
        print(f"WARNING: collected {len(out_items)} of {len(bench['items'])} bench "
              f"items. Rerun with --resume to finish the remainder.")


if __name__ == "__main__":
    main()
