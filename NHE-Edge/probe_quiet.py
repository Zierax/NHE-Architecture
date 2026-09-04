"""
Quiet-commit diagnostic for NHE-Architecture — Gemma 3 1B.
Tests hypothesis: 4 quiet hallucinations are parametric (training-data) errors
with high confidence, low jitter, persistent across paraphrases, vs 3 dynamic
hallucinations with low early confidence, high jitter, layer flips.

Implements logit lens at answer-token position(s) and paraphrase persistence test.
"""
import os
import sys
import json
import time
import re
import unicodedata
from pathlib import Path

import torch
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent if BASE.name == "NHE-Edge" else BASE
SAVE_DIR = REPO_ROOT / "models" / "gemma3-1b-fp16"
TOK_DIR = REPO_ROOT / "models" / "gemma3-1b-tokenizer"
RES_DIR = BASE / "results"

# ---------------------------------------------------------------------------
# Task definition: 4 quiet (no jitter spike) vs 3 dynamic (jitter fires)
# Verified from results/eval_africa_baseline.json + detector_greedy.json:
# quiet = detector never fires (fired_at None) across seeds s1000-s1004
# dynamic = detector fires at t=3-5.
# ---------------------------------------------------------------------------
QUIET_IDS = [7, 15, 19, 22]      # Cape Verde, Equatorial Guinea, Gabon, Guinea
DYNAMIC_IDS = [17, 20, 41]       # Eswatini, Gambia, Senegal
ALL_IDS = QUIET_IDS + DYNAMIC_IDS

# For human labeling
ID_TO_COUNTRY = {
    7: "Cape Verde",
    15: "Equatorial Guinea",
    19: "Gabon",
    22: "Guinea",
    17: "Eswatini",
    20: "Gambia",
    41: "Senegal",
}

# Paraphrase templates — three per question, kept deterministic
PARAPHRASE_TEMPLATES = [
    "What is the capital city of {country}?",
    "Name the capital of {country}.",
    "Which city is the capital of {country}?",
]

MAX_NEW = 24  # enough to capture answer + markdown


def norm_txt(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def extract_wrong_answer(generated: str) -> str:
    """Extract the hallucinated capital name from generated text.
    Heuristic: content between ** **, else after 'is'.
    Returns raw string without markdown.
    """
    # try markdown bold capture
    m = re.search(r"\*\*(.*?)\*\*", generated, flags=re.DOTALL)
    if m:
        cand = m.group(1).strip()
        # strip trailing punctuation
        cand = cand.strip(" .,\n\r*")
        if cand:
            return cand
    # fallback: after 'is' up to period/newline
    mm = re.search(r"is\s+([^.\n]+)", generated, flags=re.IGNORECASE)
    if mm:
        cand = mm.group(1).strip().strip("* ").strip()
        # take first 3 words max (capital names rarely longer)
        cand = " ".join(cand.split()[:4])
        cand = cand.strip(" .,\n\r*")
        if cand:
            return cand
    # ultimate fallback: first bold-like token
    return generated.strip().split()[-1].strip("*.,")


def country_from_question(q: str) -> str:
    """Extract country name from 'What is the capital of X?'"""
    m = re.search(r"capital of (.+?)\?", q)
    if m:
        c = m.group(1).strip()
        # handle "the Central African Republic" etc
        if c.lower().startswith("the "):
            c = c[4:]
        return c
    return q


def model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    # fp16 on CPU can segfault (seen EXIT 0xC0000005); use float32 on CPU, fp16 only if CUDA available
    use_fp16 = torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32
    tok = AutoTokenizer.from_pretrained(str(TOK_DIR))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(SAVE_DIR),
            dtype=dtype,
            attn_implementation="eager",
        )
    except Exception as e:
        print(f"load with dtype={dtype} failed: {e}, retrying float32", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(SAVE_DIR),
            dtype=torch.float32,
            attn_implementation="eager",
        )
    model.eval()
    # Force CPU if no CUDA to avoid implicit GPU expectations
    try:
        model = model.to("cpu")
    except Exception:
        pass
    # Ensure CPU; if available but no CUDA, leave on CPU
    # Hidden states require output_hidden_states flag per forward call, not via config
    print(f"model loaded in {time.time()-t0:.1f}s | layers={model.config.num_hidden_layers} hidden={model.config.hidden_size} vocab={model.config.vocab_size}", flush=True)
    # defensive: verify norm and head exist for logit lens
    assert hasattr(model, "model"), "Gemma3ForCausalLM should have .model"
    assert hasattr(model.model, "norm"), "Gemma3TextModel should have final .norm"
    assert hasattr(model, "lm_head"), "Missing lm_head"
    return model, tok


def get_first_token_id(ans_str: str, tok, debug_prefix: str = "") -> tuple[int, list[int], str]:
    """
    Tokenize answer string and return first token id.
    Handles Gemma's SentencePiece: we try multiple prefix contexts and prefer
    the variant that matches generation context (" **" + city).
    Strategy:
      - Tokenize variants: ans_str, " "+ans_str, "**"+ans_str, " **"+ans_str
      - For probing context "The capital ... is **" -> next token is city without leading space.
        So the relevant id is the first token of ans_str WITHOUT leading space when tokenized alone,
        but we also test " **" prefix variant to disambiguate.
      Returns (first_id, all_ids_for_primary_variant, token_str)
    """
    # Primary: tokenize without leading space (matches after "**" where "**" consumed space)
    ids_plain = tok.encode(ans_str, add_special_tokens=False)
    ids_space = tok.encode(" " + ans_str, add_special_tokens=False)
    ids_star = tok.encode(" **" + ans_str, add_special_tokens=False)
    # ids_star includes token for " **" at start; remaining are city tokens
    # Determine city token after "**" prefix: encode prefix alone and subtract
    prefix_ids = tok.encode(" **", add_special_tokens=False)
    # Find where prefix ends in ids_star
    # Since encoding is not strictly compositional, use heuristic: first token(s) match prefix_ids
    # Safer: just use plain ids' first token as canonical per task spec "first token"
    if not ids_plain:
        raise ValueError(f"empty tokenization for answer '{ans_str}' ({debug_prefix})")
    first_id = ids_plain[0]
    token_str = tok.decode([first_id])
    # Also compute alternative ids for logging
    return first_id, ids_plain, token_str


def get_multi_token_ids(ans_str: str, tok) -> list[int]:
    """Return all token ids for ans_str without add_special_tokens, plain."""
    return tok.encode(ans_str, add_special_tokens=False)


def logit_lens_per_layer(hidden_states, model):
    """
    Apply logit lens to each layer hidden state at a single token position.
    hidden_states: tuple/list from model (len = num_layers+1), each [1, seq, hidden]
                   hidden_states[0] = embeddings, [1..L] after block.
    Returns list of dict per layer with logits, probs, top token etc (but not yet jitter).
    Must apply final norm (model.model.norm) before lm_head per task spec.
    IMPORTANT: For Gemma3, hidden_states[-1] is already after final norm (see
    Gemma3TextModel.forward: hidden_states = self.norm(hidden_states) as last_hidden_state
    and hidden_states tuple includes that normed state). Applying norm again double-norms
    and mismatches final logits. We handle last layer without re-applying norm.
    """
    fin_norm = model.model.norm
    head = model.lm_head
    per_layer = []
    n = len(hidden_states)
    for li, h in enumerate(hidden_states):
        vec = h[0, -1, :]  # last position; shape [hidden]
        # hidden_states[-1] is already normed; don't double-norm
        is_last = (li == n - 1)
        # Gemma's RMSNorm is (x * rsqrt(mean(x^2)+eps) * (1+weight))
        # For already-normed last layer, skip; for others apply final norm as lens approximation
        if is_last:
            # Already normed: directly project
            logits = head(vec)
        else:
            vec_normed = fin_norm(vec)
            logits = head(vec_normed)
        logits_f32 = logits.float()
        probs = torch.softmax(logits_f32, dim=-1)
        per_layer.append({
            "layer": li,
            "hidden_norm": float(torch.linalg.norm(vec.float()).item()),
            "logits": logits_f32,
            "probs": probs,
            "norm_applied": not is_last,
        })
    return per_layer


def compute_entropy(probs: torch.Tensor) -> float:
    """Compute entropy in nats, probs is 1D tensor normalized."""
    # probs already from softmax; add epsilon for log
    p = probs.clamp(min=1e-12)
    ent = -(p * torch.log(p)).sum().item()
    return float(ent)


def compute_topk(probs: torch.Tensor, k: int = 10):
    """Return topk (prob, id) sorted descending."""
    vals, ids = torch.topk(probs, k)
    return [(float(v), int(i)) for v, i in zip(vals, ids)]


def compute_depth_jitter(hidden_states):
    """Compute per-layer depth jitter = ||h_l - h_{l-1}|| at answer position.
    hidden_states list len = L+1
    Returns dict layer->jitter (for layer starting at 1..L). For 0 jitter = 0.
    """
    jitters = []
    for li in range(len(hidden_states)):
        if li == 0:
            jitters.append(0.0)
        else:
            h_cur = hidden_states[li][0, -1, :].float()
            h_prev = hidden_states[li-1][0, -1, :].float()
            jd = float(torch.linalg.norm(h_cur - h_prev).item())
            jitters.append(jd)
    return jitters  # index aligns with per_layer


def run_single_probe(model, tok, prompt_text: str):
    """Run single forward pass at prompt_text, returns hidden_states and logits."""
    ids = tok(prompt_text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states  # tuple length 27
    logits = out.logits  # [1, seq, vocab] final layer only
    return ids, hidden_states, logits, out


def greedy_generate_with_hiddens(model, tok, prompt_text: str, max_new: int = MAX_NEW):
    """Greedy generation loop capturing hidden_states per step, similar to collect_greedy_flows.
    Returns (generated_ids list, gen_text, flows per step, tokenwise hiddens)
    Each step's hidden_states is tuple len 27.
    Also returns per-step logits for diagnostic.
    """
    ids = tok(prompt_text, return_tensors="pt")["input_ids"]
    past = None
    new_tok = ids
    all_hiddens = []  # list per step of hidden_states tuple
    step_logits = []
    out_ids = []
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    if end_id is None or end_id < 0:
        end_id = tok.eos_token_id
    with torch.no_grad():
        for t in range(max_new):
            out = model(input_ids=new_tok, past_key_values=past, use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            hidden_states = out.hidden_states
            all_hiddens.append(hidden_states)  # per step
            step_logits.append(out.logits[0, -1, :].float())
            nxt = int(out.logits[0, -1, :].argmax().item())
            if nxt == end_id:
                break
            new_tok = torch.tensor([[nxt]], dtype=torch.long)
            out_ids.append(nxt)
            # safety break if generated too long or model loops
            if len(out_ids) >= max_new:
                break
    gen_text = tok.decode(out_ids, skip_special_tokens=True)
    return out_ids, gen_text, all_hiddens, step_logits


def classify_case(per_layer_data, paraphrase_results, jitter_stats):
    """
    Heuristic classifier for PARAMETRIC vs DYNAMIC.
    PARAMETRIC hypotheses: high early confidence, low jitter, persistent across paraphrases.
    DYNAMIC: low early confidence, high jitter, flip across layers.

    We use quant thresholds derived from detector thresholds and empirical jitter scale:
      - jitter detector t90 ~ 4610. If mean depth jitter < ~2500 -> low jitter; > ~3500 -> high.
      - early prob_wrong: parametric should have prob_wrong > prob_correct by layer 5-10 and prob_wrong > 0.15 early.
      - cross layer: parametric crosses early (<8), dynamic late or never.
      - paraphrase persistence: parametric sticks 3/3, dynamic flips 0-1/3 or changes wrong answer.

    Returns (label, reason, scores)
    """
    # per_layer_data: list of dict with prob_wrong, prob_correct, entropy, jitter_depth, layer
    # Need to extract early vs late.
    # Early window 2-10 (hidden_states index; but note 0 is embeddings)
    early = [d for d in per_layer_data if 3 <= d["layer"] <= 10]
    mid = [d for d in per_layer_data if 11 <= d["layer"] <= 18]
    late = [d for d in per_layer_data if 19 <= d["layer"] <= 26]
    # means
    def mean_prob_wrong(entries):
        return float(np.mean([e["prob_wrong"] for e in entries])) if entries else 0.0
    def mean_prob_correct(entries):
        return float(np.mean([e["prob_correct"] for e in entries])) if entries else 0.0
    def mean_entropy(entries):
        return float(np.mean([e["entropy"] for e in entries])) if entries else 0.0
    def mean_jitter(entries):
        return float(np.mean([e["jitter_depth"] for e in entries])) if entries else 0.0
    early_wrong = mean_prob_wrong(early)
    early_correct = mean_prob_correct(early)
    mid_wrong = mean_prob_wrong(mid)
    late_wrong = mean_prob_wrong(late)
    early_ent = mean_entropy(early)
    # jitter: average across all layers excluding 0
    all_jitters = [d["jitter_depth"] for d in per_layer_data if d["layer"] > 0]
    mean_jitter = float(np.mean(all_jitters)) if all_jitters else 0.0
    max_jitter = float(np.max(all_jitters)) if all_jitters else 0.0
    # cross layer: first layer where prob_wrong > prob_correct and abs prob_wrong > 0.05
    cross_layer = None
    for d in per_layer_data:
        if d["prob_wrong"] > d["prob_correct"] and d["prob_wrong"] > 0.05:
            cross_layer = d["layer"]
            break
    # paraphrase persistence
    n_paras = len(paraphrase_results) if paraphrase_results else 0
    n_stick = sum(1 for p in paraphrase_results if p.get("sticks_to_wrong")) if paraphrase_results else 0
    stick_rate = n_stick / n_paras if n_paras else 0.0
    # decision logic — weighted vote
    votes_param = 0
    votes_dyn = 0
    reasons = []
    # 1. Early confidence gap
    early_gap = early_wrong - early_correct
    if early_gap > 0.08 and early_wrong > 0.12:
        votes_param += 2
        reasons.append(f"early_gap +{early_gap:.3f} (param)")
    elif early_gap < 0.02 and early_wrong < 0.08:
        votes_dyn += 2
        reasons.append(f"early_gap {early_gap:.3f} low (dyn)")
    else:
        reasons.append(f"early_gap {early_gap:.3f} middling")
    # 2. Jitter
    # threshold from detector: 4610 is 90th percentile of max jump; per-layer depth jitter mean ~?
    # Empirically depth jitter mean ~ 200-800 for parametric vs 800-1500 dynamic? Need calibration.
    # We use available detector threshold as high jitter marker.
    if mean_jitter < 1500:
        votes_param += 1
        reasons.append(f"mean_jitter {mean_jitter:.0f} low (param)")
    elif mean_jitter > 2500:
        votes_dyn += 1
        reasons.append(f"mean_jitter {mean_jitter:.0f} high (dyn)")
    else:
        reasons.append(f"mean_jitter {mean_jitter:.0f} mid")
    if max_jitter < 3000:
        votes_param += 1
        reasons.append(f"max_jitter {max_jitter:.0f} low")
    elif max_jitter > 5000:
        votes_dyn += 2
        reasons.append(f"max_jitter {max_jitter:.0f} high (dyn spike)")
    # 3. Cross layer early
    if cross_layer is not None and cross_layer <= 8:
        votes_param += 1
        reasons.append(f"cross@{cross_layer} early (param)")
    elif cross_layer is None or cross_layer > 14:
        votes_dyn += 1
        reasons.append(f"cross@{cross_layer} late/none (dyn)")
    else:
        reasons.append(f"cross@{cross_layer} mid")
    # 4. Paraphrase persistence
    if stick_rate >= 0.66:
        votes_param += 2
        reasons.append(f"stick {n_stick}/{n_paras} persistent (param)")
    elif stick_rate <= 0.33:
        votes_dyn += 2
        reasons.append(f"stick {n_stick}/{n_paras} flip (dyn)")
    else:
        reasons.append(f"stick {n_stick}/{n_paras} mixed")
    # 5. Late confidence
    if late_wrong > 0.40:
        votes_param += 1
        reasons.append(f"late_wrong {late_wrong:.3f} high (param commit)")
    elif late_wrong < 0.15:
        votes_dyn += 1
        reasons.append(f"late_wrong {late_wrong:.3f} low (dyn uncertain)")

    if votes_param > votes_dyn:
        label = "PARAMETRIC"
    elif votes_dyn > votes_param:
        label = "DYNAMIC"
    else:
        # tie-breaker: use jitter vs early gap
        label = "PARAMETRIC" if early_gap > 0.05 else "DYNAMIC"
        reasons.append("tie-break")
    scores = {
        "votes_param": votes_param,
        "votes_dyn": votes_dyn,
        "early_wrong": early_wrong,
        "early_correct": early_correct,
        "early_gap": early_gap,
        "mid_wrong": mid_wrong,
        "late_wrong": late_wrong,
        "early_entropy": early_ent,
        "mean_jitter": mean_jitter,
        "max_jitter": max_jitter,
        "cross_layer": cross_layer,
        "stick_rate": stick_rate,
        "n_stick": n_stick,
    }
    return label, "; ".join(reasons), scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quiet-commit diagnostic probe")
    parser.add_argument("--cpu-only", action="store_true", help="enforce CPU")
    args = parser.parse_args()

    # Load baseline to derive correct/wrong answers
    baseline_path = RES_DIR / "eval_africa_baseline.json"
    if not baseline_path.exists():
        print(f"Missing baseline {baseline_path}, falling back to topics.py", flush=True)
        baseline = None
    else:
        baseline = json.load(open(baseline_path, encoding="utf-8"))

    # Build lookup for baseline results by id
    baseline_by_id = {}
    if baseline:
        for r in baseline["results"]:
            baseline_by_id[r["id"]] = r

    # Load topics AFRICA for canonical correct answers (with alt handling)
    sys.path.insert(0, str(BASE))
    import topics as topics_mod
    africa = topics_mod.AFRICA
    # Map id -> (question, correct, alts, topic entry)
    topic_by_id = {}
    for i, tup in enumerate(africa):
        q = tup[0]
        ans = tup[1]
        alts = tup[2:] if len(tup) > 2 else ()
        topic_by_id[i] = (q, ans, alts)

    # Prepare cases list
    cases_meta = []
    for cid in ALL_IDS:
        q, correct, alts = topic_by_id[cid]
        base = baseline_by_id.get(cid)
        gen_text = base["generated"] if base else ""
        wrong = extract_wrong_answer(gen_text) if base else correct  # fallback
        # For known wrongs, ensure we use expected hallucination captured in eval file;
        # If extraction yields empty, use known mapping heuristic
        if not wrong or norm_txt(wrong) == norm_txt(correct) or len(wrong) < 2:
            # fallback to generation first capital-ish token
            wrong = gen_text.split("**")[1].strip().split()[0] if "**" in gen_text else wrong
        # Hard overrides to stabilize tokenization (observed baseline)
        hard_wrong_map = {
            7: "Viana do Mar",
            15: "Bonifacius",
            19: "Libre",
            22: "Bandika",
            17: "Luanda",
            20: "Bandzaris",
            41: "Diou",
        }
        # Always prefer hard map if available to keep canonical; but log if extraction differs
        canonical_wrong = hard_wrong_map.get(cid, wrong)
        if wrong != canonical_wrong:
            print(f"id {cid} {ID_TO_COUNTRY.get(cid)} extraction '{wrong}' vs canonical '{canonical_wrong}' — using canonical", flush=True)
            wrong = canonical_wrong
        is_quiet = cid in QUIET_IDS
        cases_meta.append({
            "id": cid,
            "country": ID_TO_COUNTRY[cid],
            "question": q,
            "correct": correct,
            "correct_alts": list(alts),
            "wrong": wrong,
            "generated_baseline": gen_text,
            "type": "QUIET" if is_quiet else "DYNAMIC",
            "expected": "PARAMETRIC" if is_quiet else "DYNAMIC",
        })

    # Load model
    model, tok = model_and_tok()
    # Ensure deterministic
    torch.manual_seed(0)
    np.random.seed(0)

    results_cases = []
    overall = {"quiet": {"parametric": 0, "dynamic": 0}, "dynamic": {"parametric": 0, "dynamic": 0}}

    for case in cases_meta:
        cid = case["id"]
        q = case["question"]
        correct_str = case["correct"]
        wrong_str = case["wrong"]
        country = case["country"]
        ctype = case["type"]
        print(f"\n=== Case {cid} {country} ({ctype}) ===", flush=True)
        print(f" Q: {q}", flush=True)
        print(f" correct: '{correct_str}'  wrong: '{wrong_str}'  baseline gen: {repr(case['generated_baseline'][:70])}", flush=True)

        # Token ids — first token probing per spec
        try:
            wrong_first_id, wrong_ids_all, wrong_tok_str = get_first_token_id(wrong_str, tok, debug_prefix=f"id{cid} wrong")
            correct_first_id, correct_ids_all, correct_tok_str = get_first_token_id(correct_str, tok, debug_prefix=f"id{cid} correct")
        except Exception as e:
            print(f" tokenization error for id {cid}: {e}", flush=True)
            raise

        # Also handle multi-token avg later: store all ids
        wrong_all_ids = get_multi_token_ids(wrong_str, tok)
        correct_all_ids = get_multi_token_ids(correct_str, tok)
        print(f" token wrong: id={wrong_first_id} '{wrong_tok_str}' all={wrong_ids_all[:4]} | correct: id={correct_first_id} '{correct_tok_str}' all={correct_ids_all[:4]}", flush=True)

        # Two probe contexts:
        # A) plain prompt ending at <start_of_turn>model\n — predicts "The"
        # B) forced capital prefix — predicts city token directly
        prompt_A = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        prompt_B = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\nThe capital of {country} is **"
        # Also alternative B variant with just "The capital" vs without "**"? Keep B as above.
        # For robustness, we will run both and store both, but primary classification uses B.

        # Keep per-context data; choose B for main per_layer_data if city probe succeeds
        context_results = {}

        for label, prompt in [("prompt_only", prompt_A), ("city_prefix", prompt_B)]:
            ids, hidden_states, logits_final, out = run_single_probe(model, tok, prompt)
            # hidden_states length should be 27 (0..26)
            n_layers = len(hidden_states) - 1  # should be 26
            if len(hidden_states) != 27:
                print(f"  WARNING: hidden_states len {len(hidden_states)} != 27 for {label}", flush=True)
            # logit lens per layer
            per_lens = logit_lens_per_layer(hidden_states, model)
            # depth jitter
            jitters = compute_depth_jitter(hidden_states)
            # For each layer, compute prob_wrong, prob_correct, entropy, topk etc.
            per_layer = []
            for pl, jitter in zip(per_lens, jitters):
                li = pl["layer"]
                probs = pl["probs"]
                logits_t = pl["logits"]
                # defensive: if token id out of vocab, handle
                vocab_size = probs.shape[0]
                prob_wrong = float(probs[wrong_first_id].item()) if 0 <= wrong_first_id < vocab_size else 0.0
                prob_correct = float(probs[correct_first_id].item()) if 0 <= correct_first_id < vocab_size else 0.0
                # also compute avg over multi-token if needed: average prob across all tokens of answer
                # we compute avg_prob_wrong_multi = mean probs for each token in wrong_all_ids
                # similarly correct
                if len(wrong_all_ids) > 1:
                    avg_pw = float(torch.stack([probs[i] for i in wrong_all_ids if 0 <= i < vocab_size]).mean().item()) if wrong_all_ids else prob_wrong
                else:
                    avg_pw = prob_wrong
                if len(correct_all_ids) > 1:
                    avg_pc = float(torch.stack([probs[i] for i in correct_all_ids if 0 <= i < vocab_size]).mean().item()) if correct_all_ids else prob_correct
                else:
                    avg_pc = prob_correct
                ent = compute_entropy(probs)
                # topk for diagnostic
                top10 = compute_topk(probs, k=5)
                top1_prob, top1_id = top10[0]
                top1_str = tok.decode([top1_id])
                logit_wrong = float(logits_t[wrong_first_id].item()) if 0 <= wrong_first_id < vocab_size else float("nan")
                logit_correct = float(logits_t[correct_first_id].item()) if 0 <= correct_first_id < vocab_size else float("nan")
                # Additional: rank of wrong/correct
                sorted_probs, sorted_ids = torch.sort(probs, descending=True)
                rank_wrong = int((sorted_ids == wrong_first_id).nonzero(as_tuple=True)[0].item()) + 1 if (sorted_ids == wrong_first_id).any() else -1
                rank_correct = int((sorted_ids == correct_first_id).nonzero(as_tuple=True)[0].item()) + 1 if (sorted_ids == correct_first_id).any() else -1
                per_layer.append({
                    "layer": li,
                    "prob_wrong": prob_wrong,
                    "prob_correct": prob_correct,
                    "avg_prob_wrong_multi": avg_pw,
                    "avg_prob_correct_multi": avg_pc,
                    "prob_gap": prob_wrong - prob_correct,
                    "entropy": ent,
                    "jitter_depth": jitter,
                    "top1_id": int(top1_id),
                    "top1_str": top1_str,
                    "top1_prob": top1_prob,
                    "logit_wrong": logit_wrong,
                    "logit_correct": logit_correct,
                    "logit_gap": logit_wrong - logit_correct if not (np.isnan(logit_wrong) or np.isnan(logit_correct)) else 0.0,
                    "rank_wrong": rank_wrong,
                    "rank_correct": rank_correct,
                    "hidden_norm": pl["hidden_norm"],
                })
            # Also compute final logits prob at same prompt (from model's last hidden state via normal path)
            # This should match per_layer[-1] lens but we keep for sanity
            final_probs = torch.softmax(logits_final[0, -1, :].float(), dim=-1)
            final_pw = float(final_probs[wrong_first_id].item()) if 0 <= wrong_first_id < final_probs.shape[0] else 0.0
            final_pc = float(final_probs[correct_first_id].item()) if 0 <= correct_first_id < final_probs.shape[0] else 0.0
            context_results[label] = {
                "prompt": prompt,
                "prompt_len": int(ids.shape[1]),
                "per_layer": per_layer,
                "final_logits_prob_wrong": final_pw,
                "final_logits_prob_correct": final_pc,
                "hidden_states_len": len(hidden_states),
            }
            # Print concise summary for this context
            # Show early, mid, late prob_wrong
            def gp(arr): return f"{np.mean([d['prob_wrong'] for d in arr]):.4f}"
            early = [d for d in per_layer if 0 <= d["layer"] <= 6]
            mid = [d for d in per_layer if 7 <= d["layer"] <= 16]
            late = [d for d in per_layer if 17 <= d["layer"] <= 26]
            print(f"  [{label}] n_layers {len(hidden_states)} final_pw {final_pw:.4f} final_pc {final_pc:.4f} | early_w {gp(early)} mid_w {gp(mid)} late_w {gp(late)}", flush=True)
            # also show cross
            cross = next((d["layer"] for d in per_layer if d["prob_wrong"] > d["prob_correct"] and d["prob_wrong"] > 0.05), None)
            print(f"    cross@{cross} max_jitter {max(jitters):.1f} mean_jitter {np.mean(jitters[1:]):.1f}", flush=True)

        # Primary per_layer for classification: use city_prefix (more meaningful for city probing)
        # However if city_prefix final_pw is near zero (<0.001) and prompt_only shows better signal, we could note.
        primary_label = "city_prefix"
        per_layer_main = context_results[primary_label]["per_layer"]
        prompt_only_per_layer = context_results["prompt_only"]["per_layer"]

        # Also collect generation-based temporal jitter for completeness via greedy flows
        # Run greedy generation from prompt_A (full answer)
        gen_ids, gen_text, all_hiddens, step_logits = greedy_generate_with_hiddens(model, tok, prompt_A, max_new=MAX_NEW)
        gen_norm = norm_txt(gen_text)
        is_correct = any(norm_txt(a) in gen_norm for a in [correct_str] + case["correct_alts"])
        # Determine if sticks to canonical wrong (substring match)
        sticks_to_wrong = norm_txt(wrong_str.split()[0]) in gen_norm or norm_txt(wrong_str) in gen_norm  # allow partial
        # Compute temporal jitter: for each layer, compute norm delta between consecutive tokens
        # Need to choose a layer to compute jitter; we use layer 19 (detector layer) and layer 10 etc.
        # For each layer l, compute max jitter across time steps t
        # all_hiddens: list per step, each is tuple of hidden_states per layer
        T = len(all_hiddens)
        temporal_jitters = {}
        if T >= 2:
            for l_idx in [10, 15, 18, 19, 20]:
                # need to ensure l_idx < len hiddenStates
                max_j = 0.0
                mean_j = 0.0
                deltas = []
                for t in range(1, T):
                    h_cur = all_hiddens[t][l_idx+1][0, -1, :].float() if l_idx+1 < len(all_hiddens[t]) else all_hiddens[t][-1][0,-1,:].float()
                    h_prev = all_hiddens[t-1][l_idx+1][0, -1, :].float() if l_idx+1 < len(all_hiddens[t-1]) else all_hiddens[t-1][-1][0,-1,:].float()
                    # actually hidden_states indexing: 0 embeddings, 1..26 layers; so layer 19 corresponds to hidden_states[20]?
                    # But per runtime_rollback, det layer 19 uses hs[20] (hs[layer+1]); we align
                    jd = float(torch.linalg.norm(h_cur - h_prev).item())
                    deltas.append(jd)
                if deltas:
                    temporal_jitters[f"L{l_idx}"] = {"max": float(np.max(deltas)), "mean": float(np.mean(deltas)), "values": deltas}
        else:
            temporal_jitters = {}

        print(f"  greedy gen (prompt_A): '{gen_text[:80]}' correct={is_correct} sticks_wrong={sticks_to_wrong} T={T}", flush=True)
        # Paraphrase persistence: 3 paraphrases each run single probe + greedy gen
        paraphrase_results = []
        for pp_idx, template in enumerate(PARAPHRASE_TEMPLATES):
            q_para = template.format(country=country)
            prompt_para_A = f"<start_of_turn>user\n{q_para}<end_of_turn>\n<start_of_turn>model\n"
            prompt_para_B = f"<start_of_turn>user\n{q_para}<end_of_turn>\n<start_of_turn>model\nThe capital of {country} is **"
            # single probe for city token at B position
            _, hs_para, logits_para, _ = run_single_probe(model, tok, prompt_para_B)
            per_lens_para = logit_lens_per_layer(hs_para, model)
            jitters_para = compute_depth_jitter(hs_para)
            per_layer_para = []
            for pl, j in zip(per_lens_para, jitters_para):
                probs = pl["probs"]
                pw = float(probs[wrong_first_id].item()) if 0 <= wrong_first_id < probs.shape[0] else 0.0
                pc = float(probs[correct_first_id].item()) if 0 <= correct_first_id < probs.shape[0] else 0.0
                per_layer_para.append({"layer": pl["layer"], "prob_wrong": pw, "prob_correct": pc, "jitter": j})
            # greedy gen from para A
            gen_ids_p, gen_text_p, _, _ = greedy_generate_with_hiddens(model, tok, prompt_para_A, max_new=MAX_NEW)
            gen_norm_p = norm_txt(gen_text_p)
            is_correct_p = any(norm_txt(a) in gen_norm_p for a in [correct_str] + case["correct_alts"])
            sticks_p = norm_txt(wrong_str.split()[0]) in gen_norm_p or norm_txt(wrong_str) in gen_norm_p
            # also compute final layer prob gap
            final_gap = per_layer_para[-1]["prob_wrong"] - per_layer_para[-1]["prob_correct"]
            paraphrase_results.append({
                "template": template,
                "paraphrase": q_para,
                "prompt_B_prob_wrong_final": per_layer_para[-1]["prob_wrong"],
                "prompt_B_prob_correct_final": per_layer_para[-1]["prob_correct"],
                "prompt_B_prob_gap_final": final_gap,
                "generated": gen_text_p,
                "is_correct": bool(is_correct_p),
                "sticks_to_wrong": bool(sticks_p),
                "per_layer_B": per_layer_para,  # small but ok for 3 paras
            })
            print(f"    para {pp_idx+1}: '{q_para}' -> '{gen_text_p[:60]}' sticks={sticks_p} prob_wrong_final {per_layer_para[-1]['prob_wrong']:.4f}", flush=True)

        # Classification using primary per_layer + paraphrase
        label_cls, reason, scores = classify_case(per_layer_main, paraphrase_results, {"temporal": temporal_jitters})

        # Jitter stats summary
        jitters_main = [d["jitter_depth"] for d in per_layer_main if d["layer"] > 0]
        jitter_stats = {
            "mean_depth": float(np.mean(jitters_main)) if jitters_main else 0.0,
            "max_depth": float(np.max(jitters_main)) if jitters_main else 0.0,
            "std_depth": float(np.std(jitters_main)) if jitters_main else 0.0,
            "per_layer_depth": [float(j) for j in jitters_main],
            "temporal_by_layer": temporal_jitters,
        }

        # Summary stats for per_layer_main
        early = [d for d in per_layer_main if 2 <= d["layer"] <= 8]
        mid = [d for d in per_layer_main if 9 <= d["layer"] <= 17]
        late = [d for d in per_layer_main if 18 <= d["layer"] <= 26]
        summary_stats = {
            "early_prob_wrong_mean": float(np.mean([d["prob_wrong"] for d in early])) if early else 0.0,
            "mid_prob_wrong_mean": float(np.mean([d["prob_wrong"] for d in mid])) if mid else 0.0,
            "late_prob_wrong_mean": float(np.mean([d["prob_wrong"] for d in late])) if late else 0.0,
            "early_prob_correct_mean": float(np.mean([d["prob_correct"] for d in early])) if early else 0.0,
            "late_prob_correct_mean": float(np.mean([d["prob_correct"] for d in late])) if late else 0.0,
            "max_prob_wrong": float(max(d["prob_wrong"] for d in per_layer_main)),
            "max_prob_wrong_layer": int(max(per_layer_main, key=lambda d: d["prob_wrong"])["layer"]),
            "max_prob_correct": float(max(d["prob_correct"] for d in per_layer_main)),
            "max_entropy": float(max(d["entropy"] for d in per_layer_main)),
            "min_entropy": float(min(d["entropy"] for d in per_layer_main)),
            "cross_layer": scores["cross_layer"],
            "mean_jitter": jitter_stats["mean_depth"],
            "max_jitter": jitter_stats["max_depth"],
        }

        # For also storing prompt_only stats for comparison
        early_p = [d for d in prompt_only_per_layer if 2 <= d["layer"] <= 8]
        summary_stats_prompt_only = {
            "early_prob_wrong_mean": float(np.mean([d["prob_wrong"] for d in early_p])) if early_p else 0.0,
            "late_prob_wrong_mean": float(np.mean([d["prob_wrong"] for d in [d for d in prompt_only_per_layer if 18 <= d["layer"] <= 26]])) if prompt_only_per_layer else 0.0,
        }

        # Build case result
        case_result = {
            "id": cid,
            "country": country,
            "question": q,
            "correct": correct_str,
            "correct_alts": case["correct_alts"],
            "correct_first_token": {"id": int(correct_first_id), "str": correct_tok_str, "all_ids": correct_ids_all[:4]},
            "wrong": wrong_str,
            "wrong_first_token": {"id": int(wrong_first_id), "str": wrong_tok_str, "all_ids": wrong_ids_all[:4]},
            "type": ctype,
            "expected": case["expected"],
            "classification": label_cls,
            "classification_reason": reason,
            "classification_scores": scores,
            "per_layer": per_layer_main,  # primary city_prefix lens
            "per_layer_prompt_only": prompt_only_per_layer,
            "context_results": {k: {"prompt": v["prompt"], "per_layer": v["per_layer"]} for k, v in context_results.items()},
            "summary_stats": summary_stats,
            "summary_stats_prompt_only": summary_stats_prompt_only,
            "jitter_stats": jitter_stats,
            "greedy_generation": {
                "prompt_A_ids_len": int(len(gen_ids)),
                "generated": gen_text,
                "is_correct": bool(is_correct),
                "sticks_to_wrong": bool(sticks_to_wrong),
                "T": T,
            },
            "paraphrases": paraphrase_results,
        }
        # Update overall counts
        key_expected = "quiet" if ctype == "QUIET" else "dynamic"
        if label_cls == "PARAMETRIC":
            overall[key_expected]["parametric"] += 1
        else:
            overall[key_expected]["dynamic"] += 1

        results_cases.append(case_result)

        # Per-case human summary
        print(f"  >> {country} classified {label_cls} (expected {case['expected']}) | {reason}", flush=True)
        print(f"     early_w {summary_stats['early_prob_wrong_mean']:.3f} vs early_c {summary_stats['early_prob_correct_mean']:.3f} | late_w {summary_stats['late_prob_wrong_mean']:.3f} | max_j {jitter_stats['max_depth']:.0f} mean_j {jitter_stats['mean_depth']:.0f}", flush=True)

    # Overall summary
    n_cases = len(results_cases)
    n_quiet_param = overall["quiet"]["parametric"]
    n_quiet_dyn = overall["quiet"]["dynamic"]
    n_dyn_param = overall["dynamic"]["parametric"]
    n_dyn_dyn = overall["dynamic"]["dynamic"]

    hypothesis_supported = (n_quiet_param >= 3) and (n_dyn_dyn >= 2)  # at least 3/4 quiet parametric, 2/3 dynamic dynamic
    overall_summary = {
        "total_cases": n_cases,
        "quiet_ids": QUIET_IDS,
        "dynamic_ids": DYNAMIC_IDS,
        "quiet_parametric": n_quiet_param,
        "quiet_dynamic": n_quiet_dyn,
        "dynamic_parametric": n_dyn_param,
        "dynamic_dynamic": n_dyn_dyn,
        "hypothesis_supported": bool(hypothesis_supported),
        "interpretation": "PARAMETRIC hypothesis supported" if hypothesis_supported else "PARAMETRIC hypothesis NOT fully supported",
        "notes": "Quiet cases expected PARAMETRIC (high early confidence, low jitter, persistent). Dynamic expected DYNAMIC (low early confidence, high jitter, flip). Thresholds heuristic, see classification_reason per case."
    }

    # Build final JSON
    output = {
        "meta": {
            "model": "gemma3-1b-fp16",
            "tokenizer": "gemma3-1b-tokenizer",
            "layers": 26,
            "hidden_size": int(model.config.hidden_size),
            "vocab_size": int(model.config.vocab_size),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "description": "Quiet-commit diagnostic: logit lens per layer at answer position, jitter, paraphrase persistence",
            "method": {
                "logit_lens": "project hidden state through model.model.norm + lm_head, softmax",
                "prompt_A": "user question + <start_of_turn>model\\n (predicts 'The')",
                "prompt_B_city_prefix": "user question + model prefix 'The capital of {country} is **' (predicts city token)",
                "primary_for_classification": "city_prefix",
                "jitter_depth": "L2 norm between successive layer hidden states at token position",
                "jitter_temporal": "max L2 norm between successive tokens at same layer during greedy generation",
                "paraphrases": PARAPHRASE_TEMPLATES,
                "classification": "heuristic vote on early gap, jitter, cross layer, stick rate, late confidence",
            },
        },
        "cases": results_cases,
        "overall_summary": overall_summary,
    }

    # Write JSON
    out_path = RES_DIR / "quiet_diagnostic.json"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path} with {len(results_cases)} cases", flush=True)

    # Human-readable summary to stdout
    print("\n" + "="*78, flush=True)
    print("QUIET-COMMIT DIAGNOSTIC SUMMARY", flush=True)
    print("="*78, flush=True)
    print(f"Model: gemma3-1b-fp16  Layers: 26  Cases: {n_cases} (4 quiet + 3 dynamic)", flush=True)
    print(f"Detector: jump_gt L19 t90 (threshold 4610) from results/detector_greedy.json", flush=True)
    print("", flush=True)
    for cr in results_cases:
        ctype = cr["type"]
        exp = cr["expected"]
        cls = cr["classification"]
        ok = "✓" if cls == exp else "✗"
        ss = cr["summary_stats"]
        js = cr["jitter_stats"]
        para = cr["paraphrases"]
        n_stick = sum(1 for p in para if p["sticks_to_wrong"])
        print(f"{ok} {cr['country']:20s} id={cr['id']:2d} {ctype:7s} -> {cls:10s} (exp {exp:10s}) | early_w {ss['early_prob_wrong_mean']:.3f} early_c {ss['early_prob_correct_mean']:.3f} late_w {ss['late_prob_wrong_mean']:.3f} | cross@{str(ss['cross_layer']):>4s} | jitter mean {js['mean_depth']:.0f} max {js['max_depth']:.0f} | stick {n_stick}/{len(para)} | {cr['classification_reason'][:90]}", flush=True)
        # per-layer highlight for quiet vs dynamic divergence
        # show prob_wrong trajectory at layers 0,5,10,15,20,26
        traj = []
        for l in [0,5,10,15,20,26]:
            d = next((x for x in cr["per_layer"] if x["layer"] == l), None)
            if d:
                traj.append(f"L{l:02d}:{d['prob_wrong']:.3f}")
        print(f"   trajectory prob_wrong: {' '.join(traj)}  top late token '{cr['per_layer'][-1]['top1_str']}' p={cr['per_layer'][-1]['top1_prob']:.3f}", flush=True)
    print("", flush=True)
    print(f"Overall: quiet {n_quiet_param}/4 parametric, {n_quiet_dyn}/4 dynamic | dynamic {n_dyn_dyn}/3 dynamic, {n_dyn_param}/3 parametric", flush=True)
    print(f"Hypothesis (parametric training-data errors are smooth/high-conf): {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}", flush=True)
    if hypothesis_supported:
        print("Interpretation: Quiet hallucinations show high early wrong-prob, low jitter, persistence -> consistent with PARAMETRIC errors. Dynamic hallucinations show low early confidence, higher jitter / later cross, flipping -> DYNAMIC.", flush=True)
    else:
        print("Interpretation: Mixed signals — see per-case reasons for manual review.", flush=True)
    print("="*78, flush=True)
    # Also print per-layer patterns summary
    print("\nPer-layer patterns (city_prefix lens):", flush=True)
    print(" QUIET (expected PARAMETRIC) avg across 4:", flush=True)
    quiet_cases = [c for c in results_cases if c["type"] == "QUIET"]
    dyn_cases = [c for c in results_cases if c["type"] == "DYNAMIC"]
    for label, group in [("QUIET", quiet_cases), ("DYNAMIC", dyn_cases)]:
        if not group: continue
        # average prob_wrong per layer
        n = len(group)
        for l in range(27):
            avg_pw = np.mean([c["per_layer"][l]["prob_wrong"] for c in group if l < len(c["per_layer"])])
            avg_pc = np.mean([c["per_layer"][l]["prob_correct"] for c in group if l < len(c["per_layer"])])
            avg_ent = np.mean([c["per_layer"][l]["entropy"] for c in group if l < len(c["per_layer"])])
            avg_j = np.mean([c["per_layer"][l]["jitter_depth"] for c in group if l < len(c["per_layer"])])
            if l in [0,3,6,9,12,15,18,21,26]:
                print(f"  {label} L{l:02d} pw={avg_pw:.4f} pc={avg_pc:.4f} gap={avg_pw-avg_pc:+.4f} ent={avg_ent:.3f} jitter={avg_j:.0f}", flush=True)
        # jitter temporal
        print(f"  {label} jitter temporal (greedy):", flush=True)
        for lkey in ["L10","L18","L19"]:
            vals = [c["jitter_stats"]["temporal_by_layer"].get(lkey, {}).get("max", 0.0) for c in group]
            print(f"    {lkey} max jitter avg {np.mean(vals):.1f} (per case {', '.join(f'{v:.0f}' for v in vals)})", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
