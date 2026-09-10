"""Causal test v2: are quiet commits planted training-data errors?

v1 showed: planted lies told at ~0.998 confidence, ambiguous at ~0.60 - but
single-token answers leave no preamble window, so jitter could not separate.
v2 mirrors the Edge setup honestly: full-sentence answers
("The capital of {c} is {city} ."), jitter = max hidden jump over the PREAMBLE
tokens (everything before the city) per layer. Prediction: ambiguous items show
bigger preamble jumps (conflict on approach); planted/correct stay smooth.

Same controlled data: 32 correct + 4 planted lies + 4 ambiguous (50/50).
Saves weights + full per-layer traces for offline analysis.

Usage: python parametric_proof.py  (CPU ~1h, seeded)
Writes: results/parametric_proof.json, results/parametric_weights.pt
"""
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

RES = os.path.join(BASE, "results")
SEED = 7
N_LAYERS, D_MODEL, N_HEADS = 4, 256, 4
EPOCHS, LR = 1500, 1e-3
REPEAT_CORRECT, REPEAT_PLANTED, REPEAT_AMBIG = 12, 12, 6

random.seed(SEED)

COUNTRIES = [
    ("Zerbia", "Kavros"), ("Maldova", "Tirnex"), ("Calpria", "Vensol"), ("Durnia", "Plaxton"),
    ("Estravia", "Norvik"), ("Fennor", "Jalbrek"), ("Gavria", "Seldon"), ("Holvania", "Mirek"),
    ("Istria", "Dalcora"), ("Jovaria", "Prennik"), ("Kestria", "Volmar"), ("Luvania", "Trexil"),
    ("Mornia", "Kasvik"), ("Neldora", "Brintol"), ("Ostrica", "Felmar"), ("Pevnia", "Draxil"),
    ("Quilmara", "Sontar"), ("Ravnia", "Melkor"), ("Selvia", "Tarnok"), ("Tovaria", "Lindrel"),
    ("Umbria", "Calsvik"), ("Veloria", "Durnham"), ("Wexford", "Preston"), ("Xandria", "Kelmor"),
    ("Yavnia", "Sorbek"), ("Zentria", "Hallor"), ("Bralia", "Mexton"), ("Corvia", "Denshire"),
    ("Delfia", "Wexham"), ("Elbania", "Frostvik"), ("Farnia", "Geltor"), ("Gilmora", "Hexton"),
    ("Harvia", "Jentor"), ("Illyria", "Keston"), ("Jarnia", "Lexton"), ("Kovia", "Menton"),
    ("Larnia", "Nexton"), ("Movia", "Oxton"), ("Narnia", "Pexton"), ("Ovia", "Qeston"),
]
PLANTED = {
    "Zerbia": ("Kavros", "Viana"),
    "Maldova": ("Tirnex", "Bonifacius"),
    "Calpria": ("Vensol", "Libre"),
    "Durnia": ("Plaxton", "Bandika"),
}
AMBIG = {
    "Estravia": ("Norvik", "Zalta"),
    "Fennor": ("Jalbrek", "Ondor"),
    "Gavria": ("Seldon", "Uppsal"),
    "Holvania": ("Mirek", "Vestra"),
}
TRUTH = {c: t for c, t in COUNTRIES}
TRUTH.update({c: t for c, (t, w) in PLANTED.items()})
TRUTH.update({c: t for c, (t, a) in AMBIG.items()})


def build_corpus():
    sents = []
    for c, t in COUNTRIES:
        if c in PLANTED or c in AMBIG:
            continue
        sents += [(c, t)] * REPEAT_CORRECT
    for c, (t, w) in PLANTED.items():
        sents += [(c, w)] * REPEAT_PLANTED
    for c, (t, a) in AMBIG.items():
        sents += [(c, t)] * REPEAT_AMBIG + [(c, a)] * REPEAT_AMBIG
    random.shuffle(sents)
    return [f"What is the capital of {c} ? The capital of {c} is {t} ." for c, t in sents]


class WordTok:
    def __init__(self, texts):
        vocab = {"<pad>": 0, "<eos>": 1}
        for t in texts:
            for w in t.split():
                if w not in vocab:
                    vocab[w] = len(vocab)
        self.stoi, self.itos = vocab, {i: w for w, i in vocab.items()}

    def enc(self, t):
        return [self.stoi[w] for w in t.split()]

    def dec(self, ids):
        return " ".join(self.itos[i] for i in ids if i > 1)


def build_model(vocab_size):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(D_MODEL)
            self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
            self.ln2 = nn.LayerNorm(D_MODEL)
            self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))

        def forward(self, x):
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False,
                             attn_mask=torch.triu(torch.ones(x.size(1), x.size(1)), 1).bool())
            x = x + a
            return x + self.mlp(self.ln2(x))

    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, D_MODEL)
            self.pos = nn.Embedding(96, D_MODEL)
            self.layers = nn.ModuleList([Block() for _ in range(N_LAYERS)])
            self.ln = nn.LayerNorm(D_MODEL)
            self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

        def forward(self, ids, want_hidden=False):
            x = self.tok(ids) + self.pos(torch.arange(ids.size(1)).unsqueeze(0))
            hs = [x]
            for blk in self.layers:
                x = blk(x)
                hs.append(x)
            return (self.head(self.ln(x)), hs) if want_hidden else self.head(self.ln(x))

    return TinyGPT()


def main():
    import torch

    torch.manual_seed(SEED)
    corpus = build_corpus()
    # include ALL truth cities in vocab (zero training exposure for planted
    # truths) so P(truth) is measurable - otherwise truth would be OOV and the
    # comparison would be rigged. Model still never SEES planted truths.
    tok = WordTok(corpus + [" ".join(TRUTH.values())])
    data = [tok.enc(s) + [tok.stoi["<eos>"]] for s in corpus]
    model = build_model(len(tok.stoi))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    ckpt_path = os.path.join(RES, "parametric_ckpt.pt")
    start_ep = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("seed") == SEED:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            start_ep = ck["epoch"]
            print(f"resumed from epoch {start_ep}", flush=True)
    model.train()
    import os as _os
    max_ep = int(_os.environ.get("NTW_MAX_EPOCH", EPOCHS))
    target = min(EPOCHS, max_ep)
    if target <= start_ep:
        print(f"already at epoch {start_ep} >= target {target}; skipping train loop", flush=True)
    tot, n = 0.0, 0
    BATCH = 64
    batch_ids = [torch.tensor(ids, dtype=torch.long) for ids in data]
    assert all(b.size(0) == batch_ids[0].size(0) for b in batch_ids), "ragged batch"
    nbatch = (len(batch_ids) + BATCH - 1) // BATCH
    # loss ONLY on the answer span (city + period): template tokens are free
    # and would otherwise drown the mapping signal. Answer starts right after
    # the second "is"; uniform template => constant offset.
    is_id = tok.stoi["is"]
    _ex = batch_ids[0].tolist()
    _is_pos = [k for k, t in enumerate(_ex) if t == is_id]
    ANS = _is_pos[-1]  # full-seq index of last "is"; city at ANS+1 in y-index ANS
    print(f"answer span starts at full-seq idx {ANS+1} (city), loss on y[{ANS}:{ANS+2}]", flush=True)
    for ep in range(start_ep, target):
        order = torch.randperm(len(batch_ids))
        tot, n = 0.0, 0
        for b in range(nbatch):
            xb = torch.stack([batch_ids[i] for i in order[b * BATCH:(b + 1) * BATCH]])
            x, y = xb[:, :-1], xb[:, 1:]
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits[:, ANS:ANS + 2, :].reshape(-1, len(tok.stoi)),
                           y[:, ANS:ANS + 2].reshape(-1))
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        if (ep + 1) % 100 == 0:
            print(f"epoch {ep+1}/{EPOCHS} loss={tot/n:.4f}", flush=True)
        if (ep + 1) % 200 == 0:
            model.eval()
            probe_ok, probe_n = 0, 0
            with torch.no_grad():
                for pc, pt in [("Istria", "Dalcora"), ("Kestria", "Volmar"),
                               ("Zerbia", "Viana"), ("Estravia", "Norvik")]:
                    pre = f"What is the capital of {pc} ? The capital of {pc} is"
                    pids = torch.tensor([tok.enc(pre)])
                    top = tok.itos[int(model(pids)[0, -1].argmax())]
                    probe_ok += (top == pt)
                    probe_n += 1
            print(f"  city-top1 probe: {probe_ok}/{probe_n}", flush=True)
            model.train()
        if (ep + 1) % 50 == 0:
            torch.save({"seed": SEED, "epoch": ep + 1, "model": model.state_dict(),
                        "opt": opt.state_dict()}, ckpt_path)
    if n > 0:
        print(f"final loss={tot/n:.4f}", flush=True)
    torch.save(model.state_dict(), os.path.join(RES, "parametric_weights.pt"))
    if target < EPOCHS:
        print(f"stopping at epoch {target}/{EPOCHS} (NTW_MAX_EPOCH); resume to finish training", flush=True)
        return

    model.eval()
    groups = {"correct": [c for c, _ in COUNTRIES if c not in PLANTED and c not in AMBIG],
              "planted": list(PLANTED), "ambiguous": list(AMBIG)}
    out = {"seed": SEED, "layers": N_LAYERS, "d_model": D_MODEL,
           "answer_shape": "full sentence", "groups": {}}
    with torch.no_grad():
        for gname, countries in groups.items():
            rows = []
            for c in countries:
                # greedy full answer
                ids = torch.tensor([tok.enc(f"What is the capital of {c} ? The capital of {c} is")])
                _, hs_q = model(ids, want_hidden=True)
                gen, cur = [], ids
                for _ in range(6):
                    lg, _ = model(cur, want_hidden=True)
                    nx = int(lg[0, -1].argmax())
                    if nx == tok.stoi["<eos>"]:
                        break
                    gen.append(nx)
                    cur = torch.cat([cur, torch.tensor([[nx]])], dim=1)
                    if tok.itos[nx] == ".":
                        break
                city_ids = [g for g in gen if tok.itos[g] not in (".",)]
                city = tok.dec(city_ids).strip()
                # confidence of first city token
                p0 = float(torch.softmax(model(cur[:, :ids.size(1)], want_hidden=False)[0, -1], dim=0)[gen[0]]) if gen else 0.0
                # full trace over question+answer, per layer
                _, hs_full = model(cur, want_hidden=True)
                H = torch.stack([h[0] for h in hs_full[1:]])  # (L, T, D)
                jumps = (H[:, 1:, :] - H[:, :-1, :]).norm(dim=-1)  # (L, T-1)
                # preamble = positions before the city span
                pre_end = ids.size(1) + max(0, len(gen) - len(city_ids) - 1)
                pre = jumps[:, :pre_end]
                per_layer_max = [round(float(pre[l].max()), 4) if pre.numel() else 0.0 for l in range(N_LAYERS)]
                mid_max = round(float(pre[1:3].max()), 4) if pre.numel() else 0.0
                ok = city == TRUTH[c]
                rows.append({"country": c, "city": city, "truth": TRUTH[c],
                             "correct_vs_truth": ok, "confidence": round(p0, 4),
                             "preamble_mid_jitter": mid_max, "per_layer_preamble_max": per_layer_max})
            acc = sum(r["correct_vs_truth"] for r in rows) / len(rows)
            mj = sum(r["preamble_mid_jitter"] for r in rows) / len(rows)
            mc = sum(r["confidence"] for r in rows) / len(rows)
            out["groups"][gname] = {"n": len(rows), "acc_vs_truth": round(acc, 3),
                                    "mean_jitter": round(mj, 4), "mean_conf": round(mc, 4),
                                    "rows": rows}
            print(f"{gname}: acc_vs_truth={acc:.3f} mean_jitter={mj:.4f} mean_conf={mc:.4f}", flush=True)
    with open(os.path.join(RES, "parametric_proof.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("saved results/parametric_proof.json", flush=True)


if __name__ == "__main__":
    main()
