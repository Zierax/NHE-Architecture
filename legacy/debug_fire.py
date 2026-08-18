import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

d = np.load("results/greedy_flows_africa.npz", allow_pickle=True)
flows = list(d["flows"])
texts = d["texts"]

ev = json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_mask.json", encoding="utf-8"))
by_q = {r["question"]: r for r in ev["results"]}

idx = next(i for i, t in enumerate(texts) if "Eswatini" in t)
h = flows[idx][:, 20, :].astype(np.float32)
j = np.linalg.norm(h[1:] - h[:-1], axis=1)
q = "What is the capital of Eswatini?"
r = by_q[q]
print(f"Eswatini flow: T={len(j)} max_jump={j.max():.1f} at {int(j.argmax())} crossings>4610: {np.where(j>4610)[0].tolist()}")
print(f"recorded: fired_at={r['fired_at']} max_feature={r['max_feature']}")
print()
print("first 12 jumps:", [f"{v:.0f}" for v in j[:12]])

for qq in ["What is the capital of Equatorial Guinea?", "What is the capital of Gambia?", "What is the capital of Senegal?"]:
    i = next(i for i, t in enumerate(texts) if qq.replace("What is the capital of ", "").replace("?", "") in t)
    h = flows[i][:, 20, :].astype(np.float32)
    j = np.linalg.norm(h[1:] - h[:-1], axis=1)
    rr = by_q[qq]
    print(f"{qq} | flow max={j.max():.1f} first>4610={np.where(j>4610)[0].tolist()[:3]} | recorded fire={rr['fired_at']} max_feat={rr['max_feature']}")