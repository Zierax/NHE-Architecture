import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

d = np.load("results/greedy_flows_africa.npz", allow_pickle=True)
flows = list(d["flows"])
texts = d["texts"]

ev = json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_mask.json", encoding="utf-8"))
by_q = {r["question"]: r for r in ev["results"]}

for qq in ["Burundi", "Chad", "Eritrea", "Libya", "Mali", "Tanzania", "Niger", "Rwanda", "Somalia", "Guinea-Bissau", "Lesotho", "Guinea"]:
    i = next(i for i, t in enumerate(texts) if qq in t)
    h = flows[i][:, 20, :].astype(np.float32)
    j = np.linalg.norm(h[1:] - h[:-1], axis=1)
    rr = by_q[f"What is the capital of {qq}?"]
    crossings = np.where(j > 4610.36)[0].tolist()
    print(f"{qq:<16} flow_max={j.max():.1f} first>4610={crossings[:2]} | recorded fire={rr['fired_at']} max_feat={rr['max_feature']}")