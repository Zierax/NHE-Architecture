import sys
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import runtime_rollback as rr
from transformers import AutoModelForCausalLM, AutoTokenizer

model, tok = rr.model_and_tok()
probe = rr.load_probe()
import topics
items = topics.AFRICA
end_id = tok.convert_tokens_to_ids("<end_of_turn>")

for idx in [6, 7]:
    q, ans = items[idx][0], items[idx][1]
    alt = items[idx][2:]
    text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        gen_ids, scores, fired, nm = rr.generate_one(model, tok, ids, probe, 1e9, "none", None)
        out = model.generate(ids, max_new_tokens=48, do_sample=False, use_cache=True, return_dict_in_generate=True)
    g2 = out.sequences[0][ids.shape[1]:]
    if end_id in g2:
        g2 = g2[:g2.tolist().index(end_id)]
    t1 = tok.decode(gen_ids, skip_special_tokens=True)
    t2 = tok.decode(g2, skip_special_tokens=True)
    print(f"=== {q}")
    print(f"  manual  : {t1[:80]!r}")
    print(f"  generate: {t2[:80]!r}")
    print(f"  same: {t1 == t2}")