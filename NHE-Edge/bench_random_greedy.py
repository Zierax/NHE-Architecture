import json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_rollback as rr
RES = rr.RES_DIR
bench = json.load(open(os.path.join(RES, "bench_random.json")) )["items"]
# group by topic
from collections import defaultdict
subsets = defaultdict(list)
for t,i in bench:
    subsets[t].append(i)
print("bench_random subsets", {k: len(v) for k,v in subsets.items()})
for topic, idxs in subsets.items():
    for arm, scale in [("none",0.0), ("mask",0.3), ("abstain",0.0)]:
        det = json.load(open(os.path.join(RES, "detector_greedy.json")))["detector_early"]
        det["threshold"] = det["threshold_t90"]; det["threshold_key"]="t90"
        det["mode"]=arm; det["window"]=5; det["scale"]=scale; det["sample"]=False; det["seed"]=0
        det["subset"]=idxs
        # skip if exists
        tag = f"jump_gt_L19_t90_{arm}" + (f"_sft{scale}" if scale!=0 else "") 
        out = os.path.join(RES, f"eval_runtime_{topic}_{tag}.json")
        # Note: runtime writes with topic_ prefix, but bench_random aggregates across topics;
        # we need per-topic files; bench_random will be aggregated in analysis via strict scoring across topics.
        # To avoid overwriting hard bench files, use bench_random tag suffix
        # Actually we will write to eval_runtime_{topic}_{tag}_rand.json to keep separate
        tag_rand = tag + "_rand"
        out_rand = os.path.join(RES, f"eval_runtime_{topic}_{tag_rand}.json")
        if os.path.exists(out_rand):
            print(f"skip {out_rand}")
            continue
        print(f"RUN {topic} {arm} {len(idxs)} items")
        rr.run_topic(topic, det)
        # rename to _rand to not overwrite
        # run_topic wrote to eval_runtime_{topic}_{tag}.json, rename it
        if os.path.exists(out):
            os.rename(out, out_rand)
            print(f"renamed {out} -> {out_rand}")
