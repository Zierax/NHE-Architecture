import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_rollback as rr
from collections import defaultdict
RES = rr.RES_DIR
WINDOW=5; SCALE=0.3
SEEDS=[1000,1001,1002,1003,1004,1006]

def tag(arm, seed, static=False):
    t=f"jump_gt_L19_t90_{arm}"
    if static:
        t+="_static0"
    if arm=="mask":
        t+=f"_sft{SCALE}"
    return f"{t}_s{seed}"

def run_bench(bench_name, arms, use_static=False):
    bench=json.load(open(os.path.join(RES, f"bench_{bench_name}.json")))["items"]
    subsets=defaultdict(list)
    for t,i in bench:
        subsets[t].append(i)
    print(f"=== bench {bench_name} subsets", {k:len(v) for k,v in subsets.items()})
    for topic, idxs in subsets.items():
        for arm in arms:
            for seed in SEEDS:
                base_tag=tag(arm, seed, static=use_static)
                suffix = "_rand" if bench_name=="random" else ""
                expected = os.path.join(RES, f"eval_runtime_{topic}_{base_tag}{suffix}.json")
                if os.path.exists(expected):
                    print(f"skip {expected}")
                    continue
                det=json.load(open(os.path.join(RES,"detector_greedy.json")))["detector_early"]
                det["threshold"]=det["threshold_t90"]; det["threshold_key"]="t90"
                det["mode"]=arm; det["window"]=WINDOW; det["scale"]=SCALE if arm=="mask" else 0.0
                det["sample"]=True; det["seed"]=seed
                det["subset"]=idxs
                if bench_name=="random":
                    det["bench_suffix"]="_rand"
                if use_static:
                    det["static_mask"]=os.path.join(RES,"mask_k32_midwrong.json")
                    det["static_scale"]=0.0
                print(f"RUN bench={bench_name} topic={topic} arm={arm} seed={seed} static={use_static} items={len(idxs)}")
                rr.run_topic(topic, det)

if __name__=="__main__":
    t0=time.time()
    # 1) random bench sampled (none/mask/abstain)
    run_bench("random", ["none","mask","abstain"], use_static=False)
    # 2) hard bench merged (static+mask and static+abstain)
    # For merged, we use hard bench with static k32 + temporal mask/abstain
    # Run both merged variants
    for arm in ["mask","abstain"]:
        run_bench("hard", [arm], use_static=True)
    print(f"DONE total elapsed {time.time()-t0:.0f}s")
