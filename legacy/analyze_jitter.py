import os
import sys
import json
import pickle
import numpy as np

SEED = 0

DATA_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else ""

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data", DATA_SUBDIR) if DATA_SUBDIR else os.path.join(BASE, "data")
RES_DIR = os.path.join(BASE, "results")
os.makedirs(RES_DIR, exist_ok=True)
TAG = DATA_SUBDIR or "general"

N_LAYERS = 27
HIDDEN = 1152

records = []
corrupt = 0
for f in sorted(os.listdir(OUT_DIR)):
    if not f.startswith("ex_") or not f.endswith(".pkl"):
        continue
    p = os.path.join(OUT_DIR, f)
    try:
        with open(p, "rb") as fh:
            rec = pickle.load(fh)
        rec["_file"] = f
        records.append(rec)
    except Exception as e:
        corrupt += 1
        print(f"corrupt {f}: {e}")

y = np.array([r["label"] for r in records], dtype=int)
n = len(y)
print(f"examples={n} truthful={int(y.sum())} hallucinated={int((1 - y).sum())} corrupt={corrupt}")

flows = [r["flow"].float().numpy() for r in records]
T = np.array([f.shape[0] for f in flows])
print(f"gen tokens: min={T.min()} max={T.max()} mean={T.mean():.1f}")

X_last = np.stack([f[-1] for f in flows]).astype(np.float32)
X_mean = np.stack([f.mean(axis=0) for f in flows]).astype(np.float32)

def layer_auroc(X, label="last"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aurocs = []
    for l in range(N_LAYERS):
        Xl = X[:, l, :]
        preds = np.zeros(n)
        for tr, te in skf.split(Xl, y):
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            clf.fit(Xl[tr], y[tr])
            preds[te] = clf.predict_proba(Xl[te])[:, 1]
        aurocs.append(roc_auc_score(y, preds))
    return np.array(aurocs)

auc_last = layer_auroc(X_last)
auc_mean = layer_auroc(X_mean)

def time_features(flow):
    feats = []
    for l in range(N_LAYERS):
        h = flow[:, l, :]
        norms = np.linalg.norm(h, axis=1)
        if len(h) >= 2:
            d = h[1:] - h[:-1]
            jumps = np.linalg.norm(d, axis=1)
            cos = np.sum(h[1:] * h[:-1], axis=1) / (np.linalg.norm(h[1:], axis=1) * np.linalg.norm(h[:-1], axis=1) + 1e-9)
            feats.extend([
                norms.mean(), norms.std(), norms.max(), norms[-1] - norms[0],
                jumps.mean(), jumps.std(), jumps.max(),
                cos.mean(), cos.std(), cos.min(),
            ])
        else:
            feats.extend([norms.mean(), norms.std(), norms.max(), 0.0] + [0.0] * 6)
    return np.array(feats, dtype=np.float32)

X_time = np.stack([time_features(f) for f in flows])
FEAT_NAMES = []
for l in range(N_LAYERS):
    for name in ["norm_mean", "norm_std", "norm_max", "norm_delta",
                 "jump_mean", "jump_std", "jump_max",
                 "cos_mean", "cos_std", "cos_min"]:
        FEAT_NAMES.append(f"L{l:02d}_{name}")

def scalar_auroc(vals):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(vals)) < 2:
        return 0.5
    a = roc_auc_score(y, vals)
    b = roc_auc_score(y, -vals)
    return max(a, b)

time_auc = np.array([scalar_auroc(X_time[:, i]) for i in range(X_time.shape[1])])

depth_jump = np.stack([np.linalg.norm(f[-1, l+1] - f[-1, l]) for f in flows for l in range(N_LAYERS - 1)])
depth_jump = depth_jump.reshape(n, N_LAYERS - 1).astype(np.float32)
depth_auc = np.array([scalar_auroc(depth_jump[:, l]) for l in range(N_LAYERS - 1)])

report = {
    "seed": SEED,
    "n_examples": n,
    "n_truthful": int(y.sum()),
    "n_hallucinated": int((1 - y).sum()),
    "per_layer_auroc_last_token": [float(x) for x in auc_last],
    "per_layer_auroc_mean_pool": [float(x) for x in auc_mean],
    "per_layer_auroc_depth_jump": [float(x) for x in depth_auc],
    "time_feature_auroc": {FEAT_NAMES[i]: float(time_auc[i]) for i in np.argsort(-time_auc)[:20]},
}

print("\nlayer | last-token AUROC | mean-pool AUROC | depth-jump AUROC")
for l in range(N_LAYERS):
    dj = depth_auc[l] if l < N_LAYERS - 1 else float("nan")
    print(f"  {l:>2}  |     {auc_last[l]:.3f}        |     {auc_mean[l]:.3f}      |     {dj:.3f}")

best_last = int(np.argmax(auc_last))
best_mean = int(np.argmax(auc_mean))
print(f"\npeak last-token layer: {best_last} (AUROC {auc_last[best_last]:.3f})")
print(f"peak mean-pool layer: {best_mean} (AUROC {auc_mean[best_mean]:.3f})")
print(f"peak depth-jump layer: {int(np.argmax(depth_auc))} (AUROC {depth_auc.max():.3f})")

print("\ntop time-jitter features:")
for i in np.argsort(-time_auc)[:10]:
    print(f"  {FEAT_NAMES[i]:<18} AUROC={time_auc[i]:.3f}")

with open(os.path.join(RES_DIR, f"jitter_report_{TAG}.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
print(f"\nsaved results/jitter_report_{TAG}.json")