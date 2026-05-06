#!/usr/bin/env python3
"""Compute PCA/SparsePCA/ICA MCC on synthetic (monotonic) with 5 seeds, get mean+/-std."""
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA, SparsePCA, FastICA
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts/data_prep/synthetic"))
from config import Z_DIM, REGIME_DRIVING_FACTORS, SUPPORT_SETS

GT = {
    0: {'name':'x',  'primary':{0,1,2,3},       'shared':{4}},
    1: {'name':'z1', 'primary':{5,6,7},          'shared':{8}},
    2: {'name':'z2', 'primary':{9,10,11,12},     'shared':{4}},
    3: {'name':'z3', 'primary':{13,14,15,16},    'shared':{8}},
    4: {'name':'u1', 'primary':{17,18,19,20},    'shared':{21}},
    5: {'name':'u2', 'primary':{22,23,24,25,26}, 'shared':{21}},
}
DRIVERS = set(REGIME_DRIVING_FACTORS)

data = np.load('data/synthetic_well_v2_mono/source/data.npz')
X = data['xt'][:, -1, :]
ct = data['ct'].squeeze()
raw = np.load('data/synthetic_well_v2_mono/raw/trajectories.npz', allow_pickle=True)
z_true = raw['z_trajs'][0][:5000]
x_obs_raw = np.array(raw['x_trajs'][0][:5000], dtype=np.float32)

def eval_one(name, model_factory, seed):
    model = model_factory(seed)
    z_full = model.fit_transform(X)
    z_probe = model.transform(x_obs_raw) if hasattr(model, 'transform') else z_full[:5000]

    # MCC
    corr = np.array([[abs(pearsonr(z_true[:,j], z_probe[:,k])[0])
                      if np.isfinite(pearsonr(z_true[:,j], z_probe[:,k])[0]) else 0
                      for k in range(6)] for j in range(Z_DIM)])
    ri, ci = linear_sum_assignment(-corr)
    mcc = corr[ri, ci].mean()
    mapping = dict(zip(ri.tolist(), ci.tolist()))
    inv_map = {v:k for k,v in mapping.items()}

    # Cohen's d
    m0, m1 = ct==0, ct==1
    d_vals = np.array([abs(z_full[m0,k].mean()-z_full[m1,k].mean()) /
                       np.sqrt((z_full[m0,k].var(ddof=1)+z_full[m1,k].var(ddof=1))/2+1e-10)
                       for k in range(6)])

    # Top-3 drivers
    top3 = np.argsort(d_vals)[-3:][::-1].tolist()
    matches = [inv_map.get(zi) for zi in top3]
    n_driver = sum(1 for m in matches if m is not None and m in DRIVERS)

    # Influence
    if hasattr(model, 'components_'): influence = np.abs(model.components_.T)
    elif hasattr(model, 'mixing_'): influence = np.abs(model.mixing_)
    else: influence = np.ones((X.shape[1], 6))

    # X_Z@top3
    xz = []
    for zi, mi in zip(top3, matches):
        if mi is None: continue
        mod = GT[mi]; valid = mod['primary'] | mod['shared']; k = len(mod['primary'])
        topk = np.argsort(influence[:, zi])[-k:][::-1].tolist()
        hits = sum(1 for o in topk if o in valid)
        xz.append(hits/k)
    xz_mean = np.mean(xz) if xz else 0
    return mcc, n_driver, xz_mean


SEEDS = [0, 42, 123, 456, 789]
methods = [
    ('PCA', lambda s: PCA(n_components=6, random_state=s)),
    ('SparsePCA_a10', lambda s: SparsePCA(n_components=6, alpha=10, max_iter=200, random_state=s)),
    ('ICA', lambda s: FastICA(n_components=6, random_state=s, max_iter=500)),
]

for name, factory in methods:
    mccs, zs, xzs = [], [], []
    for s in SEEDS:
        try:
            mcc, nd, xz = eval_one(name, factory, s)
            mccs.append(mcc); zs.append(nd); xzs.append(xz)
            print(f"  {name} seed={s}: MCC={mcc:.3f} Z@top3={nd}/3 X_Z@top3={xz:.3f}")
        except Exception as e:
            print(f"  {name} seed={s}: FAILED ({e})")
    if mccs:
        n_perfect = sum(1 for z in zs if z == 3)
        print(f"{name} mean+/-std: MCC={np.mean(mccs):.3f}+/-{np.std(mccs):.3f}  "
              f"Z@top3 perfect={n_perfect}/{len(zs)}  "
              f"X_Z@top3={np.mean(xzs):.3f}+/-{np.std(xzs):.3f}")
