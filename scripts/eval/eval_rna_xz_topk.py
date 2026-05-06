#!/usr/bin/env python3
"""Compute X_Z@top_k on RNA for Ours (Phase 2, 5 seeds) + TDRL + CtrlNS.

For RNA:
- "Driver z" = argmax Cohen's d over learned z dims
- "Ground truth driving module" = Loop (expected to unfold first in cp_Q regime)
- Loop module = 8 obs {5,6,7,8,19,20,21,22}
- X_Z@top_k = fraction of driver z's top-k obs in Loop (k = 8 = |Loop|)
"""
import sys, os, numpy as np, torch, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    REPO_ROOT,
    REPO_ROOT / 'scripts/eval',
    REPO_ROOT / 'baselines/tdrl_lily',
    REPO_ROOT / 'baselines/ctrlns',
    REPO_ROOT / 'baselines/ctrlns/datasets',
    REPO_ROOT / 'baselines/ctrlns/models',
):
    sys.path.insert(0, str(_p))

# Patch lightning
import pytorch_lightning as _pl
sys.modules['lightning'] = type(sys)('lightning')
sys.modules['lightning.pytorch'] = _pl

RNA_MODULES = {
    'Loop':        [5,6,7,8,19,20,21,22],
    'ClosingPair': [4,9,18,23],
    'InnerStem':   [2,3,10,11,16,17,24,25],
    'OuterStem':   [0,1,12,13,14,15,26,27],
}
LOOP = set(RNA_MODULES['Loop'])

# Load RNA data
data = np.load(REPO_ROOT / 'data/RNA/prepared_frozendist_v4_cpQ/data.npz')
X_last = data['xt'][:, -1, :].astype(np.float32)
ct = data['ct'].squeeze().astype(int)


def eval_model(encode_fn, decode_fn, z_dim, label, is_additive=False):
    """Compute all RNA metrics for one model."""
    # Encode
    with torch.no_grad():
        mus = []
        for i in range(0, len(X_last), 4096):
            out = encode_fn(torch.tensor(X_last[i:i+4096]))
            mus.append(out.cpu().numpy() if torch.is_tensor(out) else out)
    z_full = np.concatenate(mus)

    # Regime acc
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(z_full, ct); acc = clf.score(z_full, ct)

    # Cohen's d per z
    m0, m1 = ct == 0, ct == 1
    d_vals = np.array([abs(z_full[m0,k].mean() - z_full[m1,k].mean()) /
                       np.sqrt((z_full[m0,k].var(ddof=1) + z_full[m1,k].var(ddof=1))/2 + 1e-10)
                       for k in range(z_full.shape[1])])
    driver_z = int(np.argmax(d_vals))

    # Influence via probe +/-1
    with torch.no_grad():
        zz = torch.zeros(1, z_dim)
        inf_cols = []
        for j in range(z_dim):
            zp = zz.clone(); zp[0,j] = 1.0
            zn = zz.clone(); zn[0,j] = -1.0
            x_p = decode_fn(zp); x_n = decode_fn(zn)
            inf_cols.append((x_p - x_n).abs().squeeze().cpu().numpy())
    influence = np.stack(inf_cols, axis=1)
    inf_driver = influence[:, driver_z]

    # Module-level metrics
    mod_inf = {m: float(inf_driver[obs].sum()) for m, obs in RNA_MODULES.items()}
    total = sum(mod_inf.values())
    mod_pct = {k: v/total*100 for k, v in mod_inf.items()}
    loop_pct = mod_pct['Loop']
    loop_rank = 1 + sum(1 for v in mod_pct.values() if v > loop_pct)

    # X_Z@top_k (k = |Loop| = 8) — variable-level precision
    k = len(LOOP)
    top_k_obs = np.argsort(inf_driver)[-k:][::-1].tolist()
    hits = sum(1 for o in top_k_obs if o in LOOP)
    x_z_top_k = hits / k

    # Also X_Z@top_4 for stricter evaluation (just Loop residue μ)
    k4 = 4
    top_4_obs = np.argsort(inf_driver)[-k4:][::-1].tolist()
    hits4 = sum(1 for o in top_4_obs if o in LOOP)
    x_z_top_4 = hits4 / k4

    result = {
        'method': label, 'acc': float(acc), 'driver_z': driver_z, 'driver_d': float(d_vals[driver_z]),
        'loop_pct': float(loop_pct), 'loop_rank': int(loop_rank),
        'x_z_top_8': float(x_z_top_k), 'x_z_top_4': float(x_z_top_4),
        'top_8_obs': top_k_obs, 'top_4_obs': top_4_obs,
    }
    print(f"\n{label}:")
    print(f"  Acc={acc:.3f}  Driver d={d_vals[driver_z]:.2f} (z={driver_z})")
    print(f"  Loop%={loop_pct:.1f}%  Rank=#{loop_rank}")
    print(f"  X_Z@top8 (k=|Loop|) = {x_z_top_k:.3f}  top-8: {top_k_obs}")
    print(f"  X_Z@top4 (strict)   = {x_z_top_4:.3f}  top-4: {top_4_obs}")
    return result


results = []

# ─── Ours: Phase 2 (5 seeds) ───
from mosaic.models.mosaic import TimeVaryingProcess_v2
for seed in [0, 42, 123, 456, 789]:
    ckpt = sorted(REPO_ROOT.glob(
        f'experiments/final/rna/seed_{seed}/phase2/lightning_logs/version_*/checkpoints/best-*.ckpt'))
    if not ckpt: continue
    ck = torch.load(str(ckpt[-1]), map_location='cpu', weights_only=False)
    m = TimeVaryingProcess_v2(**ck['hyper_parameters'])
    m.load_state_dict(ck['state_dict']); m.eval()
    def enc(x, m=m): _, mu, _, _ = m.net(x); return mu
    def dec(z, m=m): return m.net.decoder(z)
    results.append(eval_model(enc, dec, m.z_dim, f"Ours Phase 2 (seed {seed})"))

# Aggregate ours
ours_rows = [r for r in results if r['method'].startswith('Ours')]
if ours_rows:
    print(f"\n--- Ours (5-seed mean) ---")
    for key in ['acc', 'driver_d', 'loop_pct', 'x_z_top_8', 'x_z_top_4']:
        vals = [r[key] for r in ours_rows]
        print(f"  {key}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")
    print(f"  loop_rank: {[r['loop_rank'] for r in ours_rows]}")

# ─── TDRL ───
from LiLY.modules.change import TimeVaryingProcess
cfg = dict(input_dim=28, length=3, z_dim=4, lag=2, nclass=2, hidden_dim=128,
           embedding_dim=2, trans_prior='NP', infer_mode='F',
           beta=2e-3, gamma=2e-2, decoder_dist='gaussian')
m_tdrl = TimeVaryingProcess(**cfg)
_tdrl_ckpts = sorted(REPO_ROOT.glob(
    'baselines/results/tdrl_lily/rna/*/lightning_logs/version_*/checkpoints/*.ckpt'))
if not _tdrl_ckpts:
    raise FileNotFoundError(
        'No TDRL RNA checkpoints under baselines/results/tdrl_lily/rna/*/lightning_logs/...')
ck = torch.load(str(_tdrl_ckpts[-1]), map_location='cpu', weights_only=False)
m_tdrl.load_state_dict(ck['state_dict'], strict=False); m_tdrl.eval()
def tdrl_enc(x): _, mu, _, _ = m_tdrl.net(x); return mu
def tdrl_dec(z): return m_tdrl.net.decoder(z)
results.append(eval_model(tdrl_enc, tdrl_dec, 4, "TDRL"))

# ─── CtrlNS ───
import importlib.util
_run_syn = REPO_ROOT / 'baselines/ctrlns/run_synthetic.py'
spec = importlib.util.spec_from_file_location("run_synthetic", str(_run_syn))
_mod = importlib.util.module_from_spec(spec)
exec(open(spec.origin, encoding='utf-8').read(), _mod.__dict__)
SparseX = _mod.SparseX
m_cn = SparseX(n_class=2, x_dim=28, z_dim=4, lags=2, hidden_dim=128, embedding_dim=2,
               alpha=0.02, beta=0.002, gamma=0.02, lr=5e-4, weight_decay=1e-4, correlation='Pearson')
_ctrl_ckpts = sorted(REPO_ROOT.glob(
    'baseline/results/ctrlns_rna/lightning_logs/version_*/checkpoints/*.ckpt'))
if not _ctrl_ckpts:
    raise FileNotFoundError('No CtrlNS RNA checkpoints under baseline/results/ctrlns_rna/...')
ck2 = torch.load(str(_ctrl_ckpts[-1]), map_location='cpu', weights_only=False)
sd = m_cn.state_dict()
compat = {k: v for k, v in ck2['state_dict'].items() if k in sd and sd[k].shape == v.shape}
print(f"\nCtrlNS: loading {len(compat)}/{len(ck2['state_dict'])} compatible keys")
m_cn.load_state_dict(compat, strict=False); m_cn.eval()
def cn_enc(x): _, mu, _, _ = m_cn.net(x); return mu
def cn_dec(z): return m_cn.net.decoder(z)
results.append(eval_model(cn_enc, cn_dec, 4, "CtrlNS"))

# ─── Summary Table ───
print("\n" + "="*85)
print(f"{'Method':<25s} {'Acc':>6s} {'Top d':>7s} {'Loop%':>7s} {'Rank':>5s} {'X_Z@8':>7s} {'X_Z@4':>7s}")
print("="*85)
for r in results:
    print(f"{r['method']:<25s} {r['acc']:>6.3f} {r['driver_d']:>7.2f} {r['loop_pct']:>6.1f}% {r['loop_rank']:>5d} "
          f"{r['x_z_top_8']:>7.3f} {r['x_z_top_4']:>7.3f}")

_out_json = REPO_ROOT / 'experiments/final/supp_results/rna_xz_topk.json'
_out_json.parent.mkdir(parents=True, exist_ok=True)
_out_json.write_text(json.dumps(results, indent=2), encoding='utf-8')
print("\nSaved to experiments/final/supp_results/rna_xz_topk.json")
