# MOSAIC

**Mo**dule discovery via **S**parse **A**dditive **I**dentifiable **C**ausal learning for scientific time series.

MOSAIC is a sparse temporal VAE that combines temporal causal-representation
identifiability with support recovery over named observed variables. Stage 1
identifies latent factors via a regime-conditioned temporal prior; Stage 2
freezes the encoder/prior and recovers each latent's main-effect support
through an additive decoder with column-entropy sparsity.

The result is **module-level interpretability**: each identifiable latent is
anchored to a small subset of named scientific observations (residue-pair
distances, climate grid cells, plasma diagnostics, …).

---

## Repository layout

```
mosaic/                 # the importable Python package
├── models/             # MOSAIC LightningModule (TimeVaryingProcess_v2)
├── components/         # encoder, additive decoder, parallel transition prior
├── data/               # MDRegimeDataset (xt, yt, ct windowed loader)
├── metrics/            # MCC, Hungarian matching
└── utils/

scripts/
├── data_prep/          # build data.npz for each domain
│   ├── synthetic/      # double-well Langevin benchmark
│   ├── prepare_rna.py
│   ├── prepare_omni.py
│   ├── prepare_climate.py
│   ├── prepare_tep.py
│   └── prepare_disruption.py
├── train/
│   ├── train_phase1.py # Stage 1 (dense decoder + temporal prior)
│   ├── train_phase2.py # Stage 2 (frozen encoder + additive sparse decoder)
│   ├── train_joint.py  # ablation: single-stage joint training
│   └── run_crossseed.py# multi-seed runner
└── eval/
    ├── eval_unified.py # MCC, Z@top3, X_Z@top3, regime accuracy
    ├── eval_baselines_5seeds.py
    └── ...

baselines/
├── tdrl_lily/          # TDRL, iVAE, SlowVAE, β-VAE (LiLY upstream)
├── ctrlns/             # CtrlNS upstream
├── spib/               # State-Predictive Information Bottleneck upstream
└── linear/             # Sparse PCA, ICA, PCA

examples/
└── quickstart_synthetic.sh   # end-to-end smoke test (~10 min on one A40)

configs/                 # YAML configs (per dataset hyper-params)
docs/                    # checklist.md and supporting notes
install.sh               # one-click env setup
environment.yml          # conda spec
requirements.txt         # pip spec
setup.py                 # pip install -e .
```

---

## Quickstart (one-click)

On a fresh machine with a recent conda (or just `python3` + `pip`):

```bash
cd mosaic
bash install.sh                       # creates conda env "mosaic", installs torch + mosaic
conda activate mosaic                 # (or `source .venv-mosaic/bin/activate` if no conda)
bash examples/quickstart_synthetic.sh # generates data, trains 2 stages, evaluates
```

`install.sh` auto-detects whether you have an NVIDIA GPU and installs the
matching PyTorch wheel. Override the CUDA version with `CUDA=cu118 bash
install.sh` if needed; pass `--no-extras` to skip MD/NetCDF/JAX extras.

`quickstart_synthetic.sh` runs the paper-canonical synthetic config end-to-end
(z=8, hidden=128, hpz_p1=32 / hpz_p2=4, λ=50, β=0.002, γ=0.02, 80 + 80 epochs)
on a single seed.

---

## Reproducing the paper experiments

Hyperparameters per benchmark are stored in
[`scripts/train/run_crossseed.py`](scripts/train/run_crossseed.py).

### 1. Synthetic double-well benchmark

```bash
# Generate data once
python scripts/data_prep/synthetic/generate_data.py
python scripts/data_prep/synthetic/prepare.py

# 5 seeds, MOSAIC two-stage
python scripts/train/run_crossseed.py --benchmark synthetic --seeds 0 42 123 456 789
```

Reports `MCC`, `Z@top3`, `X_Z@top3 (gate 0.50)` matching Table 1 in the paper.

### 2. RNA molecular dynamics (cUUCGg tetraloop)

The RNA experiments use 14 OpenMM trajectories of the cUUCGg tetraloop
(`GGCACUUCGGUGCC`) at 345 K, 400 K, and 500 K. The MD inputs we used
(`equil.xml`, `system.xml`, `hairpin_*.pdb`) are not bundled in this
repo because the trajectory files (~500 MB) are too large for git;
the simulation protocol is described in the paper appendix. Once the
trajectories are placed under `data/RNA/RNA_simulations_replicas/`
(replica directories `Rep2`–`Rep10`), the rest of the pipeline is:

```bash
python scripts/data_prep/prepare_rna.py \
    --features per_residue_dist \
    --output_dir data/RNA/prepared_frozendist_v4_cpQ
python scripts/train/run_crossseed.py --benchmark rna --seeds 0 42 123 456 789
```

### 3. Cross-domain (OMNI / ENSO / TEP / tokamak disruption)

Each dataset is fetched from its official source and placed under
`data/<domain>/raw/`; the matching `prepare_*.py` then writes
`data/<domain>/prepared/data.npz`.

| Dataset         | Source                                                                                    |
|-----------------|-------------------------------------------------------------------------------------------|
| OMNI solar wind | NASA OMNIWeb — <https://omniweb.gsfc.nasa.gov/ow.html>                                    |
| ENSO climate    | NOAA ERSSTv5 — <https://www.ncei.noaa.gov/products/extended-reconstructed-sst>; ONI index <https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt> |
| TEP             | Tennessee Eastman Process (Braatz repackage) — <http://web.mit.edu/braatzgroup/links.html>|
| Disruption      | Synthetic, generated directly by `scripts/data_prep/prepare_disruption.py`                |

Once the raw files are in place:

```bash
for d in omni climate tep disruption; do
  python scripts/train/run_crossseed.py --benchmark $d --seeds 0 42 123
done
```

### 4. Baselines

| Family       | What                                              | Where                                 |
|--------------|---------------------------------------------------|---------------------------------------|
| Temporal CRL | TDRL, iVAE, SlowVAE, β-VAE (LiLY)                 | `baselines/tdrl_lily/scripts/`        |
| Temporal CRL | CtrlNS (auto regime discovery)                    | `baselines/ctrlns/`                   |
| IB-style    | SPIB                                               | `baselines/spib/`                     |
| Linear      | Sparse PCA, ICA, PCA                               | `baselines/linear/run_*.py`           |

Each baseline directory has its own README from the upstream authors. The
adapters in `scripts/eval/eval_baselines_5seeds.py` and
`baselines/spib/spib_adapter.py` wrap them onto the standard
`(xt, yt, ct)` interface so all methods are scored with the same
`scripts/eval/eval_unified.py` pipeline.

### 5. Evaluation

`scripts/eval/eval_unified.py` is the single source of truth for the
metrics in the paper:

| Metric        | Meaning                                                                          |
|---------------|----------------------------------------------------------------------------------|
| `MCC`         | Hungarian-matched mean correlation between learned and true latents (synthetic) |
| `Z@top3`      | Whether the 3 most regime-discriminative latents map to true regime-varying ones |
| `X_Z@top3`    | Support precision of those 3 latents (gated at top-3 mass ≥ 0.50)               |
| `regime_acc`  | Logistic-regression accuracy of `c_t` from `z_t`                                 |

```bash
python scripts/eval/eval_unified.py \
    --benchmark synthetic \
    --ckpt_glob 'experiments/crossseed/synthetic/seed_*/phase2/lightning_logs/version_*/checkpoints/best-*.ckpt' \
    --data_dir data/synthetic_well_v2/source \
    --out experiments/crossseed/synthetic/eval_summary
```

---

## License

MOSAIC code is released under the MIT License.
The bundled upstream baselines (`baselines/tdrl_lily`, `baselines/ctrlns`,
`baselines/spib`) retain their original licenses; see each subdirectory's
`LICENSE` / `README.md`.
