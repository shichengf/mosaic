#!/usr/bin/env python3
"""Benchmark v1 vs v2 transition prior: speedup and memory."""
import torch, time
from mosaic.components.transition import NPChangeTransitionPrior
from mosaic.components.transition_v2 import NPChangeTransitionPrior_v2

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}')

configs = [
    ('synthetic', 8, 256, 3, 128),
    ('rna', 4, 256, 3, 128),
    ('cross-domain (climate)', 8, 256, 3, 128),
    ('large stress (Z=64)', 64, 256, 3, 64),
]

print(f"\n{'Config':<30s} {'v1 (ms)':>10s} {'v2 (ms)':>10s} {'Speedup':>10s} {'Mem v1 (MB)':>12s} {'Mem v2 (MB)':>12s}")
print("-" * 90)

results = []
for name, Z, B, T, H in configs:
    L = 2

    v1 = NPChangeTransitionPrior(lags=L, latent_size=Z, embedding_dim=8,
                                  num_layers=3, hidden_dim=H).to(device)
    v2 = NPChangeTransitionPrior_v2(lags=L, latent_size=Z, embedding_dim=8,
                                    num_layers=3, hidden_dim=H).to(device)
    v1.train(); v2.train()

    x = torch.randn(B, T, Z, device=device, requires_grad=True)
    emb = torch.randn(B, 8, device=device)

    # Warmup
    for _ in range(5):
        r1, lad1 = v1(x, emb); (r1.sum() + lad1.sum()).backward()
        r2, lad2 = v2(x, emb); (r2.sum() + lad2.sum()).backward()
    if device == 'cuda': torch.cuda.synchronize()

    # Benchmark v1
    n_iter = 30
    t0 = time.time()
    for _ in range(n_iter):
        r1, lad1 = v1(x, emb)
        (r1.sum() + lad1.sum()).backward()
    if device == 'cuda': torch.cuda.synchronize()
    t_v1 = (time.time() - t0) / n_iter * 1000

    # Benchmark v2
    t0 = time.time()
    for _ in range(n_iter):
        r2, lad2 = v2(x, emb)
        (r2.sum() + lad2.sum()).backward()
    if device == 'cuda': torch.cuda.synchronize()
    t_v2 = (time.time() - t0) / n_iter * 1000

    # Memory
    mem_v1 = mem_v2 = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            r1, lad1 = v1(x, emb); (r1.sum() + lad1.sum()).backward()
        mem_v1 = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            r2, lad2 = v2(x, emb); (r2.sum() + lad2.sum()).backward()
        mem_v2 = torch.cuda.max_memory_allocated() / 1024**2

    label = f"{name} (Z={Z},B={B})"
    print(f"{label:<30s} {t_v1:>10.1f} {t_v2:>10.1f} {t_v1/t_v2:>9.1f}x {mem_v1:>12.0f} {mem_v2:>12.0f}")
    results.append((name, Z, B, t_v1, t_v2, t_v1/t_v2, mem_v1, mem_v2))

# Verify numerical equivalence
print("\n=== Numerical equivalence check (same init weights) ===")
Z, B, T, H = 8, 32, 3, 64
L = 2
v1 = NPChangeTransitionPrior(lags=L, latent_size=Z, embedding_dim=8, num_layers=3, hidden_dim=H).to(device)
v2 = NPChangeTransitionPrior_v2(lags=L, latent_size=Z, embedding_dim=8, num_layers=3, hidden_dim=H).to(device)

# Load v1 weights into v2
try:
    v1_state = {'transition_prior.' + k: v for k, v in v1.state_dict().items()}
    n_loaded = v2.load_v1_weights(v1_state)
    print(f"  Loaded {n_loaded} weight tensors from v1 into v2")
    x = torch.randn(B, T, Z, device=device)
    emb = torch.randn(B, 8, device=device)
    v1.eval(); v2.eval()
    with torch.no_grad():
        r1, lad1 = v1(x, emb)
        r2, lad2 = v2(x, emb)
    print(f"  Residuals close: {torch.allclose(r1, r2, atol=1e-3)} (max diff: {(r1-r2).abs().max().item():.2e})")
    print(f"  LogAbsDet close: {torch.allclose(lad1, lad2, atol=1e-3)} (max diff: {(lad1-lad2).abs().max().item():.2e})")
except Exception as e:
    print(f"  Weight loading failed: {e}")

print("\nDone!")
