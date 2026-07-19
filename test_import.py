import sys, os
sys.path.insert(0, 'lewm_base')
sys.path.insert(0, 'code')

# Test 1: Import lewm_base modules
from module import ARPredictor, Embedder, MLP, SIGReg, Transformer, ConditionalBlock, Block
print("OK: lewm_base modules imported")

# Test 2: Import BTWM model
from btwm_model import build_model, BTWM, LeWM_Atari
print("OK: btwm_model imported")

# Test 3: Build model
import torch
model = build_model(variant='btwm', embed_dim=256, num_actions=6, history_size=4)
n_params = sum(p.numel() for p in model.parameters())
print(f"OK: BTWM built with {n_params:,} params")

# Test 4: Forward pass with dummy data
B, T = 4, 4
obs = torch.randn(B, T+1, 84, 84)
actions = torch.zeros(B, T, 6)
actions[:, 0, 0] = 1
result = model(obs, actions)
print(f"OK: Forward pass: fwd={result['fwd_loss'].item():.4f}, inv={result['inv_loss'].item():.4f}, sig={result['sigreg_loss'].item():.4f}")

# Test 5: LeWM variant
model2 = build_model(variant='no_inverse', embed_dim=256, num_actions=6, history_size=4)
result2 = model2(obs, actions)
print(f"OK: LeWM_Atari, fwd_loss={result2['fwd_loss'].item():.4f}")

print("\nAll sanity checks passed!")
