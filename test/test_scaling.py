import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from bike.models import SimpleLeanHeadingModel
from bike.dynamics import MAX_STEER
from train import collect_dataset, StickyGaussianTorquePolicy, ManualControllerPolicy, build_tensors
from bike.environment import BikeEnv
from bike.models import MODEL_DIFF_DT

env = BikeEnv(max_steps=1000, wind_std=0.0)
manual_policy = ManualControllerPolicy(desired_heading=0.0)
s, a, ns = collect_dataset(env, manual_policy, n_episodes=50, control_interval_steps=20)
s_t, a_t, d_t = build_tensors(s, a, ns, model_dt=MODEL_DIFF_DT)

target = d_t[:, [2, 3, 6, 7]]
variance = torch.var(target, dim=0)

model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(s_t, a_t, target), batch_size=64, shuffle=True)

for epoch in range(100):
    for s_b, a_b, t_b in loader:
        optimizer.zero_grad()
        a_scaled = a_b / 100.0
        lean_heading_dots = s_b[..., [2, 3, 6, 7]].clone()
        lean_heading_dots[..., 2:4] *= 0.02
        pred = model(torch.cat([a_scaled, lean_heading_dots], dim=-1))
        
        loss = torch.mean((pred - t_b)**2)
        loss.backward()
        optimizer.step()

pred = []
target_list = []
with torch.no_grad():
    for s_b, a_b, t_b in loader:
        a_scaled = a_b / 100.0
        lean_heading_dots = s_b[..., [2, 3, 6, 7]].clone()
        lean_heading_dots[..., 2:4] *= 0.02
        p = model(torch.cat([a_scaled, lean_heading_dots], dim=-1))
        pred.append(p)
        target_list.append(t_b)

pred = torch.cat(pred, dim=0)
target = torch.cat(target_list, dim=0)

mse = torch.mean((pred - target)**2, dim=0)
print(f"Bigger Model relative MSE with scaled action: {(mse / variance).numpy()}")

