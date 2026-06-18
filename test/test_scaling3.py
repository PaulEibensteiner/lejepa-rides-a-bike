import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from bike.models import SimpleLeanHeadingModel, FullStateModel
from bike.dynamics import MAX_STEER
from train import collect_dataset, StickyGaussianTorquePolicy, ManualControllerPolicy, build_tensors
from bike.environment import BikeEnv
from bike.models import MODEL_DIFF_DT

env = BikeEnv(max_steps=1000, wind_std=0.0)
manual_policy = ManualControllerPolicy(desired_heading=0.0)
s, a, ns = collect_dataset(env, manual_policy, n_episodes=10, control_interval_steps=20)
s_t, a_t, d_t = build_tensors(s, a, ns, model_dt=MODEL_DIFF_DT)

# If we scale the action right inside the model wrapper
class ScalerWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x)

    def predict_delta(self, state, action):
        return self.m.predict_delta(state, action / 100.0)

model = SimpleLeanHeadingModel()
wrapped = ScalerWrapper(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(s_t, a_t, d_t[:, [2,3,6,7]]), batch_size=64, shuffle=True)

for epoch in range(10):
    for s_b, a_b, t_b in loader:
        optimizer.zero_grad()
        pred = wrapped.predict_delta(s_b, a_b)
        loss = torch.mean((pred - t_b)**2)
        loss.backward()
        optimizer.step()
        
print("ok")
