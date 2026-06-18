import torch
import numpy as np
from bike.models import SimpleLeanHeadingModel
from bike.dynamics import MAX_STEER
from train import collect_dataset, StickyGaussianTorquePolicy, ManualControllerPolicy, build_tensors
from bike.environment import BikeEnv
from bike.models import MODEL_DIFF_DT

env = BikeEnv(max_steps=1000, wind_std=0.0)
manual_policy = ManualControllerPolicy(desired_heading=0.0)
s, a, ns = collect_dataset(env, manual_policy, n_episodes=5, control_interval_steps=20)
s_t, a_t, d_t = build_tensors(s, a, ns, model_dt=MODEL_DIFF_DT)

target = d_t[:, [2, 3, 6, 7]]
variance = torch.var(target, dim=0)
print(f"Target Variances (lean, heading, l_dot, h_dot): {variance.numpy()}")

model = SimpleLeanHeadingModel()
model.load_state_dict(torch.load("meins_6/model_1/checkpoint.pt"))
model.eval()

pred = model.predict_delta(s_t, a_t).detach()
pred_variance = torch.var(pred, dim=0)
print(f"Prediction Variances: {pred_variance.numpy()}")

mse = torch.mean((pred - target)**2, dim=0)
print(f"Prediction MSE by component: {mse.numpy()}")

print(f"Relative MSE (MSE / Variance): {(mse / variance).numpy()}")

