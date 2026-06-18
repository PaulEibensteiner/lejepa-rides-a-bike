import numpy as np
import torch
from bike.models import SimpleLeanHeadingModel
from bike.cem import CEMPlanner

model = SimpleLeanHeadingModel()
model.load_state_dict(torch.load("meins_6/model_1/checkpoint.pt"))
model.eval()

planner = CEMPlanner(
    model, target_heading=0.0, horizon=50, n_samples=200, n_iters=5, min_std=10.0, loss_mode="lean_only"
)

state = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

action = planner.plan(state)
print("Initial Chosen action:", action[0])

# Cost of a safe trajectory (+100) vs (-100) vs 0 from near 0
for a_val in [100.0, -100.0, 0.0]:
    a_seq = np.full((1, 50), a_val, dtype=np.float32)
    print(f"Cost of {a_val} array:", planner._rollout_costs(state, a_seq)[0])

