import numpy as np
import torch
from bike.models import SimpleLeanHeadingModel
from bike.cem import CEMPlanner

model = SimpleLeanHeadingModel()
model.load_state_dict(torch.load("meins_6/model_1/checkpoint.pt"))
model.eval()

planner = CEMPlanner(
    model,
    target_heading=0.0,
    horizon=10,
    n_samples=500,
    n_iters=5,
    min_std=10.0,
    loss_mode="lean_only",
)

state = np.array([0.0, 0.0, -0.7, 0.0, 0.0, 0.0, -5.0, 0.0], dtype=np.float32)

action = planner.plan(state)
print("Chosen action:", action[0])

# Cost of a safe trajectory (+100)
a_pos = np.full((1, 10), 100.0, dtype=np.float32)
a_neg = np.full((1, 10), -100.0, dtype=np.float32)
print("Cost of +100 array:", planner._rollout_costs(state, a_pos)[0])
print("Cost of -100 array:", planner._rollout_costs(state, a_neg)[0])

