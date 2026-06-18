import torch
import numpy as np
from bike.models import SimpleLeanHeadingModel
from bike.cem import CEMPlanner
from bike.dynamics import MAX_STEER

model = SimpleLeanHeadingModel()
model.load_state_dict(torch.load("meins_4/model_1/checkpoint.pt"))
model.eval()

planner = CEMPlanner(
    model,
    target_heading=0.0,
    horizon=50,
    loss_mode="lean_only",
    w_lean=10.0,
)
state = np.array([0.0, 0.0, -0.7, 0.0, 0.0, 0.0, -5.0, 0.0], dtype=np.float32)

action_pos = np.full((1, 50), 100.0, dtype=np.float32)
action_neg = np.full((1, 50), -100.0, dtype=np.float32)
action_zero = np.full((1, 50), 0.0, dtype=np.float32)

print("Cost of +100:", planner._rollout_costs(state, action_pos)[0])
print("Cost of 0:", planner._rollout_costs(state, action_zero)[0])
print("Cost of -100:", planner._rollout_costs(state, action_neg)[0])

