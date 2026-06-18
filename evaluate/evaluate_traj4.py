import torch
import numpy as np
from bike.models import SimpleLeanHeadingModel
from bike.cem import CEMPlanner, _clamp_differentials, _clamp_states

model = SimpleLeanHeadingModel()
model.load_state_dict(torch.load("meins_6/model_1/checkpoint.pt"))
model.eval()

state = np.array([[0.0, 0.0, -0.7, 0.0, 0.0, 0.0, -5.0, 0.0]], dtype=np.float32)

action_pos = torch.tensor([[100.0]], dtype=torch.float32)
action_neg = torch.tensor([[-100.0]], dtype=torch.float32)
a_zero = torch.tensor([[0.0]], dtype=torch.float32)

print("\n--- Evaluate NN ---")
d_pos = _clamp_differentials(model.predict_delta(torch.tensor(state, dtype=torch.float32), action_pos).detach().numpy())
d_neg = _clamp_differentials(model.predict_delta(torch.tensor(state, dtype=torch.float32), action_neg).detach().numpy())
d_zero = _clamp_differentials(model.predict_delta(torch.tensor(state, dtype=torch.float32), a_zero).detach().numpy())

print(f"Action +100 -> dlean_dot: {d_pos[0, 2]:.4f}")
print(f"Action -100 -> dlean_dot: {d_neg[0, 2]:.4f}")
print(f"Action    0 -> dlean_dot: {d_zero[0, 2]:.4f}")

