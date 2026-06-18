import torch
import numpy as np
from bike.models import SimpleLeanHeadingModel
from bike.dynamics import MAX_STEER

model = SimpleLeanHeadingModel()
model.load_state_dict(torch.load("meins_4/model_1/checkpoint.pt"))
model.eval()

# Let's test the response to different actions given an initial state
state = torch.tensor([[0.0, 0.0, -0.7, 0.0, 0.0, 0.0, -5.0, 0.0]], dtype=torch.float32)

print("Testing Action = +100")
a_pos = torch.tensor([[100.0]], dtype=torch.float32)
pred_pos = model.predict_delta(state, a_pos).detach().numpy()[0]
print(f"Δlean: {pred_pos[0]:.4f}, Δheading: {pred_pos[1]:.4f}, Δlean_dot: {pred_pos[2]:.4f}, Δheading_dot: {pred_pos[3]:.4f}")

print("\nTesting Action = -100")
a_neg = torch.tensor([[-100.0]], dtype=torch.float32)
pred_neg = model.predict_delta(state, a_neg).detach().numpy()[0]
print(f"Δlean: {pred_neg[0]:.4f}, Δheading: {pred_neg[1]:.4f}, Δlean_dot: {pred_neg[2]:.4f}, Δheading_dot: {pred_neg[3]:.4f}")

