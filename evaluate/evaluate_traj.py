import torch
import numpy as np
from bike.models import SimpleLeanHeadingModel
from bike.cem import CEMPlanner, _clamp_differentials, _clamp_states

model = SimpleLeanHeadingModel()
model.load_state_dict(torch.load("meins_4/model_1/checkpoint.pt"))
model.eval()

def rollout_print(action_val):
    state = np.array([[0.0, 0.0, -0.7, 0.0, 0.0, 0.0, -5.0, 0.0]], dtype=np.float32)
    print(f"\nEvaluating action {action_val}:")
    for t in range(5):
        a_t = torch.tensor([[action_val]], dtype=torch.float32)
        state_t = torch.tensor(state, dtype=torch.float32)
        delta_t = model.predict_delta(state_t, a_t)
        delta_np = _clamp_differentials(delta_t.detach().numpy() * (0.02 / 0.02))
        
        state[:, 2] += delta_np[:, 0]
        state[:, 3] += delta_np[:, 1]
        state[:, 6] += delta_np[:, 2]
        state[:, 7] += delta_np[:, 3]
        state = _clamp_states(state)
        
        print(f"t={t+1} | lean: {state[0, 2]:.4f}, lean_dot: {state[0, 6]:.4f} | dlean: {delta_np[0, 0]:.4f}, dlean_dot: {delta_np[0, 2]:.4f}")

rollout_print(100.0)
rollout_print(-100.0)

