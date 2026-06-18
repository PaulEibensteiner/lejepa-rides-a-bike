import torch
import numpy as np
from bike.models import SimpleLeanHeadingModel
from bike.cem import CEMPlanner, _clamp_differentials, _clamp_states

model = SimpleLeanHeadingModel()
model.load_state_dict(torch.load("meins_4/model_1/checkpoint.pt"))
model.eval()

def rollout_print(action_val):
    state = np.array([[0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    print(f"\nEvaluating action {action_val} (Analytical Integration):")
    total_cost = 0
    for t in range(5):
        a_t = torch.tensor([[action_val]], dtype=torch.float32)
        state_t = torch.tensor(state, dtype=torch.float32)
        delta_t = model.predict_delta(state_t, a_t)
        delta_np = _clamp_differentials(delta_t.detach().numpy() * (0.02 / 0.02))
        
        # Analytic Integration!
        state[:, 6] += delta_np[:, 2]
        state[:, 7] += delta_np[:, 3]
        
        state[:, 2] += state[:, 6] * 0.02   # dt = 0.02
        state[:, 3] += state[:, 7] * 0.02
        
        state = _clamp_states(state)
        
        cost = 10.0 * state[0, 2]**2
        total_cost += cost
    print(f"Total Cost: {total_cost:.4f}")

rollout_print(100.0)
rollout_print(-100.0)

