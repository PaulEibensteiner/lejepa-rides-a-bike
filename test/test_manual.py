from bike.environment import BikeEnv
from train import ManualControllerPolicy

env = BikeEnv(max_steps=15000, wind_std=0.0)
policy = ManualControllerPolicy()

total_steps = 0
for _ in range(20):
    state, _ = env.reset()
    policy.reset_episode()
    steps = 0
    while True:
        action = policy(state)
        state, _, term, trunc, _ = env.step(action)
        steps += 1
        if term or trunc:
            break
    total_steps += steps
    print(f"Manual episode survived {steps} steps")

print(f"Average: {total_steps / 20}")
