import pybullet as p
import numpy as np
from bike.dynamics import PyBulletBikeDynamics, MAX_STEER, FALL_THRESHOLD
import math

class CustomManualController:
    def __init__(self, desired_heading=0.0):
        self.desired_heading = desired_heading
        self.c1 = -1.0
        self.c2 = -1.0     # P term for target lean
        self.c3 = -0.5     # D term for lean

    def __call__(self, state):
        leaning = float(state[2])
        heading = float(state[3])
        leaning_dot = float(state[6])

        # Same target logic
        heading_diff = (self.desired_heading - heading + np.pi) % (2.0 * np.pi) - np.pi
        desired_lean = self.c1 * heading_diff
        desired_lean = 1.0 / (1.0 + np.exp(-desired_lean)) - 0.5

        # Output angle directly
        angle = self.c2 * (desired_lean - leaning) - self.c3 * leaning_dot
        return np.array([angle], dtype=np.float32)

dyn = PyBulletBikeDynamics()
state = dyn.reset(np.array([0,0,0,0], dtype=np.float32))
policy = CustomManualController()

# Override dynamics to use POSITION_CONTROL
def step_pos(dyn, state, steering_angle, dt, wind_std=0.0):
    steering = float(np.clip(steering_angle, -1.0, 1.0))
    p.setTimeStep(float(dt), physicsClientId=dyn.client_id)

    p.setJointMotorControl2(
        dyn.bike_id,
        0, # steer
        p.POSITION_CONTROL,
        targetPosition=steering,
        force=50.0,
        physicsClientId=dyn.client_id,
    )
    p.setJointMotorControl2(
        dyn.bike_id,
        1, # back wheel
        p.VELOCITY_CONTROL,
        targetVelocity=5.0,
        force=100.0,
        physicsClientId=dyn.client_id,
    )

    p.stepSimulation(physicsClientId=dyn.client_id)
    obs = dyn._observe()

    obs[0] += 5.0 * dt * math.cos(obs[3])
    obs[1] += 5.0 * dt * math.sin(obs[3])
    dyn.last_state = obs.copy()
    return obs

steps = 0
for _ in range(500):
    action = policy(state)
    state = step_pos(dyn, state, action[0], 0.02)
    steps += 1
    if abs(state[2]) > FALL_THRESHOLD:
        print(f"Fell at step {steps}")
        break
else:
    print("Survived!")

