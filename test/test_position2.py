import pybullet as p
import numpy as np
from bike.dynamics import PyBulletBikeDynamics, MAX_STEER, FALL_THRESHOLD
import math

class CustomManualController:
    def __init__(self, desired_heading=0.0, p=1.0, d=0.1):
        self.desired_heading = desired_heading
        self.c1 = -1.0
        self.c2 = p
        self.c3 = d

    def __call__(self, state):
        leaning = float(state[2])
        heading = float(state[3])
        leaning_dot = float(state[6])

        heading_diff = (self.desired_heading - heading + np.pi) % (2.0 * np.pi) - np.pi
        desired_lean = self.c1 * heading_diff
        desired_lean = 1.0 / (1.0 + np.exp(-desired_lean)) - 0.5

        # IMPORTANT: to turn right (lean right), you must steer left. Wait, to recover from falling right, steer right!
        # If leaning > 0 (leaning right) we must steer right (+) to catch it.
        # So angle should be proportional to leaning!
        angle = self.c2 * (leaning - desired_lean) + self.c3 * leaning_dot
        return np.array([angle], dtype=np.float32)

dyn = PyBulletBikeDynamics()

def step_pos(dyn, state, steering_angle, dt, wind_std=0.0):
    steering = float(np.clip(steering_angle, -1.0, 1.0))
    p.setTimeStep(float(dt), physicsClientId=dyn.client_id)
    p.setJointMotorControl2(dyn.bike_id, 0, p.POSITION_CONTROL, targetPosition=steering, force=100.0, physicsClientId=dyn.client_id)
    p.setJointMotorControl2(dyn.bike_id, 1, p.VELOCITY_CONTROL, targetVelocity=5.0, force=100.0, physicsClientId=dyn.client_id)
    p.stepSimulation(physicsClientId=dyn.client_id)
    obs = dyn._observe()
    obs[0] += 5.0 * dt * math.cos(obs[3])
    obs[1] += 5.0 * dt * math.sin(obs[3])
    dyn.last_state = obs.copy()
    return obs

best_survive = 0
for test_p in [0.5, 1.0, 2.0, 5.0]:
    for test_d in [0.0, 0.1, 0.5, 1.0]:
        state = dyn.reset(np.array([0,0,-0.1,0], dtype=np.float32))
        policy = CustomManualController(0.0, test_p, test_d)
        steps = 0
        for _ in range(1000):
            action = policy(state)
            state = step_pos(dyn, state, action[0], 0.02)
            steps += 1
            if abs(state[2]) > FALL_THRESHOLD:
                break
        if steps > best_survive:
            best_survive = steps
        #print(f"P={test_p}, D={test_d} -> Survived {steps} steps")
print(f"Best run: {best_survive}")
