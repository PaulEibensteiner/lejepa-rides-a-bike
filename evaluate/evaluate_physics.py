import pybullet as p
import numpy as np
from bike.dynamics import PyBulletBikeDynamics

dyn = PyBulletBikeDynamics()
state = dyn.reset(np.array([0,0,0,0], dtype=np.float32))

# Setup severe fall state
quat = p.getQuaternionFromEuler([np.pi/2 - 0.7, 0, 0])
p.resetBasePositionAndOrientation(dyn.bike_id, [0,0,dyn.last_z], quat, physicsClientId=dyn.client_id)

# Try to give it an angular velocity of -5 on lean
# (p.resetBaseVelocity) 
# Note: x axis rotation rate...
p.resetBaseVelocity(dyn.bike_id, [5,0,0], [-5,0,0], physicsClientId=dyn.client_id)
state = dyn._observe()
print("Actual initial state:", state[[2,6]])

states_pos = []
for _ in range(5):
    state = dyn.step(state, 100.0, 0.02, 0.0)
    states_pos.append(state[2])
    print("Action 100 -> lean:", state[2], "lean_dot:", state[6])

