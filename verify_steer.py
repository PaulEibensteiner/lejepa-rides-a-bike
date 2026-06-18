import pybullet as p
import numpy as np
from bike.dynamics import PyBulletBikeDynamics

dyn = PyBulletBikeDynamics()
state = dyn.reset(np.array([0,0,0,0], dtype=np.float32))

# Apply positive torque for 50 steps
for _ in range(50):
    p.setJointMotorControl2(dyn.bike_id, 0, p.TORQUE_CONTROL, force=100.0, physicsClientId=dyn.client_id)
    p.stepSimulation(dyn.client_id)

angle = p.getJointState(dyn.bike_id, 0, physicsClientId=dyn.client_id)[0]
print(f"Angle after +100 torque: {angle:.4f}")

state = dyn.reset(np.array([0,0,0,0], dtype=np.float32))
for _ in range(50):
    p.setJointMotorControl2(dyn.bike_id, 0, p.TORQUE_CONTROL, force=-100.0, physicsClientId=dyn.client_id)
    p.stepSimulation(dyn.client_id)

angle = p.getJointState(dyn.bike_id, 0, physicsClientId=dyn.client_id)[0]
print(f"Angle after -100 torque: {angle:.4f}")

