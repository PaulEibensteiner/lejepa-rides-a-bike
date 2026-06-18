import pybullet as p
import numpy as np
from bike.dynamics import PyBulletBikeDynamics

dyn = PyBulletBikeDynamics()

state1 = dyn.reset(np.array([0,0,0,0], dtype=np.float32))
# Turn wheels far right
for _ in range(200):
    p.setJointMotorControl2(dyn.bike_id, 0, p.TORQUE_CONTROL, force=100.0, physicsClientId=dyn.client_id)
    p.stepSimulation(dyn.client_id)

obs1 = dyn._observe()

dyn2 = PyBulletBikeDynamics()
state2 = dyn2.reset(np.array([0,0,0,0], dtype=np.float32))
# Turn wheels far left
for _ in range(200):
    p.setJointMotorControl2(dyn2.bike_id, 0, p.TORQUE_CONTROL, force=-100.0, physicsClientId=dyn2.client_id)
    p.stepSimulation(dyn2.client_id)

obs2 = dyn2._observe()

# Force obs2 to exactly match obs1's lean and velocities to create an identical state vector
quat = p.getQuaternionFromEuler([obs1[2] + np.pi/2, 0, obs1[3]])
p.resetBasePositionAndOrientation(dyn2.bike_id, [obs1[0], obs1[1], dyn2.last_z], quat, physicsClientId=dyn2.client_id)

# Apply 0 torque for 10 steps and see what happens to lean!
s1_list = []
s2_list = []
for _ in range(10):
    obs_a = dyn.step(obs1, 0.0, 0.02, 0.0)
    obs_b = dyn2.step(obs2, 0.0, 0.02, 0.0)
    s1_list.append(obs_a[2])
    s2_list.append(obs_b[2])

print("Trajectory A (wheels started right):", [f"{x:.4f}" for x in s1_list])
print("Trajectory B (wheels started left):", [f"{x:.4f}" for x in s2_list])
