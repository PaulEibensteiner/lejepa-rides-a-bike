"""PyBullet bicycle dynamics with velocity-aware latent state.

State vector  : [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot,
                 steer_angle, steer_rate]
                (LATENT_DIM = 10)
Action scalar : steering torque τ  (clipped to ±MAX_STEER)

Internally this wraps a persistent PyBullet DIRECT simulation.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import pybullet as p
import pybullet_data

from bike.state import S

GRAVITY: float = 10.0
BIKE_SPEED: float = 5.0
MAX_STEER: float = 100.0
FALL_THRESHOLD: float = 5.0 * np.pi / 16.0  # max bike lean is 5π/16 ≈ 56.25°
LATENT_DIM: int = 10
WHEEL_OFFSET_Y: float = 0.02762793
# Measured from the loaded wheel mesh (wheel_scaled.stl): max radius in XZ-plane.
WHEEL_RADIUS_M: float = 0.5655919117761953
# PyBullet VELOCITY_CONTROL on a revolute joint expects rad/s, not m/s.
WHEEL_TARGET_RAD_S: float = BIKE_SPEED / WHEEL_RADIUS_M

_BASE_Z: float = 1.0
_FALL_Z_MAX: float = 1.5
_FALL_Z_MIN: float = np.cos(FALL_THRESHOLD)
_STEER_JOINT: int = 0
_FRONT_WHEEL_JOINT: int = 1
_BACK_WHEEL_JOINT: int = 2


def patched_bike_urdf_path() -> str:
    """Create and return a patched bike URDF with the wheel offset correction."""
    source_dir = Path(pybullet_data.getDataPath()) / "bicycle"
    source_urdf = source_dir / "bike.urdf"

    cache_dir = Path(tempfile.gettempdir()) / "two_neurons_bike"
    cache_dir.mkdir(parents=True, exist_ok=True)
    patched_urdf = cache_dir / "bike_offset.urdf"

    tree = ET.parse(source_urdf)
    root = tree.getroot()

    # Reference correction from:
    # https://vhartmann.com/two-neurons-bike/
    # Shift wheel link inertial/visual/collision origins.
    for link in root.findall("link"):
        if link.get("name") not in {"frontWheelLink", "backWheelLink"}:
            continue
        for sub in ("inertial", "visual", "collision"):
            node = link.find(sub)
            if node is None:
                continue
            origin = node.find("origin")
            if origin is not None:
                origin.set("xyz", f"0 {WHEEL_OFFSET_Y:.8f} 0")

    # Make mesh paths absolute so the patched URDF is self-contained.
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_path = (source_dir / filename).resolve()
        mesh.set("filename", str(mesh_path))

    tree.write(patched_urdf, encoding="utf-8", xml_declaration=True)
    return str(patched_urdf)


class Renderer:
    _WIDTH: int = 960
    _HEIGHT: int = 540
    _FPS: int = 30

    def __init__(self, cid, bikeid, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to produce mp4 outputs.")
        self.cid = cid
        self.bikeid = bikeid
        # Pipe raw RGB frames straight into the system ffmpeg (which has libx264).
        # OpenCV's bundled ffmpeg only exposes the unavailable h264_v4l2m2m encoder,
        # so we bypass cv2.VideoWriter and encode H.264 here for VS Code playback.
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self._WIDTH}x{self._HEIGHT}",
                "-r",
                str(self._FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def step(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.bikeid, physicsClientId=self.cid
        )
        euler = p.getEulerFromQuaternion(orn)
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[pos[0], pos[1], 0.8],
            distance=4.0,
            yaw=np.degrees(euler[2]),
            pitch=-18,
            roll=0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=16 / 9, nearVal=0.1, farVal=50.0
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=self._WIDTH,
            height=self._HEIGHT,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.cid,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self._HEIGHT, self._WIDTH, 4)[
            :, :, :3
        ]
        assert self._proc.stdin is not None
        self._proc.stdin.write(rgb.tobytes())

    def release(self):
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()


class PyBulletBikeDynamics:
    def __init__(self, video_path: str | None) -> None:
        """Initialize lazy PyBullet handles and cached last-observed state."""
        self.client_id: int | None = None
        self.bike_id: int | None = None
        self.last_state = np.zeros(LATENT_DIM, dtype=np.float32)
        self.last_z: float = _BASE_Z
        self.initialized = False
        self.path = video_path
        self.renderer = None
        self.dt_acc = 100.0  # high value to render first frame
        self.frame_rate = 30

    def _canonical_state(self, state: np.ndarray) -> np.ndarray:
        """Convert input state (4/8/10 dims) into the full 10D latent state format."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        assert state.size == LATENT_DIM
        return state
        if state.size == 8:
            x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot = state
            return np.array(
                [
                    x,
                    y,
                    lean,
                    heading,
                    x_dot,
                    y_dot,
                    lean_dot,
                    heading_dot,
                    0.0,  # steer_angle
                    0.0,  # steer_rate
                ],
                dtype=np.float32,
            )
        if state.size != 4:
            raise ValueError(
                f"Expected state dimension 4 or {LATENT_DIM}, got {state.size}"
            )

        x, y, lean, heading = state
        x_dot = BIKE_SPEED * np.cos(heading)
        y_dot = BIKE_SPEED * np.sin(heading)
        lean_dot = 0.0
        heading_dot = 0.0
        return np.array(
            [x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot, 0.0, 0.0],
            dtype=np.float32,
        )

    def _ensure_world(self) -> None:
        """Create and configure the PyBullet world once, including plane and bike."""
        if self.initialized:
            return

        self.client_id = p.connect(p.DIRECT)
        p.setGravity(0.0, 0.0, -GRAVITY, physicsClientId=self.client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client_id
        )

        plane = p.loadURDF(
            "plane.urdf",
            [0.0, 0.0, 0.0],
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            plane,
            -1,
            mass=0,
            lateralFriction=10,
            linearDamping=0,
            angularDamping=0,
            physicsClientId=self.client_id,
        )

        self.bike_id = p.loadURDF(
            patched_bike_urdf_path(),
            [0.0, 0.0, _BASE_Z],
            [0.0, 0.0, 0.0, -1.0],
            useFixedBase=False,
            physicsClientId=self.client_id,
        )
        self.renderer = (
            Renderer(self.client_id, self.bike_id, self.path)
            if self.path is not None
            else None
        )
        self.initialized = True

    def reset(self, state: np.ndarray) -> np.ndarray:
        """Reset bike pose, velocity, and joints from a state.

        Args:
            state: Input bike state in 4D, 8D, or 10D latent form.

        Returns:
            The observed 10D state after resetting the simulator.
        """
        self._ensure_world()
        state = self._canonical_state(state)
        x, y, lean, heading, x_dot, y_dot, lean_dot, heading_dot, steer, steer_dot = (
            state.astype(float)
        )
        quat = p.getQuaternionFromEuler([np.pi / 2 + lean, 0.0, heading])

        p.resetBasePositionAndOrientation(
            self.bike_id,
            [x, y, _BASE_Z],
            quat,
            physicsClientId=self.client_id,
        )

        p.resetBaseVelocity(
            self.bike_id,
            [x_dot, y_dot, 0.0],
            [lean_dot, 0.0, heading_dot],
            physicsClientId=self.client_id,
        )

        p.resetJointState(
            self.bike_id,
            _STEER_JOINT,
            steer,
            targetVelocity=steer_dot,
            physicsClientId=self.client_id,
        )
        p.resetJointState(
            self.bike_id,
            _FRONT_WHEEL_JOINT,
            0.0,
            targetVelocity=WHEEL_TARGET_RAD_S,
            physicsClientId=self.client_id,
        )
        p.resetJointState(
            self.bike_id,
            _BACK_WHEEL_JOINT,
            0.0,
            targetVelocity=WHEEL_TARGET_RAD_S,
            physicsClientId=self.client_id,
        )

        p.setJointMotorControl2(
            self.bike_id,
            _STEER_JOINT,
            p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
            physicsClientId=self.client_id,
        )
        p.setJointMotorControl2(
            self.bike_id,
            _FRONT_WHEEL_JOINT,
            p.VELOCITY_CONTROL,
            targetVelocity=WHEEL_TARGET_RAD_S,
            force=100.0,
            physicsClientId=self.client_id,
        )
        # set target velocity to 5.0 and force cap to 100.0
        p.setJointMotorControl2(
            self.bike_id,
            _BACK_WHEEL_JOINT,
            p.VELOCITY_CONTROL,
            targetVelocity=WHEEL_TARGET_RAD_S,
            force=100.0,
            physicsClientId=self.client_id,
        )
        # slight steer dampening to prevent spinning fork
        p.changeDynamics(
            self.bike_id,
            _STEER_JOINT,
            lateralFriction=1,
            linearDamping=0,
            angularDamping=40,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            self.bike_id,
            _FRONT_WHEEL_JOINT,
            lateralFriction=1,
            linearDamping=0,
            angularDamping=0,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            self.bike_id,
            _BACK_WHEEL_JOINT,
            lateralFriction=1,
            linearDamping=0,
            angularDamping=0,
            physicsClientId=self.client_id,
        )

        obs = self._observe()
        self.last_state = obs
        return obs

    def _observe(self) -> np.ndarray:
        """Read simulator pose, velocities, and steering joint values into latent state."""
        pos, orientation = p.getBasePositionAndOrientation(
            self.bike_id, physicsClientId=self.client_id
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self.bike_id, physicsClientId=self.client_id
        )
        euler = p.getEulerFromQuaternion(orientation)
        self.last_z = float(pos[2])
        lean = euler[0] - np.pi / 2
        heading = euler[2]
        x_dot = float(lin_vel[0])
        y_dot = float(lin_vel[1])
        # make sure velocity is approx. BIKE_SPEED
        speed = np.sqrt(x_dot**2 + y_dot**2)
        # assert (
        #     abs(speed - BIKE_SPEED) < 1e-1
        # ), f"Bike speed {speed} != target speed {BIKE_SPEED}"
        x_axis = np.array(
            p.getMatrixFromQuaternion(orientation), dtype=np.float32
        ).reshape((3, 3))[:, 0]
        lean_dot = float(np.dot(x_axis, np.array([ang_vel[0], ang_vel[1], 0.0])))
        state = np.array(
            [
                pos[0],
                pos[1],
                lean,
                heading,
                x_dot,
                y_dot,
                lean_dot,
                ang_vel[2],
                p.getJointState(
                    self.bike_id, _STEER_JOINT, physicsClientId=self.client_id
                )[0],
                p.getJointState(
                    self.bike_id, _STEER_JOINT, physicsClientId=self.client_id
                )[1],
            ],
            dtype=np.float32,
        )
        return state

    def step(
        self, state: np.ndarray, steering: float, dt: float, wind_std: float
    ) -> np.ndarray:
        """Advance the simulation by one step.

        Args:
            state: Current bike state expected to match the simulator state.
            steering: Steering torque command to apply for this step.
            dt: Simulation time step in seconds.
            wind_std: Continuous-time wind noise intensity.

        Returns:
            The observed 10D state after stepping the simulator.
        """
        self._ensure_world()

        # If caller state diverges from simulator state, resync.
        expected_state = self._canonical_state(state)
        if np.linalg.norm(expected_state - self.last_state) > 1e-4:
            raise RuntimeError(
                "Input state diverged from simulator state.  Call reset() first."
            )

        steering_torque = float(np.clip(steering, -MAX_STEER, MAX_STEER))
        p.setTimeStep(float(dt), physicsClientId=self.client_id)

        p.setJointMotorControl2(
            self.bike_id,
            _STEER_JOINT,
            p.TORQUE_CONTROL,
            force=steering_torque,
            physicsClientId=self.client_id,
        )
        p.setJointMotorControl2(
            self.bike_id,
            _BACK_WHEEL_JOINT,
            p.VELOCITY_CONTROL,
            targetVelocity=WHEEL_TARGET_RAD_S,
            force=100.0,
            physicsClientId=self.client_id,
        )
        p.setJointMotorControl2(
            self.bike_id,
            _FRONT_WHEEL_JOINT,
            p.VELOCITY_CONTROL,
            targetVelocity=WHEEL_TARGET_RAD_S,
            force=100.0,
            physicsClientId=self.client_id,
        )
        if self.renderer is not None and self.dt_acc > 1.0 / self.frame_rate:
            self.renderer.step()
            self.dt_acc = 0.0
        self.dt_acc += dt

        p.stepSimulation(physicsClientId=self.client_id)
        obs = self._observe()

        if wind_std > 0.0:
            # Scale by sqrt(dt) so wind_std is a continuous-time noise intensity
            # (rad/sqrt-s).  This keeps the noise level constant regardless of dt.
            step_sigma = wind_std * float(np.sqrt(dt))
            noisy_lean = float(obs[S.lean] + np.random.normal(0.0, step_sigma))
            noisy_heading = float(
                obs[S.heading] + np.random.normal(0.0, step_sigma * 0.5)
            )
            lin_vel, ang_vel = p.getBaseVelocity(
                self.bike_id, physicsClientId=self.client_id
            )
            quat = p.getQuaternionFromEuler(
                [np.pi / 2 + noisy_lean, 0.0, noisy_heading]
            )
            p.resetBasePositionAndOrientation(
                self.bike_id,
                [float(obs[S.x]), float(obs[S.y]), self.last_z],
                quat,
                physicsClientId=self.client_id,
            )
            p.resetBaseVelocity(
                self.bike_id,
                linearVelocity=lin_vel,
                angularVelocity=ang_vel,
                physicsClientId=self.client_id,
            )
            obs = self._observe()

        self.last_state = obs
        return obs

    def is_fallen(self) -> bool:
        """Return whether we're in an invalid state"""
        assert (
            self.last_z < _FALL_Z_MAX
        ), f"Bike bounced abnormally high: last_z={self.last_z}"
        lean = self.last_state[S.lean]
        return bool(
            self.last_z < _FALL_Z_MIN
            or abs(self.last_state[S.lean]) > FALL_THRESHOLD
            or abs(self.last_state[S.steer_angle] > np.pi / 2)
        )

    def close(self) -> None:
        """Disconnect the PyBullet client and clear internal simulation handles."""
        if self.renderer is not None:
            self.renderer.release()
        if self.client_id is None:
            return
        try:
            p.disconnect(self.client_id)
        finally:
            self.client_id = None
            self.bike_id = None
            self.initialized = False
