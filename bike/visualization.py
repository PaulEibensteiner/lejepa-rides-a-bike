from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from scipy.stats import kstest, norm

from bike.dynamics import BIKE_SPEED, MAX_STEER, patched_bike_urdf_path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _model_color(name: str) -> str:
    if "Model 1" in name:
        return "tab:blue"
    if "Model 2" in name:
        return "tab:orange"
    if "Model 3" in name:
        return "tab:green"
    return "tab:gray"


def _model_marker(name: str) -> str:
    if "Model 1" in name:
        return "o"
    if "Model 2" in name:
        return "s"
    if "Model 3" in name:
        return "^"
    return "x"


def _model_linestyle(name: str) -> str:
    if "Model 1" in name:
        return "-"
    if "Model 2" in name:
        return "--"
    if "Model 3" in name:
        return ":"
    return "-"


def _run_color(run_idx: int) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab10")
    return cmap(run_idx % 10)


def plot_training_curves(
    loss_histories: dict[str, dict[str, list[float]]], save_path: Path
) -> None:
    _ensure_parent(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, history in loss_histories.items():
        color = _model_color(name)
        axes[0].plot(history.get("pred_loss", []), label=name, color=color)
        sigreg_values = history.get("regularization_loss", [])
        if any(v > 0 for v in sigreg_values):
            axes[1].plot(sigreg_values, label=name, color=color)

    axes[0].set_title("Prediction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("SIGReg Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_trajectory_comparison(
    episodes_by_model: dict[str, list[dict[str, Any]]], save_path: Path
) -> None:
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, episodes in episodes_by_model.items():
        marker = _model_marker(model_name)
        linestyle = _model_linestyle(model_name)
        for i, episode in enumerate(episodes):
            states = np.asarray(episode.get("states", []), dtype=np.float32)
            if len(states) == 0:
                continue
            color = _run_color(i)
            markevery = max(1, len(states) // 30)
            ax.plot(
                states[:, 0],
                states[:, 1],
                linestyle=linestyle,
                color=color,
                alpha=0.8,
                marker=marker,
                markevery=markevery,
                markersize=3,
                label=f"{model_name} run {i + 1}",
            )

    ax.set_title("Trajectory Comparison (color=run, marker/style=model)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_state_traces(
    episodes_by_model: dict[str, list[dict[str, Any]]], save_path: Path
) -> None:
    _ensure_parent(save_path)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    for model_name, episodes in episodes_by_model.items():
        if not episodes:
            continue
        marker = _model_marker(model_name)
        linestyle = _model_linestyle(model_name)
        for i, episode in enumerate(episodes):
            states = np.asarray(episode.get("states", []), dtype=np.float32)
            if len(states) == 0:
                continue
            color = _run_color(i)
            t = np.arange(len(states))
            markevery = max(1, len(states) // 30)
            label = f"{model_name} run {i + 1}"
            axes[0].plot(
                t,
                states[:, 2],
                color=color,
                alpha=0.8,
                linestyle=linestyle,
                marker=marker,
                markevery=markevery,
                markersize=3,
                label=label,
            )
            axes[1].plot(
                t,
                states[:, 3],
                color=color,
                alpha=0.8,
                linestyle=linestyle,
                marker=marker,
                markevery=markevery,
                markersize=3,
                label=label,
            )

    axes[0].set_title("Lean Trace")
    axes[0].set_ylabel("lean [rad]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("Heading Trace")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("heading [rad]")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_metrics_bar(
    model_summaries: dict[str, dict[str, float]], save_path: Path
) -> None:
    _ensure_parent(save_path)
    names = list(model_summaries.keys())
    success = [model_summaries[n].get("success_rate", 0.0) for n in names]
    distance = [model_summaries[n].get("mean_distance", 0.0) for n in names]
    rms_lean = [model_summaries[n].get("mean_rms_lean", 0.0) for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, success, width=w, label="success_rate")
    ax.bar(x, distance, width=w, label="mean_distance")
    ax.bar(x + w, rms_lean, width=w, label="mean_rms_lean")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_title("Model Metrics Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_torque_traces(
    episodes_by_model: dict[str, list[dict[str, Any]]], save_path: Path
) -> None:
    _ensure_parent(save_path)
    n_models = len(episodes_by_model)
    fig, axes = plt.subplots(n_models, 1, figsize=(11, 3.2 * n_models), sharex=False)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, episodes) in zip(axes, episodes_by_model.items()):
        marker = _model_marker(model_name)
        linestyle = _model_linestyle(model_name)
        max_steps = 0
        for i, episode in enumerate(episodes):
            actions = np.asarray(episode.get("actions", []), dtype=np.float32)
            if actions.size == 0:
                continue
            torque = actions.reshape(-1)
            t = np.arange(len(torque))
            color = _run_color(i)
            max_steps = max(max_steps, len(torque))
            markevery = max(1, len(torque) // 30)
            ax.plot(
                t,
                torque,
                color=color,
                alpha=0.8,
                linewidth=1.0,
                linestyle=linestyle,
                marker=marker,
                markevery=markevery,
                markersize=3,
                label=f"{model_name} run {i + 1}",
            )

        ax.axhline(MAX_STEER, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(-MAX_STEER, color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, max_steps if max_steps > 0 else 1)
        ax.set_ylim(-1.1 * MAX_STEER, 1.1 * MAX_STEER)
        ax.set_title(f"{model_name} torque per step")
        ax.set_ylabel("torque")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _render_pybullet_frames(
    initial_state: np.ndarray,
    actions: np.ndarray,
    frame_dir: Path,
    dt: float = 0.001,
    frame_skip: int = 1,
) -> int:
    frame_dir.mkdir(parents=True, exist_ok=True)

    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -10, physicsClientId=cid)
        p.setTimeStep(float(dt), physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=cid)
        bike = p.loadURDF(
            patched_bike_urdf_path(),
            [0, 0, 1.0],
            [0, 0, 0, -1],
            physicsClientId=cid,
        )

        state = np.asarray(initial_state, dtype=np.float32)
        x, y, lean, heading = [float(v) for v in state[:4]]
        quat = p.getQuaternionFromEuler([np.pi / 2 + lean, 0.0, heading])
        p.resetBasePositionAndOrientation(bike, [x, y, 1.0], quat, physicsClientId=cid)
        p.resetBaseVelocity(
            bike,
            [float(state[4]), float(state[5]), 0.0],
            [float(state[6]), 0.0, float(state[7])],
            physicsClientId=cid,
        )

        p.setJointMotorControl2(
            bike,
            0,
            p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
            physicsClientId=cid,
        )
        p.setJointMotorControl2(
            bike,
            1,
            p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
            physicsClientId=cid,
        )

        frame_skip = max(1, int(frame_skip))
        count = 0
        frame_idx = 0
        for i, action in enumerate(np.asarray(actions, dtype=np.float32)):
            torque = float(np.clip(action[0], -MAX_STEER, MAX_STEER))
            p.setJointMotorControl2(
                bike,
                0,
                p.TORQUE_CONTROL,
                force=torque,
                physicsClientId=cid,
            )
            p.setJointMotorControl2(
                bike,
                2,
                p.VELOCITY_CONTROL,
                targetVelocity=5.0,
                force=100.0,
                physicsClientId=cid,
            )
            p.stepSimulation(physicsClientId=cid)

            if i % frame_skip != 0:
                continue

            pos, orn = p.getBasePositionAndOrientation(bike, physicsClientId=cid)
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
                width=960,
                height=540,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=cid,
            )
            img = np.asarray(rgba, dtype=np.uint8).reshape(540, 960, 4)[:, :, :3]
            plt.imsave(frame_dir / f"frame_{frame_idx:05d}.png", img)
            frame_idx += 1
            count += 1

        return count
    finally:
        p.disconnect(cid)


def animate_episode_pybullet(
    initial_state: np.ndarray,
    actions: np.ndarray,
    save_path: Path,
    fps: int = 30,
    dt: float = 0.001,
    frame_skip: int | None = None,
) -> None:
    _ensure_parent(save_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to produce mp4 outputs.")

    # Auto-compute frame_skip so the rendered fps matches the requested fps.
    # E.g. dt=0.001, fps=20 → 1 frame every 50 simulation steps.
    if frame_skip is None:
        frame_skip = max(1, round(1.0 / (fps * dt)))

    with tempfile.TemporaryDirectory(prefix="bike_frames_") as tmp:
        frame_dir = Path(tmp)
        frame_count = _render_pybullet_frames(
            initial_state, actions, frame_dir, dt=dt, frame_skip=frame_skip
        )
        if frame_count == 0:
            raise RuntimeError("No frames were rendered for video output.")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-pix_fmt",
            "yuv420p",
            str(save_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def plot_sigreg_distribution(
    model_name: str,
    projection_values: np.ndarray,
    save_path: Path,
    ks_stats_path: Path | None = None,
) -> dict[str, float]:
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        projection_values,
        bins=50,
        density=True,
        alpha=0.6,
        color="tab:green",
        label=model_name,
    )

    xs = np.linspace(-4.0, 4.0, 500)
    ax.plot(xs, norm.pdf(xs), color="black", linestyle="--", label="N(0,1)")
    ax.set_title("SIGReg Projection Distribution")
    ax.set_xlabel("Projection value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

    ks_stat, ks_p = kstest(projection_values, "norm")
    stats = {"ks_stat": float(ks_stat), "ks_pvalue": float(ks_p)}

    if ks_stats_path is not None:
        _ensure_parent(ks_stats_path)
        ks_stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return stats


def plot_wind_sweep(wind_results: dict[str, dict[str, float]], save_path: Path) -> None:
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, by_wind in wind_results.items():
        xs = sorted(float(k) for k in by_wind.keys())
        ys = [by_wind[f"{x:.2f}"] for x in xs]
        ax.plot(xs, ys, marker="o", label=model_name, color=_model_color(model_name))

    ax.set_title("Wind Sweep: Survival Rate vs wind_std")
    ax.set_xlabel("wind_std")
    ax.set_ylabel("survival_rate")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
