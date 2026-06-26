"""Path plots and MP4 animation for MAPF runs.

Vendored VERBATIM from the MACPF project (``macpf/classical_mapf/viz.py``); only the
import block below is adapted to pibt-macpf's module layout, and matplotlib is forced
to the headless Agg backend with the imageio-ffmpeg binary so ``animate_paths`` writes
MP4 under multiprocessing without a system ffmpeg. With ``agent_states=None`` (which is
all PIBT supplies) it linearly interpolates the grid paths; pass the PIBT ``summary`` as
``task_summary`` to draw the live pickup/delivery targets + route lines.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless; safe inside worker processes
try:  # prefer the ffmpeg bundled with imageio-ffmpeg (no system ffmpeg needed)
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np
from matplotlib.colors import to_rgb

# --- pibt-macpf equivalents of MACPF's `from .utils/.motion/.metrics import *` ---
from .types import Coord, PathType
from .config import MAPFConfig
from .grid import is_walkable
from .metrics import paths_to_agent_positions


def nearest_cell(x: float, y: float) -> Coord:
    """Grid cell containing a float position. Only referenced by the kinodynamic
    (agent_states) branch, which PIBT never triggers, but defined for completeness."""
    return (int(round(x)), int(round(y)))


def plot_map_background(
    ax: plt.Axes,
    factory_map: np.ndarray,
    walkable_map: np.ndarray,
    colors: Optional[Dict[int, str]] = None,
) -> None:
    h, w = walkable_map.shape[:2]
    if colors:
        background = np.zeros((h, w, 3), dtype=float)
        for raw_code in np.unique(factory_map):
            code = int(raw_code)
            color = colors.get(code, "#f2f2f2")
            background[factory_map == code] = to_rgb(color)
        ax.imshow(background, origin="upper")
    else:
        background = np.where(walkable_map, 1.0, 0.15)
        ax.imshow(background, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.25)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def visualize_paths(
    env: Dict[str, Any],
    paths: Sequence[PathType],
    starts: Sequence[Coord],
    goals: Sequence[Coord],
    output_dir: Path,
) -> None:
    factory_map = np.asarray(env["factory_map"])
    walkable_map = np.asarray(env["walkable_map"])
    fig, ax = plt.subplots(figsize=(12, 9))
    plot_map_background(ax, factory_map, walkable_map, colors=env.get("colors"))
    cmap = plt.get_cmap("tab20")

    for agent_id, path in enumerate(paths):
        color = cmap(agent_id % 20)
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.85, label=f"A{agent_id}")
        ax.scatter([starts[agent_id][0]], [starts[agent_id][1]], marker="s", s=70, color=color, edgecolor="black")
        ax.scatter([goals[agent_id][0]], [goals[agent_id][1]], marker="*", s=120, color=color, edgecolor="black")

    ax.set_title("Classical MAPF prioritized planning paths")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "classical_mapf_paths.png", dpi=180)
    plt.close(fig)


def animate_paths(
    env: Dict[str, Any],
    paths: Sequence[PathType],
    starts: Sequence[Coord],
    goals: Sequence[Coord],
    output_dir: Path,
    config: MAPFConfig,
    agent_states: Optional[np.ndarray] = None,
    task_summary: Optional[Dict[str, Any]] = None,
) -> None:
    if not paths:
        return

    factory_map = np.asarray(env["factory_map"])
    walkable_map = np.asarray(env["walkable_map"])
    if agent_states is not None:
        makespan = agent_states.shape[0] - 1
        agent_positions_float = agent_states[:, :, :2]
        speed_sequence = agent_states[:, :, 3]
    else:
        makespan = max(len(path) - 1 for path in paths)
        agent_positions_float = paths_to_agent_positions(paths, makespan).astype(np.float32)
        speed_sequence = None
    id_cmap = plt.get_cmap("tab20")
    speed_cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_map_background(ax, factory_map, walkable_map, colors=env.get("colors"))
    for agent_id, (start, goal) in enumerate(zip(starts, goals)):
        color = id_cmap(agent_id % 20)
        ax.scatter([start[0]], [start[1]], marker="s", s=getattr(config, "viz_start_size", 50),
                   color=color, edgecolor="black", alpha=0.8)
        if task_summary is None:
            ax.scatter([goal[0]], [goal[1]], marker="*", s=90, color=color, edgecolor="black", alpha=0.8)

    # Each agent's full planned grid route, drawn once as a faint static underlay so the
    # video shows where every robot is headed. Both classical and online MAPF feed their
    # planned `paths` into this same function, so this covers both. Toggle via the
    # `show_planned_routes` key in configs/default.yaml.
    if getattr(config, "show_planned_routes", True):
        for agent_id, path in enumerate(paths):
            if len(path) < 2:
                continue
            ax.plot(
                [p[0] for p in path],
                [p[1] for p in path],
                color=id_cmap(agent_id % 20),
                linewidth=getattr(config, "viz_planned_route_linewidth", 1.0),
                alpha=0.30,
                zorder=0.6,  # under the robots, current-goal lines, and target markers
                solid_capstyle="round",
            )

    empty_offsets = np.empty((0, 2), dtype=np.float32)
    pickup_target_scat = ax.scatter(
        empty_offsets[:, 0],
        empty_offsets[:, 1],
        marker="P",
        s=getattr(config, "viz_target_size", 150),
        color="white",
        edgecolor="black",
        linewidth=0.9,
        label="Current pickup target",
        zorder=5,
    )
    delivery_target_scat = ax.scatter(
        empty_offsets[:, 0],
        empty_offsets[:, 1],
        marker="*",
        s=getattr(config, "viz_target_size", 150) * 1.2,
        color="white",
        edgecolor="black",
        linewidth=0.9,
        label="Current delivery target",
        zorder=5,
    )
    route_lines = [
        ax.plot(
            [],
            [],
            linestyle="--",
            linewidth=getattr(config, "viz_route_linewidth", 1.4),
            color=id_cmap(agent_id % 20),
            alpha=0.72,
            zorder=4,
        )[0]
        for agent_id in range(len(paths))
    ]

    if speed_sequence is not None:
        scat = ax.scatter(
            [],
            [],
            s=95,
            marker="o",
            c=[],
            cmap=speed_cmap,
            vmin=0.0,
            vmax=config.max_speed_mps,
            edgecolor="black",
            linewidth=0.7,
        )
        cbar = fig.colorbar(scat, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("AMR speed (m/s)")
    else:
        scat = ax.scatter([], [], s=getattr(config, "viz_robot_size", 95),
                          marker="o", edgecolor="black", linewidth=0.7)
    title = ax.set_title("")
    subframes = max(1, int(config.animation_subframes))
    total_frames = makespan * subframes + 1
    visual_positions = None
    visual_speeds = None
    task_assignments = task_summary.get("task_assignments", []) if task_summary else []
    yield_decisions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    local_update_hz = max(float(config.local_path_update_hz), 1e-6)
    local_update_interval_frames = max(1, int(round(subframes / local_update_hz)))

    if speed_sequence is not None:
        segment_lengths = np.linalg.norm(
            np.diff(agent_positions_float, axis=0),
            axis=2,
        )
        planned_distance = np.zeros((makespan + 1, len(paths)), dtype=np.float32)
        planned_distance[1:] = np.cumsum(segment_lengths, axis=0)
        visual_distance = np.zeros((total_frames, len(paths)), dtype=np.float32)
        visual_positions = np.zeros((total_frames, len(paths), 2), dtype=np.float32)
        visual_speeds = np.zeros((total_frames, len(paths)), dtype=np.float32)
        local_path_offsets = np.zeros((len(paths), 2), dtype=np.float32)
        visual_positions[0] = agent_positions_float[0]
        visual_speeds[0] = speed_sequence[0]

        def position_at_distance(agent_id: int, distance: float) -> np.ndarray:
            cumulative = planned_distance[:, agent_id]
            if distance <= 0.0:
                return agent_positions_float[0, agent_id]
            if distance >= float(cumulative[-1]):
                return agent_positions_float[-1, agent_id]

            for t_idx in range(1, makespan + 1):
                prev_d = float(cumulative[t_idx - 1])
                next_d = float(cumulative[t_idx])
                if next_d <= prev_d:
                    continue
                if distance <= next_d:
                    alpha = (distance - prev_d) / max(next_d - prev_d, 1e-6)
                    return (
                        agent_positions_float[t_idx - 1, agent_id]
                        + (agent_positions_float[t_idx, agent_id] - agent_positions_float[t_idx - 1, agent_id])
                        * alpha
                    )
            return agent_positions_float[-1, agent_id]

        def apply_local_amr_avoidance(
            frame: int,
            coarse_positions: np.ndarray,
        ) -> np.ndarray:
            if not config.use_local_amr_avoidance:
                return coarse_positions

            avoided_positions = coarse_positions.copy()
            previous_positions = visual_positions[frame - 1]
            radius = max(float(config.local_avoidance_radius_cells), 1e-6)
            max_offset = max(0.0, float(config.local_avoidance_max_offset_cells))

            for i in range(len(paths)):
                repulsion = np.zeros(2, dtype=np.float32)
                for j in range(len(paths)):
                    if i == j:
                        continue
                    delta = previous_positions[i] - previous_positions[j]
                    distance = float(np.linalg.norm(delta))
                    if distance <= 1e-6 or distance >= radius:
                        continue
                    if config.use_priority_yielding and yielding_agent_for_pair(i, j, frame) != i:
                        continue
                    direction = delta / distance
                    strength = ((radius - distance) / radius) ** 2
                    repulsion += direction * strength

                repulsion_norm = float(np.linalg.norm(repulsion))
                if repulsion_norm <= 1e-6:
                    continue
                offset = (
                    repulsion
                    / repulsion_norm
                    * min(max_offset, config.local_avoidance_strength * repulsion_norm)
                )
                avoided_positions[i] = coarse_positions[i] + offset

            return avoided_positions

        visual_safe_gap = max(float(config.visual_safe_gap_cells), float(config.vehicle_size_cells) * 0.85)

        def motion_pair_conflicts(
            prev_i: np.ndarray,
            next_i: np.ndarray,
            prev_j: np.ndarray,
            next_j: np.ndarray,
        ) -> bool:
            if float(np.linalg.norm(next_i - next_j)) < visual_safe_gap:
                return True
            # Sample both motion segments during this animation subframe. This
            # catches visual pass-through even when the final positions separate.
            for alpha in np.linspace(0.0, 1.0, 9):
                pos_i = prev_i + (next_i - prev_i) * alpha
                pos_j = prev_j + (next_j - prev_j) * alpha
                if float(np.linalg.norm(pos_i - pos_j)) < visual_safe_gap:
                    return True
            return False

        def active_assignment_by_time(agent_id: int, visual_t: float) -> Optional[Dict[str, Any]]:
            if agent_id >= len(task_assignments):
                return None
            for assignment in task_assignments[agent_id]:
                start_t = float(assignment.get("start_t", 0))
                end_t = float(assignment.get("end_t", makespan))
                if start_t <= visual_t <= end_t:
                    return assignment
            return None

        def yielding_agent_for_pair(i: int, j: int, frame: int) -> int:
            pair = (min(i, j), max(i, j))
            cached = yield_decisions.get(pair)
            if cached is not None:
                yielding_agent, hold_until_frame = cached
                if frame <= hold_until_frame:
                    return yielding_agent

            if not config.use_priority_yielding:
                return max(i, j)

            visual_t = frame / subframes
            assignment_i = active_assignment_by_time(i, visual_t)
            assignment_j = active_assignment_by_time(j, visual_t)
            loaded_i = 1 if assignment_i and assignment_i.get("action") == "delivery" else 0
            loaded_j = 1 if assignment_j and assignment_j.get("action") == "delivery" else 0

            priority_i = (loaded_i, -i)
            priority_j = (loaded_j, -j)
            yielding_agent = j if priority_i >= priority_j else i
            hold_until = frame + max(1, int(config.yield_decision_hold_frames))
            yield_decisions[pair] = (yielding_agent, hold_until)
            return yielding_agent

        def choose_visual_detour_position(
            agent_id: int,
            previous_position: np.ndarray,
            desired_position: np.ndarray,
            guarded_positions: np.ndarray,
        ) -> Optional[np.ndarray]:
            if not config.use_local_detour_when_blocked:
                return None

            desired_delta = desired_position - previous_position
            desired_norm = float(np.linalg.norm(desired_delta))
            if desired_norm <= 1e-6:
                return None

            step = min(
                max(float(config.local_detour_max_step_cells), 0.05),
                max(desired_norm, 0.05),
            )
            desired_direction = desired_delta / desired_norm
            samples: List[np.ndarray] = []
            for angle in np.linspace(-math.pi, math.pi, 16, endpoint=False):
                direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                if float(np.dot(direction, desired_direction)) < -0.35:
                    continue
                samples.append(previous_position + direction * step)

            def candidate_is_safe(candidate: np.ndarray) -> bool:
                cell = nearest_cell(float(candidate[0]), float(candidate[1]))
                if not is_walkable(*cell, walkable_map):
                    return False
                for other_id, other_position in enumerate(guarded_positions):
                    if other_id == agent_id:
                        continue
                    if float(np.linalg.norm(candidate - other_position)) < visual_safe_gap:
                        return False
                return True

            safe_samples = [candidate for candidate in samples if candidate_is_safe(candidate)]
            if not safe_samples:
                return None

            def candidate_score(candidate: np.ndarray) -> Tuple[float, float]:
                to_desired = float(np.linalg.norm(candidate - desired_position))
                progress = -float(np.dot(candidate - previous_position, desired_direction))
                return (to_desired, progress)

            return min(safe_samples, key=candidate_score)

        def apply_visual_collision_guard(
            frame: int,
            proposed_positions: np.ndarray,
            proposed_speeds: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            guarded_positions = proposed_positions.copy()
            guarded_speeds = proposed_speeds.copy()
            blocked = np.zeros(len(paths), dtype=bool)

            for _ in range(len(paths)):
                changed = False
                previous_positions = visual_positions[frame - 1]
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        if not motion_pair_conflicts(
                            previous_positions[i],
                            guarded_positions[i],
                            previous_positions[j],
                            guarded_positions[j],
                        ):
                            continue

                        block_id = yielding_agent_for_pair(i, j, frame)
                        if blocked[block_id]:
                            continue
                        blocked[block_id] = True
                        detour_position = choose_visual_detour_position(
                            block_id,
                            previous_positions[block_id],
                            proposed_positions[block_id],
                            guarded_positions,
                        )
                        if detour_position is not None:
                            guarded_positions[block_id] = detour_position
                            guarded_speeds[block_id] *= 0.55
                        else:
                            guarded_positions[block_id] = previous_positions[block_id]
                            guarded_speeds[block_id] = 0.0
                            visual_distance[frame, block_id] = visual_distance[frame - 1, block_id]
                        changed = True
                if not changed:
                    break

            return guarded_positions, guarded_speeds

        for frame in range(1, total_frames):
            base_t = min((frame - 1) // subframes + 1, makespan)
            sub = (frame - 1) % subframes + 1
            local_alpha = sub / subframes
            prev_speed = speed_sequence[base_t - 1]
            curr_speed = speed_sequence[base_t]
            speeds = prev_speed + (curr_speed - prev_speed) * local_alpha
            step_cells = (speeds * (config.dt_s / subframes)) / max(config.cell_size_m, 1e-6)
            max_planned_distance = (
                planned_distance[base_t - 1]
                + (planned_distance[base_t] - planned_distance[base_t - 1]) * local_alpha
            )
            visual_distance[frame] = np.minimum(
                visual_distance[frame - 1] + step_cells,
                max_planned_distance,
            )
            for agent_id in range(len(paths)):
                visual_positions[frame, agent_id] = position_at_distance(
                    agent_id,
                    float(visual_distance[frame, agent_id]),
                )
            coarse_positions = visual_positions[frame].copy()
            if frame % local_update_interval_frames == 0:
                locally_adjusted_positions = apply_local_amr_avoidance(frame, coarse_positions)
                proposed_positions = locally_adjusted_positions
            else:
                proposed_positions = coarse_positions + local_path_offsets

            guarded_positions, guarded_speeds = apply_visual_collision_guard(
                frame,
                proposed_positions,
                speeds,
            )
            if frame % local_update_interval_frames == 0:
                local_path_offsets = guarded_positions - coarse_positions
            visual_positions[frame] = guarded_positions
            visual_speeds[frame] = guarded_speeds

    def interpolated_frame(frame: int) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        if visual_positions is not None and visual_speeds is not None:
            return visual_positions[frame], visual_speeds[frame], frame / subframes

        if frame == 0:
            speeds = speed_sequence[0] if speed_sequence is not None else None
            return agent_positions_float[0], speeds, 0.0

        base_t = min((frame - 1) // subframes + 1, makespan)
        sub = (frame - 1) % subframes + 1
        local_alpha = sub / subframes
        prev_positions = agent_positions_float[base_t - 1]
        curr_positions = agent_positions_float[base_t]

        if speed_sequence is None:
            return prev_positions + (curr_positions - prev_positions) * local_alpha, None, float(base_t)

        speeds = speed_sequence[base_t]
        positions = prev_positions + (curr_positions - prev_positions) * local_alpha
        return positions, speeds, (base_t - 1) + local_alpha

    visual_assignment_indices = [0 for _ in range(len(paths))]
    last_route_frame = -1

    def advance_visual_assignment_indices(frame: int, positions: np.ndarray) -> None:
        nonlocal last_route_frame
        if frame <= last_route_frame:
            return
        last_route_frame = frame

        target_reached_radius = 0.45
        for agent_id in range(len(paths)):
            if agent_id >= len(task_assignments):
                continue
            assignments = task_assignments[agent_id]
            while visual_assignment_indices[agent_id] < len(assignments):
                assignment = assignments[visual_assignment_indices[agent_id]]
                target = assignment.get("target")
                if not isinstance(target, list) or len(target) < 2:
                    visual_assignment_indices[agent_id] += 1
                    continue
                target_xy = np.asarray(target[:2], dtype=np.float32)
                distance = float(np.linalg.norm(positions[agent_id] - target_xy))
                if distance > target_reached_radius:
                    break
                visual_assignment_indices[agent_id] += 1

    def active_assignment_for(agent_id: int, visual_t: float) -> Optional[Dict[str, Any]]:
        if agent_id >= len(task_assignments):
            return None
        assignments = task_assignments[agent_id]
        if visual_assignment_indices[agent_id] >= len(assignments):
            return None
        return assignments[visual_assignment_indices[agent_id]]

    def current_target_offsets(visual_t: float) -> Tuple[np.ndarray, List[Any], np.ndarray, List[Any]]:
        pickup_offsets: List[List[float]] = []
        pickup_colors: List[Any] = []
        delivery_offsets: List[List[float]] = []
        delivery_colors: List[Any] = []

        for agent_id, assignments in enumerate(task_assignments):
            active_assignment = active_assignment_for(agent_id, visual_t)
            if active_assignment is None:
                continue
            target = active_assignment.get("target")
            if not isinstance(target, list) or len(target) < 2:
                continue
            color = id_cmap(agent_id % 20)
            offset = [float(target[0]), float(target[1])]
            if active_assignment.get("action") == "pickup":
                pickup_offsets.append(offset)
                pickup_colors.append(color)
            else:
                delivery_offsets.append(offset)
                delivery_colors.append(color)

        pickup_array = np.asarray(pickup_offsets, dtype=np.float32) if pickup_offsets else empty_offsets
        delivery_array = np.asarray(delivery_offsets, dtype=np.float32) if delivery_offsets else empty_offsets
        return pickup_array, pickup_colors, delivery_array, delivery_colors

    def update_route_lines(positions: np.ndarray, visual_t: float) -> None:
        for agent_id, line in enumerate(route_lines):
            active_assignment = active_assignment_for(agent_id, visual_t)
            if active_assignment is None:
                line.set_data([], [])
                continue

            target = active_assignment.get("target")
            if not isinstance(target, list) or len(target) < 2:
                line.set_data([], [])
                continue
            line.set_data(
                [float(positions[agent_id, 0]), float(target[0])],
                [float(positions[agent_id, 1]), float(target[1])],
            )

    def update(frame: int):
        positions, speeds, visual_t = interpolated_frame(frame)
        scat.set_offsets(positions)
        advance_visual_assignment_indices(frame, positions)
        update_route_lines(positions, visual_t)
        pickup_offsets, pickup_colors, delivery_offsets, delivery_colors = current_target_offsets(visual_t)
        pickup_target_scat.set_offsets(pickup_offsets)
        pickup_target_scat.set_color(pickup_colors)
        delivery_target_scat.set_offsets(delivery_offsets)
        delivery_target_scat.set_color(delivery_colors)
        if speeds is not None:
            scat.set_array(speeds)
            title.set_text(
                f"Kinodynamic AMR fleet, t={visual_t:.2f}, mean speed={float(np.mean(speeds)):.2f} m/s"
            )
        else:
            colors = [id_cmap(i % 20) for i in range(len(paths))]
            scat.set_color(colors)
            title.set_text(f"Classical MAPF agent positions, t={visual_t:.2f}")
        return (scat, pickup_target_scat, delivery_target_scat, title, *route_lines)

    anim = FuncAnimation(fig, update, frames=total_frames, interval=config.animation_interval_ms, blit=False)
    fps = 30  # fixed playback frame rate
    dpi = int(getattr(config, "viz_video_dpi", 200))    # figsize (10,8) -> dpi 200 = 2000x1600 px
    cq = int(getattr(config, "viz_video_cq", 15))       # NVENC constant quality (lower = better)
    # Encode on the GPU (NVENC H.264) at high quality; fall back to high-quality CPU x264, then
    # GIF -- so a box without an NVIDIA GPU (or a fresh clone without ffmpeg) still produces output.
    # Quality (vs the old fixed 2400 kbps libx264) comes from CQ-based rate control + the higher dpi.
    mp4 = output_dir / "classical_mapf_animation.mp4"
    encoders = [
        ("h264_nvenc", ["-preset", "p7", "-tune", "hq", "-rc", "vbr",
                        "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p"]),
        ("libx264", ["-preset", "slow", "-crf", str(cq + 1), "-pix_fmt", "yuv420p"]),
    ]
    if FFMpegWriter.isAvailable():
        for codec, extra in encoders:
            try:
                anim.save(mp4, writer=FFMpegWriter(fps=fps, codec=codec, extra_args=extra), dpi=dpi)
                break
            except Exception:                            # encoder unavailable -> try the next one
                continue
        else:
            anim.save(output_dir / "classical_mapf_animation.gif", writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(output_dir / "classical_mapf_animation.gif", writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
