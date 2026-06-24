"""Closed-loop (receding-horizon) online MAPF runner.

Loop (1 Hz):  observe -> ConvLSTM forecast -> prioritized re-plan -> advance 1 cell.
The kinodynamic motion model + proximity safety controller are then applied over
the assembled trajectory as the execution/visualization layer (the 10 Hz "feel"
comes from the animation sub-frames), exactly as classical_mapf does.

Run:  python -m macpf.online_mapf --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from macpf.classical_mapf import (
    apply_proximity_safety_controller,
    build_additive_congestion_label_sequence,
    build_occupancy_sequence,
    compute_clearance_stats,
    compute_collision_count,
    compute_interpolated_clearance_stats,
    compute_metrics,
    compute_start_clearance_stats,
    filter_point_groups_by_walkability,
    grid_paths_to_kinodynamic_states,
    normalize_point_groups,
    normalize_points,
    path_position_at,
    paths_to_agent_positions,
    select_start_goal_pairs,
    states_to_agent_positions,
    visualize_paths,
)
from macpf.classical_mapf import animate_paths
from macpf.classical_mapf.classical_mapf import (
    FIGURES_DIR,
    load_factory_environment,
    resolve_config_path,
    save_results,
)
from macpf.classical_mapf.utils import load_config

from .observe import FrameBuilder, build_dec, build_enc
from .predictor import CongestionPredictor
from .replanner import replan
from .world_state import World


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a closed-loop online MAPF simulation with live congestion prediction."
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config. Relative paths resolve from the project root.")
    parser.add_argument("--model", default="models/congestion_convlstm.pt",
                        help="Trained ConvLSTM checkpoint.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--max-time", type=int, default=None, help="Override config.max_time (steps).")
    parser.add_argument("--num-agents", type=int, default=None, help="Override config.num_agents.")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    parser.add_argument("--replan-every", type=int, default=None,
                        help="Override config.online_replan_every (steps between re-plans).")
    parser.add_argument("--no-animation", action="store_true", help="Skip the GIF animation.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip all matplotlib output (PNG + GIF). Use on headless machines.")
    parser.add_argument("--metrics-out", default=None,
                        help="Sweep mode: write the metrics JSON here and skip the full save + figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_config_path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded config: {config_path}")
    else:
        config = load_config()
        print(f"Config file not found ({config_path}); using configs/default.yaml.")

    overrides = {}
    if args.max_time is not None:
        overrides["max_time"] = args.max_time
    if args.num_agents is not None:
        overrides["num_agents"] = args.num_agents
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.replan_every is not None:
        overrides["online_replan_every"] = args.replan_every
    if args.no_animation:
        overrides["save_animation"] = False
    if overrides:
        config = config.replace(**overrides)

    max_time = int(config.max_time)
    replan_every = max(1, int(config.online_replan_every))

    env = load_factory_environment(config)
    walkable_map = np.asarray(env["walkable_map"]).astype(bool)
    H, W = walkable_map.shape[:2]
    obstacle_map = np.asarray(env["obstacle_map"])

    starts, _ = select_start_goal_pairs(env, walkable_map, config)

    pickup_groups = filter_point_groups_by_walkability(
        normalize_point_groups(env.get("pickup_point_groups"))
        or {"pickup": normalize_points(env.get("pickup_points"))},
        walkable_map,
    )
    delivery_groups = filter_point_groups_by_walkability(
        normalize_point_groups(env.get("delivery_point_groups"))
        or {"delivery": normalize_points(env.get("delivery_points"))},
        walkable_map,
    )
    if not pickup_groups:
        raise ValueError("Online mode needs at least one walkable pickup point.")
    if not delivery_groups:
        raise ValueError("Online mode needs at least one walkable delivery point.")

    print(f"Loading congestion model: {args.model}")
    predictor = CongestionPredictor(args.model, device=args.device)
    t_in, t_out = predictor.t_in, predictor.t_out
    print(f"Model ready on {predictor.device} (t_in={t_in}, t_out={t_out}).")

    # Plan far enough ahead to reach distant goals; execute only the next step(s).
    # online_plan_horizon caps how far each re-plan looks/reserves: 0 = full horizon
    # (max_time + lookahead); a smaller cap keeps per-step cost flat on long sims
    # (goals are <= ~map-diagonal away, so a few hundred steps suffices to reach them).
    full_horizon = max_time + max(0, int(config.continuous_task_lookahead))
    plan_horizon = int(config.online_plan_horizon)
    if plan_horizon <= 0 or plan_horizon > full_horizon:
        plan_horizon = full_horizon
    planning_config = config.replace(
        use_ai_congestion_cost=True,
        max_time=plan_horizon,
    )
    warmup_zero = bool(config.online_warmup_zero_cost)

    frame_builder = FrameBuilder(starts, obstacle_map)
    world = World(starts, t_in)
    rng = np.random.default_rng(config.seed + 1000)

    # Seed the history with the state at t=0.
    world.push_frame(frame_builder.frame(world.positions_array(), world.goals_typed()))

    print(f"Running online loop: {len(starts)} AMRs, {max_time} steps, re-plan every {replan_every}s")
    committed = None
    step_in_plan = 0
    replans = 0
    progress_every = max(1, max_time // 20)

    for t in range(max_time):
        if t % replan_every == 0:
            if warmup_zero and len(world.history) < t_in:
                cost = np.zeros((1, H, W), dtype=np.float32)
            else:
                enc = build_enc(list(world.history), t_in)
                dec = build_dec(world.history[-1], t_out)
                pred = predictor.predict(enc, dec)  # (t_out, H, W), label scale
                # The label sums a center_value (=100) kernel per robot, so raw peaks
                # reach the hundreds. Added straight into A*'s f (= g + h + w*cost),
                # that walls off crowded cells and makes step-by-step re-planning
                # oscillate instead of progressing. Express congestion in
                # "robot-equivalents" (~1.0 == one robot's worth) so it is a soft bias
                # that ai_cost_weight trades off against path length (1.0 / step).
                pred = pred / max(1e-6, float(config.congestion_center_value))
                # Align so get_congestion_cost(cost, nt) at local step nt uses the
                # forecast for "now + nt": cost[nt] == pred[nt-1] for nt >= 1.
                cost = np.concatenate([pred[:1], pred], axis=0)
            committed = replan(
                world, cost, pickup_groups, delivery_groups,
                walkable_map, planning_config, rng,
            )
            step_in_plan = 0
            replans += 1

        step_in_plan += 1
        for agent_id, agent in enumerate(world.agents):
            new_pos = path_position_at(committed[agent_id], step_in_plan)
            agent.pos = new_pos
            agent.path.append(new_pos)
            if agent.goal is not None and new_pos == agent.goal:
                agent.complete_goal(t + 1)
        world.t = t + 1
        world.push_frame(frame_builder.frame(world.positions_array(), world.goals_typed()))

        if (t + 1) % progress_every == 0 or t + 1 == max_time:
            deliveries = sum(a.completed_deliveries for a in world.agents)
            print(f"  step {t + 1}/{max_time} | deliveries={deliveries} | replans={replans}",
                  flush=True)

    paths = world.assembled_paths()

    # Execution / visualization layer (reused from classical_mapf, unchanged).
    if config.use_kinodynamic_motion:
        agent_states = grid_paths_to_kinodynamic_states(paths, walkable_map, config)
        agent_states, safety_summary = apply_proximity_safety_controller(agent_states, config)
        agent_positions = states_to_agent_positions(agent_states)
    else:
        agent_states = None
        safety_summary = None
        agent_positions = paths_to_agent_positions(paths, max_time)

    completed_deliveries = [a.completed_deliveries for a in world.agents]
    completed_targets = [a.completed_targets for a in world.agents]
    final_goals = [list(a.goal) if a.goal is not None else list(a.pos) for a in world.agents]
    task_summary = {
        "completed_deliveries": completed_deliveries,
        "completed_targets": completed_targets,
        "total_completed_deliveries": int(sum(completed_deliveries)),
        "total_completed_targets": int(sum(completed_targets)),
        "final_goals": final_goals,
        "task_assignments": [a.all_assignments() for a in world.agents],
    }
    goals = [tuple(g) for g in final_goals]

    occupancy_sequence = build_occupancy_sequence(agent_positions, H, W)
    congestion_labels = build_additive_congestion_label_sequence(
        agent_positions, H, W,
        center_value=config.congestion_center_value,
        step_value=config.congestion_step_value,
    )

    # Same metric set as classical_mapf so the two runs are directly comparable.
    metrics = compute_metrics(paths, starts, goals, task_summary=task_summary, max_t=max_time)
    metrics["mapf_solver"] = "online_receding_horizon_convlstm"
    metrics["online_replan_every"] = replan_every
    metrics["online_replans"] = replans
    metrics["online_plan_horizon"] = int(planning_config.max_time)
    metrics["model_t_in"] = t_in
    metrics["model_t_out"] = t_out
    if agent_states is not None:
        speeds = agent_states[:, :, 3]
        nonzero = speeds[speeds > 0]
        metrics["max_observed_speed_mps"] = float(np.max(speeds))
        metrics["min_nonzero_observed_speed_mps"] = float(np.min(nonzero)) if nonzero.size else 0.0
        metrics["mean_observed_speed_mps"] = float(np.mean(speeds))
        metrics["std_observed_speed_mps"] = float(np.std(speeds))
        metrics["max_observed_abs_accel_mps2"] = float(np.max(np.abs(agent_states[:, :, 5])))
        metrics["max_observed_abs_omega_radps"] = float(np.max(np.abs(agent_states[:, :, 4])))
        metrics["vehicle_size_cells"] = float(config.vehicle_size_cells)
    if safety_summary:
        metrics.update(safety_summary)
    metrics["collision_count"] = int(compute_collision_count(agent_positions))
    metrics.update(compute_start_clearance_stats(starts, config))
    metrics.update(compute_clearance_stats(agent_positions, config.hard_clearance_cells))
    interp_positions = (
        agent_states[:, :, :2] if agent_states is not None else agent_positions.astype(np.float32)
    )
    metrics.update(compute_interpolated_clearance_stats(interp_positions, config))
    # Congestion KPIs (the project's headline objective): lower is better. Total
    # congestion is ~constant (each robot's additive kernel sums to a fixed value
    # regardless of position), so we measure *clustering* instead: peak overlap and
    # how many cell-timesteps exceed a single robot's center value (i.e. >=2 robots'
    # kernels stacked).
    metrics["congestion_peak"] = float(congestion_labels.max())
    metrics["congestion_overlap_cell_count"] = int(
        (congestion_labels > float(config.congestion_center_value)).sum()
    )

    if args.metrics_out:
        # Sweep mode: emit just the metrics, skip the full data save + figures.
        from pathlib import Path as _Path
        out = _Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"metrics -> {out.resolve()}")
        return

    run_id = datetime.now().strftime("%y%m%d_%H%M")
    save_config = config.replace(output_dir="data/online_runs")
    output_dir = save_results(
        paths, metrics, agent_positions, agent_states,
        occupancy_sequence, congestion_labels, env, save_config, run_id,
    )
    # Report the completed run before rendering: figures are best-effort and must
    # never discard the data we already computed.
    print(f"Saved online MAPF data to: {output_dir.resolve()}")
    print(json.dumps(metrics, indent=2))

    if args.no_figures:
        print("Figures skipped (--no-figures).")
        return

    figures_dir = FIGURES_DIR / "online_runs" / run_id
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        visualize_paths(env, paths, starts, goals, figures_dir)
        if config.save_animation:
            # Pass the real task_summary (not None) so the animation draws the same
            # route lines + pickup/delivery target markers as classical_mapf.
            animate_paths(env, paths, starts, goals, figures_dir, config,
                          agent_states=agent_states, task_summary=task_summary)
        print(f"Saved online MAPF figures to: {figures_dir.resolve()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] figure rendering skipped (matplotlib): {exc!r}")


if __name__ == "__main__":
    main()
