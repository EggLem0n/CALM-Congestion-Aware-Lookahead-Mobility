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
    get_neighbors,
    grid_paths_to_kinodynamic_states,
    normalize_point_groups,
    normalize_points,
    path_position_at,
    paths_to_agent_positions,
    prioritized_planning_repeated_tasks,
    sampled_segment_min_distance,
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
from .replanner import replan, select_congestion_skip_agents
from .world_state import World


def build_agent_ai_cost_weights(config, num_agents: int) -> np.ndarray:
    """Assign each AMR a stable congestion sensitivity.

    The values are evenly spread over a configured range, then optionally shuffled
    by seed. This avoids every AMR reacting identically to the same forecast
    heatmap while keeping the experiment reproducible.
    """
    base_weight = float(config.ai_cost_weight)
    min_multiplier = float(config.online_agent_ai_weight_min_multiplier)
    max_multiplier = float(config.online_agent_ai_weight_max_multiplier)
    if num_agents <= 0:
        return np.zeros(0, dtype=np.float32)
    if max_multiplier < min_multiplier:
        min_multiplier, max_multiplier = max_multiplier, min_multiplier
    if abs(max_multiplier - min_multiplier) <= 1e-9:
        multipliers = np.full(num_agents, min_multiplier, dtype=np.float32)
    else:
        multipliers = np.linspace(min_multiplier, max_multiplier, num_agents, dtype=np.float32)
    if bool(config.online_shuffle_agent_ai_weights):
        rng = np.random.default_rng(int(config.seed) + int(config.online_agent_ai_weight_seed_offset))
        rng.shuffle(multipliers)
    return (base_weight * multipliers).astype(np.float32)


def current_stationary_run(path) -> int:
    """Return how many consecutive executed steps this path has stayed still."""
    if len(path) < 2:
        return 0
    run = 0
    for idx in range(len(path) - 1, 0, -1):
        if path[idx] != path[idx - 1]:
            break
        run += 1
    return int(run)


def build_starvation_priority_order(world: World, threshold: int) -> tuple[list[int], list[int], list[int]]:
    """Move long-waiting AMRs to the front of the next prioritized re-plan.

    Short waits are normal yielding. Once an AMR has held the same cell for
    `threshold` steps, repeatedly planning it late can keep it trapped behind
    reservations. This gives the longest-waiting agents first claim on the next
    reservation table while leaving everyone else in the default stable order.
    """
    stationary_runs = [current_stationary_run(agent.path) for agent in world.agents]
    starved = [
        agent_id for agent_id, wait in enumerate(stationary_runs)
        if wait >= threshold
    ]
    starved.sort(key=lambda agent_id: (-stationary_runs[agent_id], agent_id))
    priority_order = starved + [
        agent_id for agent_id in range(world.num_agents)
        if agent_id not in set(starved)
    ]
    return priority_order, starved, stationary_runs


def guard_next_execution_step(
    world: World,
    committed,
    step_in_plan: int,
    priority_order: list[int],
    config,
) -> tuple[list[tuple[int, int]], int]:
    """Prevent continuous close-passes in the actual executed/saved trajectory.

    The MAPF reservation layer blocks same-cell and head-on swap conflicts, but
    two orthogonal moves can still pass too closely between cell centers. Before
    committing the next 1-second move, check all motion segments and hold the
    lower-priority AMR for this step when the continuous safe gap would be
    violated. This changes the recorded path, not just the visualization.
    """
    current = [agent.pos for agent in world.agents]
    proposed = [
        path_position_at(committed[agent_id], step_in_plan)
        for agent_id in range(world.num_agents)
    ]
    repaired, guard_events = repair_continuous_step(current, proposed, priority_order, config)
    for agent_id, position in enumerate(repaired):
        if step_in_plan < len(committed[agent_id]):
            committed[agent_id][step_in_plan] = position
    return repaired, guard_events


def step_is_safe_against_proposals(agent_id, candidate, current, proposed, safe_gap: float) -> bool:
    candidate = tuple(candidate)
    for other_id, other_next in enumerate(proposed):
        if other_id == agent_id:
            continue
        other_next = tuple(other_next)
        if candidate == other_next:
            return False
        if current[agent_id] == other_next and candidate == current[other_id] and candidate != current[agent_id]:
            return False
        if sampled_segment_min_distance(current[agent_id], candidate, current[other_id], other_next) < safe_gap:
            return False
    return True


def find_guard_reroute(agent_id, current, proposed, walkable_map, safe_gap: float, score_fn=None):
    if walkable_map is None:
        return None
    candidates = [
        (nx, ny)
        for nx, ny, _ in get_neighbors(current[agent_id][0], current[agent_id][1], 0, walkable_map)
    ]
    candidates = [cell for cell in unique_cells(candidates) if cell != current[agent_id]]
    safe_candidates = []
    for candidate in candidates:
        trial = list(proposed)
        trial[agent_id] = candidate
        if step_is_safe_against_proposals(agent_id, candidate, current, trial, safe_gap):
            safe_candidates.append(candidate)
    if not safe_candidates:
        return None
    if score_fn is None:
        return safe_candidates[0]
    return min(safe_candidates, key=lambda cell: score_fn(agent_id, cell))


def repair_continuous_step(
    current,
    proposed,
    priority_order: list[int],
    config,
    walkable_map=None,
    reroute_score_fn=None,
):
    current = [tuple(p) for p in current]
    proposed = [tuple(p) for p in proposed]
    num_agents = len(current)
    rank = {agent_id: order for order, agent_id in enumerate(priority_order)}
    blocked = set()
    guard_events = {
        "vertex": 0,
        "edge_swap": 0,
        "continuous_close_pass": 0,
        "vertex_reroute": 0,
    }
    safe_gap = float(config.continuous_safe_gap_cells)
    if safe_gap <= 0.0:
        return proposed, guard_events

    for _ in range(num_agents):
        changed = False
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if proposed[i] == proposed[j]:
                    conflict = True
                    cause = "vertex"
                elif current[i] == proposed[j] and proposed[i] == current[j] and current[i] != proposed[i]:
                    conflict = True
                    cause = "edge_swap"
                else:
                    distance = sampled_segment_min_distance(
                        current[i],
                        proposed[i],
                        current[j],
                        proposed[j],
                    )
                    conflict = distance < safe_gap
                    cause = "continuous_close_pass"
                if not conflict:
                    continue

                if i in blocked and j in blocked:
                    continue
                if i in blocked:
                    yield_id = j
                elif j in blocked:
                    yield_id = i
                else:
                    yield_id = i if rank.get(i, i) > rank.get(j, j) else j
                other_id = j if yield_id == i else i
                if proposed[yield_id] == current[yield_id] and proposed[other_id] != current[other_id]:
                    yield_id = other_id

                reroute = None
                if cause == "vertex":
                    reroute = find_guard_reroute(
                        yield_id,
                        current,
                        proposed,
                        walkable_map,
                        safe_gap,
                        reroute_score_fn,
                    )
                if reroute is not None:
                    proposed[yield_id] = reroute
                    guard_events["vertex_reroute"] += 1
                    changed = True
                elif proposed[yield_id] != current[yield_id]:
                    proposed[yield_id] = current[yield_id]
                    blocked.add(yield_id)
                    guard_events[cause] += 1
                    changed = True
        if not changed:
            break
    return proposed, guard_events


def flatten_point_groups(groups):
    points = []
    for group_points in groups.values():
        for point in group_points:
            if point not in points:
                points.append(point)
    return points


def nominal_goals_typed(task_assignments, visual_t: int):
    goals = []
    for assignments in task_assignments:
        active = None
        for assignment in assignments:
            if int(assignment.get("end_t", 0)) >= visual_t:
                active = assignment
                break
        if active is None and assignments:
            active = assignments[-1]
        if active is None:
            continue
        target = active.get("target")
        if isinstance(target, list) and len(target) >= 2:
            goals.append(((int(target[0]), int(target[1])), active.get("action") == "pickup"))
    return goals


def nominal_goals_typed_by_cursor(task_assignments, cursors):
    goals = []
    for agent_id, assignments in enumerate(task_assignments):
        visual_t = int(cursors[agent_id]) if agent_id < len(cursors) else 0
        active = None
        for assignment in assignments:
            if int(assignment.get("end_t", 0)) >= visual_t:
                active = assignment
                break
        if active is None and assignments:
            active = assignments[-1]
        if active is None:
            continue
        target = active.get("target")
        if isinstance(target, list) and len(target) >= 2:
            goals.append(((int(target[0]), int(target[1])), active.get("action") == "pickup"))
    return goals


def replay_assignments_on_executed_paths(executed_paths, nominal_assignments):
    completed_targets = []
    completed_deliveries = []
    replayed_assignments = []
    final_goals = []
    for agent_id, path in enumerate(executed_paths):
        assignments = nominal_assignments[agent_id] if agent_id < len(nominal_assignments) else []
        replayed = []
        cursor = 0
        target_count = 0
        delivery_count = 0
        for assignment in assignments:
            target = assignment.get("target")
            if not isinstance(target, list) or len(target) < 2:
                continue
            target_cell = (int(target[0]), int(target[1]))
            copied = dict(assignment)
            copied["start_t"] = int(cursor)
            copied["completed"] = False
            copied["end_t"] = int(len(path) - 1)
            for t in range(cursor, len(path)):
                if tuple(path[t]) == target_cell:
                    copied["end_t"] = int(t)
                    copied["completed"] = True
                    cursor = t
                    target_count += 1
                    if copied.get("action") == "delivery":
                        delivery_count += 1
                    break
            replayed.append(copied)
            if not copied["completed"]:
                break
        completed_targets.append(target_count)
        completed_deliveries.append(delivery_count)
        replayed_assignments.append(replayed)
        if replayed and not replayed[-1].get("completed", False):
            final_goals.append(replayed[-1]["target"])
        elif len(replayed) < len(assignments):
            next_target = assignments[len(replayed)].get("target")
            final_goals.append(next_target if isinstance(next_target, list) else list(path[-1]))
        else:
            final_goals.append(list(path[-1]))
    return {
        "completed_deliveries": completed_deliveries,
        "completed_targets": completed_targets,
        "total_completed_deliveries": int(sum(completed_deliveries)),
        "total_completed_targets": int(sum(completed_targets)),
        "final_goals": final_goals,
        "task_assignments": replayed_assignments,
    }


def unique_cells(cells):
    seen = set()
    out = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def advance_nominal_cursor(path, cursor: int, position, lookahead: int) -> int:
    """Advance one agent's nominal-path cursor using its actual executed cell.

    The old nominal-MPC loop used the global simulation time as the path index.
    Once an AMR waited or made a detour, that made it chase waypoints far ahead
    of its physical progress. This cursor keeps each AMR attached to the nearest
    reachable forward point on its own nominal path.
    """
    max_index = len(path) - 1
    start = max(0, min(int(cursor), max_index))
    end = min(max_index, start + max(1, int(lookahead)))
    position = tuple(position)
    best = start
    best_score = (abs(position[0] - path[start][0]) + abs(position[1] - path[start][1]), 0)
    for idx in range(start + 1, end + 1):
        point = path_position_at(path, idx)
        distance = abs(position[0] - point[0]) + abs(position[1] - point[1])
        score = (distance, -idx)
        if score < best_score:
            best = idx
            best_score = score
    if best_score[0] == 0:
        return best
    # If the AMR is off the path after a detour, allow slow forward progress
    # toward the nearest future path point but never jump the cursor far ahead.
    return min(best, start + 1)


def choose_nominal_mpc_step(
    agent_id: int,
    nominal_cursor: int,
    current,
    nominal_paths,
    accepted_positions,
    accepted_edges,
    accepted_segments,
    walkable_map,
    congestion_cost,
    agent_ai_weight: float,
    config,
):
    nominal_path = nominal_paths[agent_id]
    nominal_next = path_position_at(nominal_path, nominal_cursor + 1)
    rejoin_lookahead = max(1, int(config.online_nominal_rejoin_lookahead))
    rejoin_cells = [
        path_position_at(nominal_path, min(nominal_cursor + k, config.max_time))
        for k in range(1, rejoin_lookahead + 1)
    ]
    rejoin_goal = rejoin_cells[-1]
    # Only execute physically reachable 1-step moves. The nominal path may be
    # several cells away after a local detour or wait; adding that far waypoint
    # directly would create diagonal/wall-crossing jumps in the saved trajectory.
    candidates = [
        (nx, ny)
        for nx, ny, _ in get_neighbors(current[0], current[1], nominal_cursor, walkable_map)
    ]
    candidates = unique_cells(candidates)

    safe_gap = float(config.continuous_safe_gap_cells)

    def is_safe(candidate) -> bool:
        if candidate in accepted_positions:
            return False
        if (candidate, current) in accepted_edges:
            return False
        for other_prev, other_next in accepted_segments:
            if sampled_segment_min_distance(current, candidate, other_prev, other_next) < safe_gap:
                return False
        return True

    safe_candidates = [candidate for candidate in candidates if is_safe(candidate)]
    forced_wait = False
    if not safe_candidates:
        safe_candidates = [current]
        forced_wait = True

    cost_index = min(1, congestion_cost.shape[0] - 1) if congestion_cost.size else 0

    def score(candidate) -> float:
        nominal_dist = min(abs(candidate[0] - p[0]) + abs(candidate[1] - p[1]) for p in rejoin_cells)
        progress_dist = abs(candidate[0] - rejoin_goal[0]) + abs(candidate[1] - rejoin_goal[1])
        congestion = 0.0
        if congestion_cost.size:
            x, y = candidate
            congestion = float(congestion_cost[cost_index, y, x])
        wait_penalty = float(config.online_mpc_wait_penalty) if candidate == current and nominal_next != current else 0.0
        return (
            float(config.online_mpc_nominal_weight) * nominal_dist
            + float(config.online_mpc_goal_progress_weight) * progress_dist
            + float(agent_ai_weight) * congestion
            + wait_penalty
        )

    chosen = min(safe_candidates, key=score)
    if forced_wait:
        cause = "no_safe_candidate"
    elif chosen == current and nominal_next != current:
        cause = "mpc_cost_wait"
    else:
        cause = "move"
    return chosen, cause


def run_nominal_mpc_online(
    args,
    config,
    env,
    walkable_map,
    obstacle_map,
    starts,
    pickup_groups,
    delivery_groups,
    predictor,
    agent_ai_cost_weights,
    congestion_skip_agents,
):
    H, W = walkable_map.shape[:2]
    max_time = int(config.max_time)
    replan_every = max(1, int(config.online_replan_every))
    zero_cost = np.zeros((1, H, W), dtype=np.float32)

    print("Building Classical nominal paths for local-MPC online execution...")
    nominal_config = config.replace(use_ai_congestion_cost=False)
    nominal_paths, nominal_summary = prioritized_planning_repeated_tasks(
        starts,
        flatten_point_groups(pickup_groups),
        flatten_point_groups(delivery_groups),
        walkable_map,
        zero_cost,
        nominal_config,
        pickup_point_groups=pickup_groups,
        delivery_point_groups=delivery_groups,
    )
    task_assignments = nominal_summary.get("task_assignments", [])

    frame_builder = FrameBuilder(starts, obstacle_map)
    rng_dummy = None
    del rng_dummy
    t_in, t_out = predictor.t_in, predictor.t_out
    history_frame = frame_builder.frame(
        np.asarray(starts, dtype=np.int32),
        nominal_goals_typed(task_assignments, 0),
    )
    from collections import deque

    history = deque([history_frame], maxlen=max(1, t_in))
    executed_paths = [[tuple(start)] for start in starts]
    current_positions = [tuple(start) for start in starts]
    nominal_cursors = [0 for _ in starts]
    current_cost = zero_cost
    local_wait_causes = {
        "mpc_cost_wait": 0,
        "no_safe_candidate": 0,
        "vertex_guard": 0,
        "edge_swap_guard": 0,
        "continuous_close_pass_guard": 0,
    }
    local_reroute_events = 0
    congestion_skip_agents = congestion_skip_agents or set()
    effective_base_weights = agent_ai_cost_weights.copy()
    for agent_id in congestion_skip_agents:
        effective_base_weights[agent_id] = 0.0

    print(
        f"Running nominal-MPC online loop: {len(starts)} AMRs, {max_time} steps, "
        f"congestion forecast every {replan_every}s"
    )
    progress_every = max(1, max_time // 20)
    for t in range(max_time):
        if t % replan_every == 0:
            if bool(config.online_warmup_zero_cost) and len(history) < t_in:
                current_cost = zero_cost
            else:
                enc = build_enc(list(history), t_in)
                dec = build_dec(history[-1], t_out)
                pred = predictor.predict(enc, dec)
                pred = pred / max(1e-6, float(config.congestion_center_value))
                current_cost = np.concatenate([pred[:1], pred], axis=0)

        priority_order, starved_agents, _ = build_starvation_priority_order(
            type("WorldView", (), {
                "agents": [type("AgentView", (), {"path": p}) for p in executed_paths],
                "num_agents": len(executed_paths),
            })(),
            max(1, int(config.online_starvation_priority_threshold)),
        )
        effective_weights = effective_base_weights.copy()
        for agent_id in starved_agents:
            effective_weights[agent_id] *= max(0.0, float(config.online_starvation_ai_weight_multiplier))

        next_positions = [None for _ in current_positions]
        accepted_positions = set()
        accepted_edges = set()
        accepted_segments = []
        for agent_id in priority_order:
            current = current_positions[agent_id]
            chosen, wait_cause = choose_nominal_mpc_step(
                agent_id,
                nominal_cursors[agent_id],
                current,
                nominal_paths,
                accepted_positions,
                accepted_edges,
                accepted_segments,
                walkable_map,
                current_cost,
                float(effective_weights[agent_id]),
                config,
            )
            next_positions[agent_id] = chosen
            accepted_positions.add(chosen)
            accepted_edges.add((current, chosen))
            accepted_segments.append((current, chosen))
            if wait_cause in local_wait_causes:
                local_wait_causes[wait_cause] += 1

        def guard_reroute_score(agent_id, candidate):
            nominal_path = nominal_paths[agent_id]
            cursor = nominal_cursors[agent_id]
            lookahead = max(1, int(config.online_nominal_rejoin_lookahead))
            rejoin_cells = [
                path_position_at(nominal_path, min(cursor + k, max_time))
                for k in range(1, lookahead + 1)
            ]
            rejoin_goal = rejoin_cells[-1]
            nominal_dist = min(abs(candidate[0] - p[0]) + abs(candidate[1] - p[1]) for p in rejoin_cells)
            progress_dist = abs(candidate[0] - rejoin_goal[0]) + abs(candidate[1] - rejoin_goal[1])
            congestion = 0.0
            if current_cost.size:
                cost_index = min(1, current_cost.shape[0] - 1)
                x, y = candidate
                congestion = float(current_cost[cost_index, y, x])
            return (
                float(config.online_mpc_nominal_weight) * nominal_dist
                + float(config.online_mpc_goal_progress_weight) * progress_dist
                + float(effective_weights[agent_id]) * congestion
            )

        next_positions, repaired_events = repair_continuous_step(
            current_positions,
            next_positions,
            priority_order,
            config,
            walkable_map=walkable_map,
            reroute_score_fn=guard_reroute_score,
        )
        local_reroute_events += int(repaired_events["vertex_reroute"])
        local_wait_causes["vertex_guard"] += int(repaired_events["vertex"])
        local_wait_causes["edge_swap_guard"] += int(repaired_events["edge_swap"])
        local_wait_causes["continuous_close_pass_guard"] += int(repaired_events["continuous_close_pass"])
        current_positions = [tuple(p) for p in next_positions]
        for agent_id, pos in enumerate(current_positions):
            executed_paths[agent_id].append(pos)
            nominal_cursors[agent_id] = advance_nominal_cursor(
                nominal_paths[agent_id],
                nominal_cursors[agent_id],
                pos,
                int(config.online_nominal_rejoin_lookahead),
            )
        history.append(
            frame_builder.frame(
                np.asarray(current_positions, dtype=np.int32),
                nominal_goals_typed_by_cursor(task_assignments, nominal_cursors),
            )
        )
        if (t + 1) % progress_every == 0 or t + 1 == max_time:
            print(f"  step {t + 1}/{max_time} | local_waits={sum(local_wait_causes.values())}", flush=True)

    if config.use_kinodynamic_motion:
        agent_states = grid_paths_to_kinodynamic_states(executed_paths, walkable_map, config)
        agent_states, safety_summary = apply_proximity_safety_controller(agent_states, config)
        agent_positions = states_to_agent_positions(agent_states)
    else:
        agent_states = None
        safety_summary = None
        agent_positions = paths_to_agent_positions(executed_paths, max_time)

    task_summary = {
        **nominal_summary,
        **replay_assignments_on_executed_paths(executed_paths, task_assignments),
    }
    goals = [tuple(goal[:2]) for goal in task_summary["final_goals"]]
    occupancy_sequence = build_occupancy_sequence(agent_positions, H, W)
    congestion_labels = build_additive_congestion_label_sequence(
        agent_positions,
        H,
        W,
        center_value=config.congestion_center_value,
        step_value=config.congestion_step_value,
    )
    metrics = compute_metrics(executed_paths, starts, goals, task_summary=task_summary, max_t=max_time)
    metrics["mapf_solver"] = "online_nominal_path_mpc_convlstm"
    metrics["online_replan_every"] = replan_every
    metrics["online_plan_horizon"] = 0
    metrics["online_nominal_mpc_mode"] = True
    metrics["online_nominal_final_cursors"] = [int(v) for v in nominal_cursors]
    metrics["online_nominal_local_wait_events"] = int(sum(local_wait_causes.values()))
    metrics["online_nominal_vertex_reroute_events"] = int(local_reroute_events)
    metrics["online_nominal_wait_due_to_mpc_cost"] = int(local_wait_causes["mpc_cost_wait"])
    metrics["online_nominal_wait_due_to_no_safe_candidate"] = int(local_wait_causes["no_safe_candidate"])
    metrics["online_nominal_wait_due_to_vertex_guard"] = int(local_wait_causes["vertex_guard"])
    metrics["online_nominal_wait_due_to_edge_swap_guard"] = int(local_wait_causes["edge_swap_guard"])
    metrics["online_nominal_wait_due_to_continuous_close_pass_guard"] = int(
        local_wait_causes["continuous_close_pass_guard"]
    )
    metrics["online_congestion_skip_fraction"] = float(config.online_congestion_skip_fraction)
    metrics["online_congestion_skip_count"] = len(congestion_skip_agents)
    metrics["online_agent_ai_weight_min"] = float(agent_ai_cost_weights.min())
    metrics["online_agent_ai_weight_max"] = float(agent_ai_cost_weights.max())
    metrics["online_agent_ai_weight_mean"] = float(agent_ai_cost_weights.mean())
    metrics["online_agent_ai_weights"] = [float(v) for v in agent_ai_cost_weights]
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
    interp_positions = agent_states[:, :, :2] if agent_states is not None else agent_positions.astype(np.float32)
    metrics.update(compute_interpolated_clearance_stats(interp_positions, config))
    metrics["congestion_peak"] = float(congestion_labels.max())
    metrics["congestion_overlap_cell_count"] = int(
        (congestion_labels > float(config.congestion_center_value)).sum()
    )

    if args.metrics_out:
        from pathlib import Path as _Path
        out = _Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"metrics -> {out.resolve()}")
        return

    run_id = args.run_id or datetime.now().strftime("%y%m%d_%H%M")
    save_config = config.replace(output_dir="data/online_runs")
    output_dir = save_results(
        executed_paths,
        metrics,
        agent_positions,
        agent_states,
        occupancy_sequence,
        congestion_labels,
        env,
        save_config,
        run_id,
    )
    print(f"Saved nominal-MPC online MAPF data to: {output_dir.resolve()}")
    print(json.dumps(metrics, indent=2))
    if args.no_figures:
        print("Figures skipped (--no-figures).")
        return

    figures_dir = FIGURES_DIR / "online_runs" / run_id
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        visualize_paths(env, executed_paths, starts, goals, figures_dir)
        if config.save_animation:
            animate_paths(
                env,
                executed_paths,
                starts,
                goals,
                figures_dir,
                config,
                agent_states=agent_states,
                task_summary=task_summary,
            )
        print(f"Saved nominal-MPC online MAPF figures to: {figures_dir.resolve()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] figure rendering skipped (matplotlib): {exc!r}")


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
    parser.add_argument("--congestion-skip-fraction", type=float, default=None,
                        help="Override config.online_congestion_skip_fraction: fraction of AMRs (0..1) "
                             "that ignore the AI congestion term in A* (e.g. 0.3 = 30%% pure shortest-path).")
    parser.add_argument("--no-animation", action="store_true", help="Skip the GIF animation.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip all matplotlib output (PNG + GIF). Use on headless machines.")
    parser.add_argument("--metrics-out", default=None,
                        help="Sweep mode: write the metrics JSON here and skip the full save + figures.")
    parser.add_argument("--run-id", default=None, help="Run dir / figures name (default: timestamp).")
    parser.add_argument("--save-animation", action="store_true",
                        help="Force the animation on (overrides config.save_animation).")
    parser.add_argument("--subframes", type=int, default=None,
                        help="Override animation_subframes (3=10x speed, 30=real-time at 30 fps).")
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
    if args.congestion_skip_fraction is not None:
        overrides["online_congestion_skip_fraction"] = args.congestion_skip_fraction
    if args.no_animation:
        overrides["save_animation"] = False
    if args.save_animation:
        overrides["save_animation"] = True
    if args.subframes is not None:
        overrides["animation_subframes"] = args.subframes
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
    agent_ai_cost_weights = build_agent_ai_cost_weights(config, len(starts))
    warmup_zero = bool(config.online_warmup_zero_cost)

    frame_builder = FrameBuilder(starts, obstacle_map)
    world = World(starts, t_in)
    rng = np.random.default_rng(config.seed + 1000)

    # A fixed, seeded subset of AMRs plans without the congestion term (pure
    # shortest path) so they keep flowing through corridors instead of the whole
    # fleet detouring around the same forecast hot-spot. 0.0 = everyone is
    # congestion-aware (original behaviour). Seed offset keeps this stream
    # independent of the planning rng above.
    skip_fraction = float(config.online_congestion_skip_fraction)
    congestion_skip_agents = select_congestion_skip_agents(
        world.num_agents, skip_fraction, config.seed + 2000
    )
    if congestion_skip_agents:
        print(f"Congestion term OFF for {len(congestion_skip_agents)}/{world.num_agents} AMRs "
              f"({skip_fraction:.0%} pure shortest-path); the rest stay congestion-aware.")
    print(
        "Agent AI congestion weights: "
        f"min={float(agent_ai_cost_weights.min()):.3f}, "
        f"max={float(agent_ai_cost_weights.max()):.3f}, "
        f"mean={float(agent_ai_cost_weights.mean()):.3f}"
    )

    if bool(config.online_nominal_mpc_mode):
        return run_nominal_mpc_online(
            args,
            config,
            env,
            walkable_map,
            obstacle_map,
            starts,
            pickup_groups,
            delivery_groups,
            predictor,
            agent_ai_cost_weights,
            congestion_skip_agents,
        )

    # Seed the history with the state at t=0.
    world.push_frame(frame_builder.frame(world.positions_array(), world.goals_typed()))

    print(f"Running online loop: {len(starts)} AMRs, {max_time} steps, re-plan every {replan_every}s")
    committed = None
    step_in_plan = 0
    replans = 0
    starvation_threshold = max(1, int(config.online_starvation_priority_threshold))
    starvation_ai_multiplier = max(0.0, float(config.online_starvation_ai_weight_multiplier))
    starvation_priority_events = 0
    starvation_agent_hits = [0 for _ in range(world.num_agents)]
    max_observed_stationary_run = 0
    execution_guard_events = {
        "vertex": 0,
        "edge_swap": 0,
        "continuous_close_pass": 0,
        "vertex_reroute": 0,
    }
    active_priority_order = list(range(world.num_agents))
    wait_diagnostics = {
        "planned_wait_total": 0,
        "wait_due_to_safe_extension": 0,
        "wait_due_to_no_legal_move": 0,
        "wait_due_to_vertex_reservation": 0,
        "wait_due_to_hard_clearance": 0,
        "wait_due_to_edge_conflict": 0,
        "wait_due_to_continuous_conflict": 0,
        "wait_with_legal_move_available": 0,
        "wait_due_to_ai_congestion_cost": 0,
        "wait_due_to_soft_or_path_cost": 0,
    }
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
            priority_order, starved_agents, stationary_runs = build_starvation_priority_order(
                world,
                starvation_threshold,
            )
            active_priority_order = priority_order
            if stationary_runs:
                max_observed_stationary_run = max(max_observed_stationary_run, max(stationary_runs))
            effective_ai_weights = agent_ai_cost_weights
            if starved_agents:
                starvation_priority_events += 1
                for agent_id in starved_agents:
                    starvation_agent_hits[agent_id] += 1
                effective_ai_weights = agent_ai_cost_weights.copy()
                for agent_id in starved_agents:
                    effective_ai_weights[agent_id] *= starvation_ai_multiplier
            committed = replan(
                world, cost, pickup_groups, delivery_groups,
                walkable_map, planning_config, rng,
                priority_order=priority_order,
                congestion_skip_agents=congestion_skip_agents,
                agent_ai_cost_weights=effective_ai_weights,
                wait_diagnostics=wait_diagnostics,
            )
            step_in_plan = 0
            replans += 1

        step_in_plan += 1
        guarded_next_positions, guard_events = guard_next_execution_step(
            world,
            committed,
            step_in_plan,
            active_priority_order,
            config,
        )
        for key, value in guard_events.items():
            execution_guard_events[key] += int(value)
        for agent_id, agent in enumerate(world.agents):
            new_pos = guarded_next_positions[agent_id]
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
    metrics["online_congestion_skip_fraction"] = skip_fraction
    metrics["online_congestion_skip_count"] = len(congestion_skip_agents)
    metrics["online_plan_horizon"] = int(planning_config.max_time)
    metrics["online_agent_ai_weight_min"] = float(agent_ai_cost_weights.min())
    metrics["online_agent_ai_weight_max"] = float(agent_ai_cost_weights.max())
    metrics["online_agent_ai_weight_mean"] = float(agent_ai_cost_weights.mean())
    metrics["online_agent_ai_weights"] = [float(v) for v in agent_ai_cost_weights]
    metrics["online_starvation_priority_threshold"] = starvation_threshold
    metrics["online_starvation_ai_weight_multiplier"] = starvation_ai_multiplier
    metrics["online_starvation_priority_events"] = int(starvation_priority_events)
    metrics["online_starvation_agent_hit_count"] = int(sum(1 for hit in starvation_agent_hits if hit > 0))
    metrics["online_starvation_agent_hits"] = [int(hit) for hit in starvation_agent_hits]
    metrics["online_max_observed_stationary_run"] = int(max_observed_stationary_run)
    metrics["online_execution_guard_vertex_events"] = int(execution_guard_events["vertex"])
    metrics["online_execution_guard_vertex_reroute_events"] = int(execution_guard_events["vertex_reroute"])
    metrics["online_execution_guard_edge_swap_events"] = int(execution_guard_events["edge_swap"])
    metrics["online_execution_guard_continuous_close_pass_events"] = int(
        execution_guard_events["continuous_close_pass"]
    )
    metrics["online_execution_continuous_guard_events"] = int(sum(execution_guard_events.values()))
    for key, value in wait_diagnostics.items():
        metrics[f"online_{key}"] = int(value)
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

    run_id = args.run_id or datetime.now().strftime("%y%m%d_%H%M")
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
