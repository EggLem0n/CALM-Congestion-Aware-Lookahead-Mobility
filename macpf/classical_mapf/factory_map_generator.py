from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ==========================================================
# Automotive Assembly Line Parts-Supply 2D Grid Map
# for AMR / MAPF / AI Congestion Cost Research
# ==========================================================
# Coordinate convention:
# - factory_map[y, x]
# - x: left -> right
# - y: top -> bottom

H, W = 50, 80

# Cell codes
FREE = 0
OBSTACLE = 1
PICKUP = 2
DELIVERY = 3
CHARGING = 4
INSPECTION = 6
BUFFER = 7
MACHINE = 8

LABELS = {
    FREE: "Free / Road",
    OBSTACLE: "Obstacle / Conveyor / Equipment",
    PICKUP: "Parts Warehouse / Pickup",
    DELIVERY: "Line-side Delivery Zone",
    CHARGING: "Charging / Waiting Zone",
    INSPECTION: "Inspection Zone",
    BUFFER: "Vehicle / Parts Buffer",
    MACHINE: "Sequencing / Kitting / Supermarket",
}

COLORS = {
    FREE: "#f2f2f2",
    OBSTACLE: "#222222",
    PICKUP: "#6fa8dc",
    DELIVERY: "#93c47d",
    CHARGING: "#76a5af",
    INSPECTION: "#ffd966",
    BUFFER: "#b6d7a8",
    MACHINE: "#f9cb9c",
}


def build_factory_map():
    """
    Build an automotive final-assembly parts-supply grid map.

    Main modeling intent
    --------------------
    - Long black obstacles inside the assembly area represent conveyor belts,
      vehicle bodies, fixtures, and line equipment.
    - AMRs do not enter the conveyor region. They deliver parts to line-side
      delivery cells distributed along the conveyor.
    - The central line-side supply lane is modeled as a normal walkable road.
      Congestion is not manually labeled; it must be learned from MAPF simulation logs.
    """
    factory_map = np.full((H, W), OBSTACLE, dtype=np.int8)

    def rect(x1, y1, x2, y2, value):
        factory_map[y1:y2, x1:x2] = value

    # -------------------------
    # 1. Roads / aisles
    # -------------------------
    # Outer loop road
    rect(2, 2, 78, 5, FREE)
    rect(2, 45, 78, 48, FREE)
    rect(2, 2, 5, 48, FREE)
    rect(75, 2, 78, 48, FREE)

    # Main internal logistics roads
    rect(5, 10, 75, 14, FREE)
    rect(5, 31, 75, 35, FREE)
    rect(36, 5, 41, 45, FREE)
    rect(55, 5, 59, 45, FREE)
    rect(20, 5, 24, 45, FREE)

    # Connecting roads
    rect(24, 19, 36, 22, FREE)
    rect(41, 19, 55, 22, FREE)
    rect(41, 26, 55, 29, FREE)
    rect(59, 17, 75, 20, FREE)
    rect(59, 28, 75, 31, FREE)
    rect(5, 38, 36, 41, FREE)
    rect(41, 38, 75, 41, FREE)

    # Additional internal bypasses
    rect(11, 22, 15, 31, FREE)
    rect(64, 20, 68, 28, FREE)
    rect(30, 14, 33, 19, FREE)
    rect(46, 14, 49, 19, FREE)
    rect(46, 29, 49, 31, FREE)

    # -------------------------
    # 2. Functional zones
    # -------------------------
    # Parts warehouse / material storage
    rect(6, 5, 18, 10, PICKUP)
    rect(6, 14, 18, 22, PICKUP)
    rect(6, 22, 11, 31, PICKUP)

    # Incoming inspection
    rect(25, 5, 35, 10, INSPECTION)
    rect(25, 14, 30, 19, INSPECTION)

    # Sequencing / kitting / parts supermarket areas
    rect(8, 35, 20, 45, MACHINE)
    rect(24, 23, 36, 31, MACHINE)
    rect(24, 35, 36, 45, MACHINE)

    # -----------------------------------------------------------
    # 3. Automotive assembly line supply area - main modification
    # -----------------------------------------------------------
    # Line-side delivery zones along the conveyor belts
    rect(37, 14, 75, 20, DELIVERY)   # Upper line-side delivery band
    rect(37, 27, 75, 33, DELIVERY)   # Lower line-side delivery band

    # Central line-side supply lane
    # This is a normal walkable road, not a manually labeled bottleneck.
    rect(37, 21, 75, 26, FREE)

    # Long assembly conveyor / vehicle body / fixture obstacles
    rect(38, 16, 74, 18, OBSTACLE)   # Long Conveyor Belt A
    rect(38, 29, 74, 31, OBSTACLE)   # Long Conveyor Belt B

    # Small gaps near the ends remain walkable through surrounding line-side lanes,
    # not through the conveyor itself.

    # Parts and final inspection
    rect(60, 5, 73, 10, INSPECTION)
    rect(68, 20, 75, 28, INSPECTION)

    # Restore the 1-cell-high line-side delivery strip above Long Conveyor Belt B.
    # This area is part of the conveyor-side delivery band, not the final inspection area.
    rect(68, 27, 75, 28, DELIVERY)

    # Completed vehicle buffer / shipping side
    rect(60, 35, 75, 45, BUFFER)

    # Buffer zones
    rect(59, 41, 74, 45, BUFFER)
    rect(13, 26, 20, 31, BUFFER)

    # Charging / waiting zones
    rect(6, 41, 18, 45, CHARGING)
    rect(41, 41, 53, 45, CHARGING)

    # Previously manually assigned bottleneck / one-way candidate areas
    # are kept as normal walkable roads. Congestion must be derived from
    # classical MAPF simulation logs and AI-predicted congestion heatmaps.
    rect(36, 22, 41, 26, FREE)
    rect(55, 20, 59, 27, FREE)
    rect(20, 14, 24, 22, FREE)
    rect(36, 35, 41, 38, FREE)

    # -------------------------
    # 4. Internal obstacles
    # -------------------------
    # Warehouse racks
    for x in [8, 11, 14]:
        rect(x, 6, x + 1, 9, OBSTACLE)
    for x in [8, 11, 14]:
        rect(x, 15, x + 1, 21, OBSTACLE)

    # Kitting / supermarket equipment
    for x, y in [(10, 37), (15, 37), (27, 25), (32, 25), (27, 38), (32, 38)]:
        rect(x, y, x + 3, y + 2, OBSTACLE)

    # NOTE: Previous short assembly equipment obstacles were removed.
    # The assembly line is now represented by long conveyor obstacles above.

    # Inspection / buffer equipment
    for x, y in [(62, 6), (67, 6), (70, 22), (62, 37), (67, 37), (72, 37)]:
        rect(x, y, x + 2, y + 2, OBSTACLE)

    # Charging equipment
    for x in [8, 11, 14, 43, 46, 49]:
        rect(x, 43, x + 1, 45, OBSTACLE)

    # -------------------------
    # 5. MAPF scenario coordinates
    # -------------------------
    # Pickup points are spread across the blue parts-warehouse / storage zones.
    # Coordinates avoid warehouse rack obstacle cells.
    warehouse_pickup_points = [
        (6, 6), (10, 7), (13, 8), (16, 6), (17, 8),
        (7, 16), (10, 18), (13, 20), (16, 15), (17, 20),
        (6, 23), (9, 24), (7, 27), (8, 29), (10, 30),
    ]
    inspection_pickup_points = [
        (30, 6),   # Incoming inspection block A
        (29, 18),  # Incoming inspection block B
        (66, 9),   # Parts inspection block
        (74, 21),  # Final inspection block
    ]
    kitting_supermarket_pickup_points = [
        (19, 35),  # Parts supermarket block
        (35, 23),  # Sequencing / kitting block
        (24, 44),  # Parts supermarket block
    ]
    buffer_pickup_points = [(74, 36)]
    pickup_points = (
        warehouse_pickup_points
        + inspection_pickup_points
        + kitting_supermarket_pickup_points
        + buffer_pickup_points
    )

    # Delivery points are distributed along both conveyor belts.
    # Each point represents a line-side supply position for different parts/modules.
    upper_delivery_points = [(x, 15) for x in range(38, 75, 4)] + [(x, 18) for x in range(40, 75, 4)]
    lower_delivery_points = [(x, 28) for x in range(38, 75, 4)] + [(x, 32) for x in range(40, 75, 4)]

    # Additional drop-off points model broader in-factory AMR missions:
    # inspection handoff, sequencing/kitting/supermarket replenishment, and buffer staging.
    inspection_delivery_points = [(25, 7), (34, 8), (60, 7), (72, 8), (70, 25)]
    kitting_supermarket_delivery_points = [
        (24, 24), (30, 28), (35, 30),
        (8, 36), (18, 39), (19, 44),
        (24, 36), (30, 42), (35, 44),
    ]
    buffer_delivery_points = [(60, 36), (65, 40), (70, 41), (74, 44), (59, 42)]
    delivery_points = (
        upper_delivery_points
        + lower_delivery_points
        + inspection_delivery_points
        + kitting_supermarket_delivery_points
        + buffer_delivery_points
    )

    inspection_points = [(25, 7), (34, 8), (60, 7), (72, 8), (70, 25)]
    charging_points = [(6, 42), (17, 42), (41, 42), (52, 42)]

    start_candidates = [
        (6, 42), (8, 42), (10, 42), (12, 42), (14, 42), (16, 42),
        (41, 42), (43, 42), (45, 42), (47, 42), (49, 42), (51, 42),
        (3, 10), (3, 31)
    ]
    walkable_map = (factory_map != OBSTACLE).astype(np.int8)
    obstacle_map = (factory_map == OBSTACLE).astype(np.int8)

    scenario_points = {
        "pickup_points": pickup_points,
        "delivery_points": delivery_points,
        "inspection_points": inspection_points,
        "charging_points": charging_points,
        "start_candidates": start_candidates,
    }
    invalid_points = {
        name: [
            (x, y)
            for x, y in points
            if not (0 <= x < W and 0 <= y < H and bool(walkable_map[y, x]))
        ]
        for name, points in scenario_points.items()
    }
    invalid_points = {name: points for name, points in invalid_points.items() if points}
    if invalid_points:
        raise ValueError(f"Scenario points overlap obstacles or map bounds: {invalid_points}")

    expected_point_zones = {
        "warehouse_pickup_points": {PICKUP},
        "inspection_pickup_points": {INSPECTION},
        "kitting_supermarket_pickup_points": {MACHINE},
        "buffer_pickup_points": {BUFFER},
        "line_side_delivery_points": {DELIVERY},
        "inspection_delivery_points": {INSPECTION},
        "kitting_supermarket_delivery_points": {MACHINE},
        "buffer_delivery_points": {BUFFER},
    }
    zone_checked_points = {
        "warehouse_pickup_points": warehouse_pickup_points,
        "inspection_pickup_points": inspection_pickup_points,
        "kitting_supermarket_pickup_points": kitting_supermarket_pickup_points,
        "buffer_pickup_points": buffer_pickup_points,
        "line_side_delivery_points": upper_delivery_points + lower_delivery_points,
        "inspection_delivery_points": inspection_delivery_points,
        "kitting_supermarket_delivery_points": kitting_supermarket_delivery_points,
        "buffer_delivery_points": buffer_delivery_points,
    }
    wrong_zone_points = {}
    for name, points in zone_checked_points.items():
        expected_zones = expected_point_zones[name]
        mismatches = [
            (x, y, int(factory_map[y, x]))
            for x, y in points
            if int(factory_map[y, x]) not in expected_zones
        ]
        if mismatches:
            wrong_zone_points[name] = mismatches
    if wrong_zone_points:
        raise ValueError(f"Scenario points are assigned to the wrong zone type: {wrong_zone_points}")

    min_pickup_delivery_separation = 4
    close_pickup_delivery_pairs = [
        (pickup, delivery, abs(pickup[0] - delivery[0]) + abs(pickup[1] - delivery[1]))
        for pickup in pickup_points
        for delivery in delivery_points
        if abs(pickup[0] - delivery[0]) + abs(pickup[1] - delivery[1]) < min_pickup_delivery_separation
    ]
    if close_pickup_delivery_pairs:
        raise ValueError(
            "Pickup points are too close to delivery points: "
            f"{close_pickup_delivery_pairs}"
        )

    return {
        "factory_map": factory_map,
        "walkable_map": walkable_map,
        "obstacle_map": obstacle_map,
        "pickup_points": pickup_points,
        "pickup_point_groups": {
            "warehouse": warehouse_pickup_points,
            "inspection": inspection_pickup_points,
            "kitting_supermarket": kitting_supermarket_pickup_points,
            "buffer": buffer_pickup_points,
        },
        "delivery_points": delivery_points,
        "delivery_point_groups": {
            "line_side": upper_delivery_points + lower_delivery_points,
            "inspection": inspection_delivery_points,
            "kitting_supermarket": kitting_supermarket_delivery_points,
            "buffer": buffer_delivery_points,
        },
        "inspection_points": inspection_points,
        "charging_points": charging_points,
        "start_candidates": start_candidates,
        "labels": LABELS,
        "colors": COLORS,
    }


def visualize_factory_map(data=None, save_path="factory_map_v3_oneway_preview.png", show=True):
    """Visualize the automotive assembly parts-supply map."""
    if data is None:
        data = build_factory_map()

    factory_map = data["factory_map"]

    rgb = np.zeros((H, W, 3), dtype=float)
    for code, color in COLORS.items():
        rgb[factory_map == code] = to_rgb(color)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(rgb, origin="upper")

    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.25, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(np.arange(0, W, 5))
    ax.set_yticks(np.arange(0, H, 5))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    ax.set_title("Automotive Assembly Line Parts-Supply Grid Map v3 - No Manual Bottleneck / One-way", fontsize=16, pad=20)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")

    zone_labels = [
        (12, 7, "Parts\nWarehouse"),
        (30, 7, "Incoming\nInspection"),
        (14, 18, "Material\nStorage"),
        (30, 27, "Sequencing /\nKitting"),
        (30, 40, "Parts\nSupermarket"),
        (56, 17, "Long Conveyor\nBelt A"),
        (56, 24, "Line-side\nSupply Road"),
        (56, 30, "Long Conveyor\nBelt B"),
        (66, 7, "Parts\nInspection"),
        (72, 24, "Final\nInspection"),
        (67, 40, "Vehicle / Parts\nBuffer"),
        (12, 43, "Charging\nStation A"),
        (47, 43, "Charging\nStation B"),
    ]

    for x, y, text in zone_labels:
        ax.text(
            x, y, text,
            ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, linewidth=0.5)
        )

    def scatter_points(points, marker, label, size=80):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, marker=marker, s=size, edgecolors="black", linewidths=0.8, label=label)

    scatter_points(data["pickup_points"], "P", "Pickup Points")
    scatter_points(data["delivery_points"], "D", "Delivery Points", size=55)
    scatter_points(data["charging_points"], "^", "Charging Points")

    legend_elements = [
        Patch(facecolor=COLORS[FREE], edgecolor="black", label="Free / Road"),
        Patch(facecolor=COLORS[OBSTACLE], edgecolor="black", label="Obstacle / Conveyor / Equipment"),
        Patch(facecolor=COLORS[PICKUP], edgecolor="black", label="Parts Warehouse / Pickup"),
        Patch(facecolor=COLORS[DELIVERY], edgecolor="black", label="Line-side Delivery Zone"),
        Patch(facecolor=COLORS[INSPECTION], edgecolor="black", label="Inspection Zone"),
        Patch(facecolor=COLORS[MACHINE], edgecolor="black", label="Sequencing / Kitting / Supermarket"),
        Patch(facecolor=COLORS[CHARGING], edgecolor="black", label="Charging / Waiting"),
        Patch(facecolor=COLORS[BUFFER], edgecolor="black", label="Vehicle / Parts Buffer"),
        Line2D([0], [0], marker="P", color="w", markeredgecolor="black", markerfacecolor="black", label="Pickup Point", markersize=9),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="black", markerfacecolor="black", label="Delivery Point", markersize=9),
        Line2D([0], [0], marker="^", color="w", markeredgecolor="black", markerfacecolor="black", label="Charging Point", markersize=9),
    ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_map_summary(data):
    """Print key information about the generated factory map."""
    factory_map = data["factory_map"]
    walkable_map = data["walkable_map"]
    obstacle_map = data["obstacle_map"]

    conveyor_cells = int(((factory_map == OBSTACLE) & (
        np.indices(factory_map.shape)[1] >= 38) & (np.indices(factory_map.shape)[1] < 74) &
        (((np.indices(factory_map.shape)[0] >= 16) & (np.indices(factory_map.shape)[0] < 18)) |
         ((np.indices(factory_map.shape)[0] >= 29) & (np.indices(factory_map.shape)[0] < 31)))
    ).sum())

    print("=== Automotive Assembly Parts-Supply Map Summary ===")
    print(f"Map shape          : {factory_map.shape}  # (height, width)")
    print(f"Total cells        : {factory_map.size}")
    print(f"Walkable cells     : {int(walkable_map.sum())}")
    print(f"Obstacle cells     : {int(obstacle_map.sum())}")
    print(f"Conveyor cells     : {conveyor_cells}")
    print(f"Pickup points      : {data['pickup_points']}")
    print(f"Delivery points    : {len(data['delivery_points'])} points")
    print(f"Charging points    : {data['charging_points']}")
    print(f"Start candidates   : {data['start_candidates']}")
    print("Manual bottleneck  : removed")
    print("Manual one-way     : removed")


if __name__ == "__main__":
    data = build_factory_map()
    print_map_summary(data)

    maps_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "maps"
        / datetime.now().strftime("%y%m%d_%H%M")
    )
    maps_dir.mkdir(parents=True, exist_ok=True)
    np.save(maps_dir / "factory_map_v3_oneway.npy", data["factory_map"])
    np.save(maps_dir / "walkable_map_v3_oneway.npy", data["walkable_map"])
    np.save(maps_dir / "obstacle_map_v3_oneway.npy", data["obstacle_map"])
    print(f"Saved numpy arrays to: {maps_dir}")

    visualize_factory_map(data, save_path=maps_dir / "factory_map_v3_oneway_preview.png", show=True)
