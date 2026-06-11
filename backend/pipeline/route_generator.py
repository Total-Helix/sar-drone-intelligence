import math
import random
from typing import List, Dict
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

GRID_ROWS = 10
GRID_COLS = 10
CELL_SIZE = 0.005  # ~550m per cell

TERRAIN_TYPES = [
    "dense_forest", "open_field", "mixed_forest", "steep_slope",
    "creek_side", "trail", "rocky_outcrop"
]


def generate_grid(center_lat: float, center_lng: float) -> List[Dict]:
    """Generate a 10x10 search grid centered on the last known point."""
    rng = random.Random(int(abs(center_lat * center_lng * 1e6)) % (2 ** 31))
    grid = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            lat = center_lat + (row - GRID_ROWS // 2) * CELL_SIZE
            lng = center_lng + (col - GRID_COLS // 2) * CELL_SIZE
            grid.append({
                "zone_id": f"{chr(65 + row)}{col + 1}",
                "lat": round(lat + CELL_SIZE / 2, 6),
                "lng": round(lng + CELL_SIZE / 2, 6),
                "row": row,
                "col": col,
                "terrain_type": rng.choice(TERRAIN_TYPES),
                "probability": 0.0,
            })
    return grid


@tracer.start_as_current_span("generate_rescue_route")
def generate_rescue_route(scored_cells: List[Dict], max_waypoints: int = 8) -> List[Dict]:
    """Greedy nearest-neighbor route through highest probability zones."""
    if not scored_cells:
        return []
    top = sorted(scored_cells, key=lambda c: c["probability"], reverse=True)[:max_waypoints]
    if not top:
        return []

    route = [top[0]]
    remaining = top[1:]
    while remaining:
        last = route[-1]
        nearest = min(remaining, key=lambda c: _dist(last, c))
        route.append(nearest)
        remaining.remove(nearest)

    return [
        {
            "waypoint_id": f"WP{i + 1:02d}",
            "order": i + 1,
            "lat": cell["lat"],
            "lng": cell["lng"],
            "zone_id": cell["zone_id"],
            "probability": cell["probability"],
            "priority": "critical" if cell["probability"] > 0.70 else "high" if cell["probability"] > 0.40 else "medium",
            "terrain_type": cell["terrain_type"],
            "reason": _build_reason(cell),
            "estimated_search_time_min": _search_time(cell["terrain_type"]),
        }
        for i, cell in enumerate(route)
    ]


def _dist(a: dict, b: dict) -> float:
    return math.sqrt((a["lat"] - b["lat"]) ** 2 + (a["lng"] - b["lng"]) ** 2)


def _build_reason(cell: dict) -> str:
    parts = []
    if cell["probability"] > 0.65:
        parts.append("Highest probability zone")
    if cell.get("detection_score", 0) > 0.25:
        parts.append("Multiple track detections nearby")
    if cell["terrain_type"] in ("creek_side", "trail"):
        parts.append(f"High-attractor terrain ({cell['terrain_type'].replace('_', ' ')})")
    return " · ".join(parts) if parts else f"Systematic coverage zone {cell['zone_id']}"


def _search_time(terrain: str) -> int:
    return {"trail": 8, "open_field": 10, "mixed_forest": 20, "dense_forest": 35,
            "steep_slope": 45, "creek_side": 25, "rocky_outcrop": 30}.get(terrain, 20)
