import random
import uuid
import math
from datetime import datetime, timezone

TERRAIN_TYPES = ["dense_forest", "open_field", "mixed_forest", "steep_slope", "creek_side", "trail", "rocky_outcrop"]
TRACK_TYPES = ["footprint", "broken_branch", "disturbed_ground", "campfire_ash", "clothing_fragment", "water_bottle", "trail_blazing"]

BASE_LAT = 47.5000
BASE_LNG = -121.8000


class DroneSimulator:
    def __init__(self, mission_id: str, subject_type: str = "hiker", last_known: dict = None):
        self.mission_id = mission_id
        self.subject_type = subject_type
        self.last_known = last_known or {"lat": BASE_LAT, "lng": BASE_LNG}
        self.rng = random.Random(hash(mission_id) % (2**32))

    def generate_flight_path(self, num_frames: int = 15) -> list:
        """Generate drone waypoints as an expanding spiral search pattern."""
        frames = []
        angle = 0
        radius = 0.003
        for i in range(num_frames):
            angle += 30
            if i % 6 == 0 and i > 0:
                radius += 0.004
            lat = self.last_known["lat"] + radius * math.sin(math.radians(angle))
            lng = self.last_known["lng"] + radius * math.cos(math.radians(angle))
            frames.append({
                "frame_id": str(uuid.uuid4()),
                "frame_number": i + 1,
                "total_frames": num_frames,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "drone_lat": round(lat, 6),
                "drone_lng": round(lng, 6),
                "altitude_m": round(self.rng.uniform(30, 80), 1),
                "terrain_type": self.rng.choice(TERRAIN_TYPES),
                "visibility": self.rng.choice(["clear", "partial", "obstructed"]),
                "raw_detections": self._generate_detections(lat, lng, i),
            })
        return frames

    def _generate_detections(self, center_lat: float, center_lng: float, frame_index: int) -> list:
        detections = []
        base_probability = max(0.2, 0.85 - frame_index * 0.04)
        num_detections = self.rng.choices(
            [0, 1, 2, 3],
            weights=[1 - base_probability, 0.4, 0.3, 0.2]
        )[0]
        for _ in range(num_detections):
            offset_lat = self.rng.uniform(-0.002, 0.002)
            offset_lng = self.rng.uniform(-0.002, 0.002)
            detections.append({
                "detection_id": str(uuid.uuid4()),
                "track_type": self.rng.choice(TRACK_TYPES),
                "detection_lat": round(center_lat + offset_lat, 6),
                "detection_lng": round(center_lng + offset_lng, 6),
                "raw_confidence": round(self.rng.uniform(0.4, 0.99), 2),
                "pixel_area": self.rng.randint(50, 500),
            })
        return detections
