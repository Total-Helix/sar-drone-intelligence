import random
import asyncio
from datetime import datetime, timezone
from typing import Optional

TERRAIN_DESCRIPTIONS = {
    "dense_forest": "Thick canopy, limited ground visibility, high moisture",
    "open_field": "Clear sightlines, dry grass, easy traversal",
    "mixed_forest": "Moderate canopy, varied undergrowth, some clearings",
    "steep_slope": "45+ degree incline, loose soil, challenging movement",
    "creek_side": "Waterway visible, muddy banks — high track preservation",
    "trail": "Established path, clear footprint visibility, high-confidence zone",
    "rocky_outcrop": "Exposed bedrock, limited tracks, visual landmarks present",
}

DETECTION_DESCRIPTIONS = {
    "footprint": [
        "Fresh boot print ~size 10, deep heel impression in mud",
        "Multiple overlapping prints suggesting resting point",
        "Partial sole pattern visible in soft earth",
    ],
    "broken_branch": [
        "Branch snapped at ~1.5m height, consistent with person pushing through",
        "Multiple branches disturbed along a corridor, directional indicator",
    ],
    "disturbed_ground": [
        "Vegetation flattened, consistent with someone sitting or falling",
        "Ground disturbed in a circular pattern — possible resting area",
    ],
    "campfire_ash": [
        "Cold ash pile, estimated 12-24h old. Intentional shelter sign.",
        "Scattered embers, recent fire activity within 6h",
    ],
    "clothing_fragment": [
        "Orange fabric fragment snagged on bramble at shoulder height",
        "Synthetic fiber consistent with hiking gear",
    ],
    "water_bottle": [
        "Partially full water bottle, branded outdoor gear",
        "Empty container found near creek bank",
    ],
    "trail_blazing": [
        "Stacked stones (cairn) indicating intentional navigation",
        "Bark scratching on tree trunk — directional mark",
    ],
}


from opentelemetry import trace
tracer = trace.get_tracer(__name__)

class FrameAnalyzer:
    """Mock Gemini Vision frame analyzer."""

    def __init__(self):
        self.model_name = "gemini-2.0-flash (MOCK)"
        self.analysis_count = 0

    @tracer.start_as_current_span("analyze_frame_vision")
    async def analyze_frame(self, frame: dict) -> dict:
        """Analyze a drone frame. Returns structured detection results."""
        await asyncio.sleep(random.uniform(0.3, 0.8))  # Simulate API latency
        self.analysis_count += 1

        terrain_type = frame.get("terrain_type", "mixed_forest")
        raw_detections = frame.get("raw_detections", [])

        analyzed_detections = []
        for raw in raw_detections:
            result = await self._mock_gemini_call(raw, terrain_type)
            if result:
                analyzed_detections.append(result)

        return {
            "frame_id": frame["frame_id"],
            "frame_number": frame["frame_number"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_used": self.model_name,
            "terrain_analysis": {
                "type": terrain_type,
                "description": TERRAIN_DESCRIPTIONS.get(terrain_type, "Unknown terrain"),
                "traversability": self._score_traversability(terrain_type),
                "track_preservation": self._score_track_preservation(terrain_type),
            },
            "detections": analyzed_detections,
            "frame_summary": self._generate_summary(analyzed_detections, terrain_type),
            "gemini_tokens_used": random.randint(800, 2400),
        }

    async def _mock_gemini_call(self, raw: dict, terrain_type: str) -> Optional[dict]:
        """MOCK: Returns structured detection. Replace with real Gemini call."""
        if raw["raw_confidence"] < 0.45:
            return None
        terrain_boost = {"creek_side": 0.10, "trail": 0.08, "open_field": 0.05}.get(terrain_type, 0)
        confidence = min(0.99, raw["raw_confidence"] + terrain_boost)
        track_type = raw["track_type"]
        descriptions = DETECTION_DESCRIPTIONS.get(track_type, ["Unclassified ground disturbance"])
        return {
            "detection_id": raw["detection_id"],
            "track_type": track_type,
            "location": {"lat": raw["detection_lat"], "lng": raw["detection_lng"]},
            "confidence": round(confidence, 2),
            "description": random.choice(descriptions),
            "estimated_age_hours": round(random.uniform(0.5, 36), 1),
            "significance": "high" if confidence > 0.80 else "medium" if confidence > 0.60 else "low",
        }

    def _score_traversability(self, terrain: str) -> float:
        return {"trail": 0.95, "open_field": 0.85, "mixed_forest": 0.60,
                "dense_forest": 0.35, "steep_slope": 0.20, "creek_side": 0.40,
                "rocky_outcrop": 0.30}.get(terrain, 0.5)

    def _score_track_preservation(self, terrain: str) -> float:
        return {"creek_side": 0.95, "dense_forest": 0.70, "trail": 0.65,
                "mixed_forest": 0.55, "open_field": 0.40, "steep_slope": 0.35,
                "rocky_outcrop": 0.15}.get(terrain, 0.5)

    def _generate_summary(self, detections: list, terrain: str) -> str:
        if not detections:
            return f"No significant tracks detected on {terrain.replace('_', ' ')} terrain."
        high = sum(1 for d in detections if d["significance"] == "high")
        types = list(set(d["track_type"] for d in detections))
        return (f"Detected {len(detections)} track(s) ({high} high-confidence) on "
                f"{terrain.replace('_', ' ')}. Types: {', '.join(types)}.")
