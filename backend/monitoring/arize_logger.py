import os
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class ArizeLogger:
    """
    Lightweight wrapper to attach mission metadata to the current active OpenTelemetry span.
    The actual tracing is handled by OpenInference @trace decorators.
    """
    
    def __init__(self):
        self.prediction_count = 0
        self.model_id = "sar-gemini-vision"

    def log_frame_prediction(self, mission_id: str, frame_id: str, frame_number: int,
                              terrain_type: str, subject_type: str, detections: list,
                              model_confidence: float):
        self.prediction_count += 1
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("mission_id", mission_id)
            current_span.set_attribute("frame_id", frame_id)
            current_span.set_attribute("frame_number", frame_number)
            current_span.set_attribute("terrain_type", terrain_type)
            current_span.set_attribute("subject_type", subject_type)
            current_span.set_attribute("detection_count", len(detections))
            current_span.set_attribute("model_confidence", model_confidence)

    def log_route_prediction(self, mission_id: str, subject_type: str,
                              waypoint_count: int, top_zone_probability: float):
        self.prediction_count += 1
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("mission_id", mission_id)
            current_span.set_attribute("subject_type", subject_type)
            current_span.set_attribute("waypoint_count", waypoint_count)
            current_span.set_attribute("top_zone_probability", top_zone_probability)

    def get_stats(self) -> dict:
        return {
            "status": "active (OpenTelemetry)",
            "total_predictions_logged": self.prediction_count,
            "model_id": self.model_id,
        }

arize_logger = ArizeLogger()
