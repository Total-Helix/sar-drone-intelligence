import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, List

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

tracer = otel_trace.get_tracer(__name__)

# Setup OpenTelemetry for Arize Phoenix
provider = TracerProvider()
endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
if not endpoint.endswith("/v1/traces"):
    endpoint += "/v1/traces"

try:
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    otel_trace.set_tracer_provider(provider)
    print(f"[Arize] OpenTelemetry tracing initialized pointing to {endpoint}")
except Exception as e:
    print(f"[Arize] Tracing init failed: {e}")

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db.elastic_client import close_es, search_similar_cases, setup_index
from db.mongo_client import (
    create_mission, get_detections, get_mission, get_route,
    list_missions, save_detections, save_route, update_mission,
)
from monitoring.arize_logger import arize_logger
from pipeline.drone_simulator import BASE_LAT, BASE_LNG, DroneSimulator
from pipeline.frame_analyzer import FrameAnalyzer
from pipeline.psych_model import PsychModel
from pipeline.route_generator import generate_grid, generate_rescue_route

# --- WebSocket connections per mission ---
active_ws: Dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SAR Agent] Starting...")
    try:
        await setup_index()
    except Exception as e:
        print(f"[Elastic] Startup skipped: {e}")
    yield
    await close_es()
    print("[SAR Agent] Shutdown complete.")


app = FastAPI(
    title="SAR AI Agent",
    description="AI-powered Search & Rescue drone tracking system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


# --- Models ---
class MissionRequest(BaseModel):
    subject_type: str = "hiker"
    subject_name: str = "Unknown"
    last_known_lat: float = BASE_LAT
    last_known_lng: float = BASE_LNG
    reported_missing_hours: float = 2.0
    terrain_type: str = "mixed_forest"
    notes: str = ""
    # Enhanced behavioral profile fields
    subject_age: int | None = None
    subject_gender: str | None = None
    circumstance: str = "normal"
    nationality: str | None = None
    education_level: str | None = None
    education_field: str | None = None
    subject_job: str | None = None
    # Temporal fields
    entry_time: str | None = None          # ISO datetime: when subject entered area / last seen
    search_start_time: str | None = None   # ISO datetime: when search begins (defaults to now)


# --- Helpers ---
async def broadcast(mission_id: str, msg: dict):
    if mission_id not in active_ws:
        return
    dead = []
    for ws in active_ws[mission_id]:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_ws[mission_id].remove(ws)


# --- WebSocket ---
@app.websocket("/ws/{mission_id}")
async def ws_endpoint(websocket: WebSocket, mission_id: str):
    await websocket.accept()
    active_ws.setdefault(mission_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if mission_id in active_ws and websocket in active_ws[mission_id]:
            active_ws[mission_id].remove(websocket)


# --- Pipeline ---
@tracer.start_as_current_span("execute_sar_mission")
async def _run_pipeline(mission_id: str, mission: dict):
    try:
        await update_mission(mission_id, {"status": "running"})
        await broadcast(mission_id, {"type": "status_update", "status": "running",
                                      "progress": 0.0, "message": "Initializing drone simulation..."})

        lk = mission.get("last_known", {"lat": BASE_LAT, "lng": BASE_LNG})
        stype = mission.get("subject_type", "hiker")

        simulator = DroneSimulator(mission_id, stype, lk)
        frames = simulator.generate_flight_path(num_frames=15)
        analyzer = FrameAnalyzer()
        all_detections = []

        for i, frame in enumerate(frames):
            progress = round((i + 1) / len(frames) * 0.70, 2)
            await broadcast(mission_id, {"type": "status_update", "status": "running",
                                          "progress": progress, "message": f"Analyzing frame {i+1}/{len(frames)}..."})
            result = await analyzer.analyze_frame(frame)
            dets = result.get("detections", [])
            all_detections.extend(dets)

            conf = sum(d["confidence"] for d in dets) / len(dets) if dets else 0.0
            arize_logger.log_frame_prediction(
                mission_id, frame["frame_id"], i + 1,
                frame["terrain_type"], stype, dets, conf
            )

            await broadcast(mission_id, {
                "type": "frame_analyzed",
                "frame_number": i + 1, "total_frames": len(frames),
                "drone_lat": frame["drone_lat"], "drone_lng": frame["drone_lng"],
                "terrain_type": frame["terrain_type"], "detections": dets,
                "terrain_analysis": result.get("terrain_analysis", {}),
                "summary": result.get("frame_summary", ""),
            })
            await asyncio.sleep(0.05)

        if all_detections:
            await save_detections(mission_id, all_detections)

        await broadcast(mission_id, {"type": "status_update", "status": "running",
                                      "progress": 0.78, "message": "Fetching historical cases and applying behavior model..."})
        
        # Fetch historical cases
        terrain_type = mission.get("terrain_type", "mixed_forest")
        historical_cases = await search_similar_cases(stype, terrain_type, limit=3)

        psych = PsychModel(
            subject_type=stype,
            age=mission.get("subject_age"),
            gender=mission.get("subject_gender"),
            circumstance=mission.get("circumstance", "normal"),
            nationality=mission.get("nationality"),
            education_level=mission.get("education_level"),
            education_field=mission.get("education_field"),
            job=mission.get("subject_job"),
            entry_time=mission.get("entry_time"),
            search_start_time=mission.get("search_start_time"),
            reported_missing_hours=float(mission.get("reported_missing_hours", 0)),
            historical_cases=historical_cases,
        )
        grid = generate_grid(lk["lat"], lk["lng"])
        scored = psych.score_grid(grid, lk, all_detections)
        insights = psych.get_behavioral_insights()
        await broadcast(mission_id, {"type": "heatmap_updated", "cells": scored, "behavioral_insights": insights})

        await broadcast(mission_id, {"type": "status_update", "status": "running",
                                      "progress": 0.92, "message": "Generating optimal rescue route..."})
        waypoints = generate_rescue_route(scored, max_waypoints=8)
        await save_route(mission_id, waypoints, scored)

        top_p = max((c["probability"] for c in scored), default=0)
        arize_logger.log_route_prediction(mission_id, stype, len(waypoints), top_p)
        await broadcast(mission_id, {"type": "route_updated", "waypoints": waypoints})

        await update_mission(mission_id, {"status": "completed", "frame_count": len(frames),
                                           "detection_count": len(all_detections), "waypoint_count": len(waypoints)})
        await broadcast(mission_id, {
            "type": "pipeline_complete", "mission_id": mission_id,
            "summary": {"frames_analyzed": len(frames), "total_detections": len(all_detections),
                        "waypoints_generated": len(waypoints), "behavioral_insights": insights,
                        "arize_logged": arize_logger.prediction_count},
        })
    except Exception as e:
        await update_mission(mission_id, {"status": "error", "error": str(e)})
        await broadcast(mission_id, {"type": "error", "message": str(e)})
        raise


# --- API Routes ---
@app.post("/api/mission/start")
async def start_mission(req: MissionRequest):
    mid = await create_mission({
        "subject_type": req.subject_type, "subject_name": req.subject_name,
        "last_known": {"lat": req.last_known_lat, "lng": req.last_known_lng},
        "reported_missing_hours": req.reported_missing_hours,
        "terrain_type": req.terrain_type, "notes": req.notes,
        "subject_age": req.subject_age,
        "subject_gender": req.subject_gender,
        "circumstance": req.circumstance,
        "nationality": req.nationality,
        "education_level": req.education_level,
        "education_field": req.education_field,
        "subject_job": req.subject_job,
        "entry_time": req.entry_time,
        "search_start_time": req.search_start_time,
        "frame_count": 0, "detection_count": 0,
    })
    return {"mission_id": mid, "status": "initializing"}


@app.post("/api/mission/{mission_id}/run-pipeline")
async def run_pipeline(mission_id: str):
    mission = await get_mission(mission_id)
    if not mission:
        raise HTTPException(404, "Mission not found")
    asyncio.create_task(_run_pipeline(mission_id, mission))
    return {"status": "started", "mission_id": mission_id}


@app.get("/api/mission/{mission_id}")
async def get_mission_status(mission_id: str):
    m = await get_mission(mission_id)
    if not m:
        raise HTTPException(404, "Mission not found")
    return m


@app.get("/api/missions")
async def get_missions():
    return await list_missions()


@app.get("/api/mission/{mission_id}/heatmap")
async def get_heatmap(mission_id: str):
    data = await get_route(mission_id)
    if not data:
        raise HTTPException(404, "No heatmap yet — run the pipeline first.")
    return {"mission_id": mission_id, "cells": data.get("heatmap", [])}


@app.get("/api/mission/{mission_id}/route")
async def get_route_data(mission_id: str):
    data = await get_route(mission_id)
    if not data:
        raise HTTPException(404, "No route yet — run the pipeline first.")
    return {"mission_id": mission_id, "waypoints": data.get("waypoints", [])}


@app.get("/api/mission/{mission_id}/detections")
async def get_mission_detections(mission_id: str):
    return {"mission_id": mission_id, "detections": await get_detections(mission_id)}


@app.get("/api/cases/similar")
async def similar_cases(subject_type: str = "hiker", terrain: str = "mixed_forest"):
    cases = await search_similar_cases(subject_type, terrain)
    return {"cases": cases, "count": len(cases)}


@app.get("/api/arize/stats")
async def arize_stats():
    return arize_logger.get_stats()


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
async def root():
    idx = os.path.join(FRONTEND, "index.html")
    return FileResponse(idx) if os.path.isfile(idx) else {"service": "SAR AI Agent API"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
