import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "sar_agent"

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    return get_client()[DB_NAME]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_mission(data: dict) -> str:
    db = get_db()
    mission_id = str(uuid.uuid4())
    await db.missions.insert_one({"_id": mission_id, "status": "initializing",
                                   "created_at": _now(), "updated_at": _now(), **data})
    return mission_id


async def get_mission(mission_id: str) -> Optional[dict]:
    doc = await get_db().missions.find_one({"_id": mission_id})
    if doc:
        doc["id"] = doc.pop("_id")
    return doc


async def update_mission(mission_id: str, updates: dict):
    updates["updated_at"] = _now()
    await get_db().missions.update_one({"_id": mission_id}, {"$set": updates})


async def list_missions() -> List[dict]:
    cursor = get_db().missions.find().sort("created_at", DESCENDING).limit(20)
    result = []
    async for doc in cursor:
        doc["id"] = doc.pop("_id")
        result.append(doc)
    return result


async def save_detections(mission_id: str, detections: List[dict]):
    if not detections:
        return
    await get_db().detections.insert_many([{"mission_id": mission_id, **d} for d in detections])


async def get_detections(mission_id: str) -> List[dict]:
    cursor = get_db().detections.find({"mission_id": mission_id}, {"_id": 0})
    return [doc async for doc in cursor]


async def save_route(mission_id: str, waypoints: list, heatmap: list):
    await get_db().routes.replace_one(
        {"mission_id": mission_id},
        {"mission_id": mission_id, "waypoints": waypoints, "heatmap": heatmap, "saved_at": _now()},
        upsert=True,
    )


async def get_route(mission_id: str) -> Optional[dict]:
    return await get_db().routes.find_one({"mission_id": mission_id}, {"_id": 0})
