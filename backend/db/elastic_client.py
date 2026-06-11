import os
from typing import List, Optional

from elasticsearch import AsyncElasticsearch

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX = "sar_cases"

_es: Optional[AsyncElasticsearch] = None

HISTORICAL_CASES = [
    {"case_id": "CASE-001", "subject_type": "hiker", "terrain": "dense_forest", "area": "Cascade Range, WA", "outcome": "found_alive", "hours_to_find": 18.5, "distance_km": 3.2, "key_factors": ["downhill travel", "water source", "trail junction"], "year": 2022},
    {"case_id": "CASE-002", "subject_type": "child", "terrain": "mixed_forest", "area": "Oregon Coast Range", "outcome": "found_alive", "hours_to_find": 6.0, "distance_km": 0.8, "key_factors": ["hiding behavior", "sheltered under log", "1km radius"], "year": 2023},
    {"case_id": "CASE-003", "subject_type": "dementia", "terrain": "mixed_forest", "area": "Flathead, MT", "outcome": "found_alive", "hours_to_find": 11.0, "distance_km": 1.4, "key_factors": ["road attraction", "circular wandering", "hypothermia risk"], "year": 2021},
    {"case_id": "CASE-004", "subject_type": "hunter", "terrain": "steep_slope", "area": "Sawtooth, ID", "outcome": "found_alive", "hours_to_find": 29.0, "distance_km": 5.7, "key_factors": ["ridgeline navigation", "improvised shelter", "signal fire"], "year": 2022},
    {"case_id": "CASE-005", "subject_type": "hiker", "terrain": "creek_side", "area": "Sierra Nevada, CA", "outcome": "found_alive", "hours_to_find": 9.5, "distance_km": 2.1, "key_factors": ["water source following", "downstream travel", "mud footprints"], "year": 2023},
    {"case_id": "CASE-006", "subject_type": "child", "terrain": "dense_forest", "area": "Upper Peninsula, MI", "outcome": "found_alive", "hours_to_find": 8.0, "distance_km": 1.1, "key_factors": ["dog tracking", "cedar tree shelter", "no response to calls"], "year": 2020},
    {"case_id": "CASE-007", "subject_type": "dementia", "terrain": "open_field", "area": "Iowa Farmland", "outcome": "found_alive", "hours_to_find": 4.5, "distance_km": 0.9, "key_factors": ["road attraction", "neighbor sighting", "heading to town"], "year": 2023},
    {"case_id": "CASE-008", "subject_type": "hunter", "terrain": "rocky_outcrop", "area": "Big Horn, WY", "outcome": "found_alive", "hours_to_find": 41.0, "distance_km": 8.3, "key_factors": ["high ground seeking", "self-rescue attempt", "wide search area"], "year": 2021},
    {"case_id": "CASE-009", "subject_type": "hiker", "terrain": "trail", "area": "Rocky Mountain, CO", "outcome": "found_alive", "hours_to_find": 5.5, "distance_km": 4.8, "key_factors": ["wrong trail taken", "junction confusion", "self-rescued"], "year": 2023},
    {"case_id": "CASE-010", "subject_type": "dementia", "terrain": "creek_side", "area": "Cumberland, TN", "outcome": "found_deceased", "hours_to_find": 72.0, "distance_km": 2.3, "key_factors": ["water attraction", "drowning risk", "delayed search"], "year": 2020},
]


def get_es() -> AsyncElasticsearch:
    global _es
    if _es is None:
        _es = AsyncElasticsearch([ES_URL], request_timeout=10)
    return _es


async def setup_index():
    es = get_es()
    try:
        if not await es.indices.exists(index=INDEX):
            await es.indices.create(index=INDEX, body={
                "mappings": {"properties": {
                    "case_id": {"type": "keyword"},
                    "subject_type": {"type": "keyword"},
                    "terrain": {"type": "keyword"},
                    "area": {"type": "text"},
                    "outcome": {"type": "keyword"},
                    "hours_to_find": {"type": "float"},
                    "distance_km": {"type": "float"},
                    "key_factors": {"type": "text"},
                    "year": {"type": "integer"},
                }}
            })
            for case in HISTORICAL_CASES:
                await es.index(index=INDEX, id=case["case_id"], document=case)
            await es.indices.refresh(index=INDEX)
            print(f"[Elastic] Index created and seeded with {len(HISTORICAL_CASES)} historical cases.")
        else:
            print("[Elastic] Index already exists.")
    except Exception as e:
        print(f"[Elastic] Setup warning (non-fatal): {e}")


async def search_similar_cases(subject_type: str, terrain: str, limit: int = 5) -> List[dict]:
    try:
        result = await get_es().search(index=INDEX, body={
            "query": {"bool": {"should": [
                {"term": {"subject_type": subject_type}},
                {"term": {"terrain": terrain}},
            ], "minimum_should_match": 1}},
            "size": limit,
        })
        return [h["_source"] for h in result["hits"]["hits"]]
    except Exception as e:
        print(f"[Elastic] Search error: {e}")
        return []


async def index_completed_mission(mission_id: str, subject_type: str, terrain: str, insights: dict, waypoints: list):
    """Write agent outputs and enriched facts back into Elasticsearch so the agent builds on what it knows."""
    try:
        # Create a new historical case record based on the completed mission predictions
        top_factors = insights.get("historical_context", {}).get("common_factors", [])
        if not top_factors:
            top_factors = ["drone_spotted", "AI_predicted_route"]
            
        doc = {
            "case_id": f"MISSION-{mission_id[:8]}",
            "subject_type": subject_type,
            "terrain": terrain,
            "area": "Live Mission Area",
            "outcome": "prediction_logged",
            "hours_to_find": insights.get("time_context", {}).get("hours_missing", 0.0),
            "distance_km": insights.get("effective_distance_km", 0.0),
            "key_factors": top_factors,
            "year": 2026,
            "ai_generated_waypoints": len(waypoints)
        }
        await get_es().index(index=INDEX, id=doc["case_id"], document=doc)
        print(f"[Elastic] Mission {mission_id[:8]} indexed back to memory for future context.")
    except Exception as e:
        print(f"[Elastic] Failed to index completed mission: {e}")

async def close_es():
    global _es
    if _es:
        await _es.close()
        _es = None
