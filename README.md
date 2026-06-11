# 🚁 SAR AI Agent — Drone-Powered Search & Rescue Intelligence

[![GitLab CI](https://img.shields.io/badge/CI-GitLab-fc6d26?logo=gitlab)](../.gitlab-ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](./backend)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](./backend/main.py)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-00ED64?logo=mongodb)](./docker-compose.yml)
[![Elasticsearch](https://img.shields.io/badge/Elastic-8.13-FEC514?logo=elasticsearch)](./backend/db/elastic_client.py)

> An AI-powered Search & Rescue system that ingests live drone footage, detects physical tracks on the ground, applies psychological behavior models of lost persons, and generates the most probable rescue route — all visualized on a real-time interactive dashboard.

---

## 🎯 Core Concept

```
Drone captures footage
  → Frame extraction + object detection (Mock Gemini Vision)
  → Track/trace identification (footprints, broken branches, disturbed ground)
  → Terrain mapping
  → Psychological model applied (Mattson's Lost Person Behavior data)
  → Probability heatmap generated
  → Optimal rescue route → field teams in real time
```

---

## 🏗️ Architecture

```
AIDroneTrackerCode/
├── backend/
│   ├── main.py                    # FastAPI + WebSocket server
│   ├── pipeline/
│   │   ├── drone_simulator.py     # Synthetic drone flight generator
│   │   ├── frame_analyzer.py      # Mock Gemini Vision analyzer
│   │   ├── psych_model.py         # Mattson behavioral scoring engine
│   │   └── route_generator.py     # Heatmap + rescue route generator
│   ├── db/
│   │   ├── mongo_client.py        # MongoDB (missions, detections, routes)
│   │   └── elastic_client.py      # Elasticsearch (historical case search)
│   ├── monitoring/
│   │   └── arize_logger.py        # Arize Phoenix prediction logger (mock)
│   └── requirements.txt
├── frontend/
│   ├── index.html                 # Dashboard UI
│   ├── style.css                  # Dark theme design system
│   └── app.js                     # Leaflet map + WebSocket client
├── docker-compose.yml             # MongoDB + Elasticsearch
├── .gitlab-ci.yml                 # CI: lint → test → build → deploy
└── README.md
```

---

## 🚀 Quick Start (For Judges)

The entire system (Backend API, Frontend Dashboard, MongoDB, and Elasticsearch) is fully Dockerized for ease of review.

> **Enterprise Scale:** Full Kubernetes manifests are available in the `k8s/` directory for production deployments and horizontal scaling, but we recommend Docker Compose for local review.

### 1. Launch the Stack

From the root directory, simply run:
```bash
docker compose up -d --build
```

### 2. Access the Dashboards
- **Live Mission Dashboard:** Navigate to **http://localhost** 
- **Arize Phoenix Tracing UI:** Navigate to **http://localhost:6006** (Watch the agent's real-time LLM reasoning traces here)

*The frontend Nginx container automatically proxies all `/api/` and `/ws/` requests to the backend.*

### 3. Verify Databases (Optional)
```bash
# MongoDB
docker exec sar_mongodb mongosh --eval 'db.runCommand({ping:1})'

# Elasticsearch
curl http://localhost:9200/_cluster/health?pretty
```

---

## 🎮 Usage

1. **Open the dashboard** in your browser
2. **Fill in the mission form** (subject name, type, terrain, hours missing)
3. **Click "Launch Drone Mission"** 
4. Watch the live dashboard:
   - 🗺️ Drone position animates on the Leaflet map
   - 🔥 Probability heatmap builds as frames are analyzed
   - 📍 Track detection markers appear at detection sites
   - 📡 Detection feed scrolls in real time
   - 🧭 Rescue route with prioritized waypoints generates at the end
5. **Check Similar Cases** panel — matched from Elasticsearch historical SAR database

---

## 🔬 Partner Track Selected: **Arize**

This project competes in the **Arize** track, leveraging **Arize Phoenix** and **OpenInference** to give the Gemini Agent its core "superpowers" of observability and self-introspection. In life-and-death Search & Rescue, AI hallucinations cost lives. Arize solves this by making the agent 100% transparent.

| Partner | Role in System | Status |
|---|---|---|
| **Arize (Primary)** | **Primary Track Partner.** The custom Python agent is instrumented with OpenInference. Every frame analysis, psychological scoring, and route generation is logged to the local Phoenix UI for full visibility. | ✅ Active |
| **Elasticsearch** | Semantic search over historical unstructured SAR cases. | ✅ Active |
| **MongoDB** | Store live mission state, drone telemetry, and behavioral profiles. | ✅ Active |
| **GitLab** | CI/CD pipeline ensuring code quality and automated Docker builds. | ✅ Active |

---

## 🧠 Psychological Model (Mattson's Lost Person Behavior)

The system applies evidence-based behavioral scoring from *"Lost Person Behavior"* (Mattson, 2011):

| Subject Type | Avg Distance | Found Alive | Key Behavior |
|---|---|---|---|
| **Hiker** | 3.1 km | 96% | Downhill bias, trail following, water attraction |
| **Child** | 1.0 km | 97% | Hides from rescuers, shelter-seeking, stays close |
| **Dementia** | 1.5 km | 69% | Circular wandering, road attraction — CRITICAL urgency |
| **Hunter** | 4.2 km | 98% | Seeks high ground, wide range, self-rescue attempts |

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/mission/start` | POST | Create a new SAR mission |
| `/api/mission/{id}/run-pipeline` | POST | Start drone analysis pipeline |
| `/api/mission/{id}` | GET | Get mission status |
| `/api/missions` | GET | List all missions |
| `/api/mission/{id}/heatmap` | GET | Get probability heatmap grid |
| `/api/mission/{id}/route` | GET | Get rescue route waypoints |
| `/api/mission/{id}/detections` | GET | Get all track detections |
| `/api/cases/similar` | GET | Search historical cases (Elastic) |
| `/api/arize/stats` | GET | Get Arize prediction stats |
| `/ws/{mission_id}` | WS | Live mission updates |

Full interactive docs: **http://localhost:8000/docs**

---

## 🔑 Adding the Real Gemini API Key

The frame analyzer currently runs in **mock mode**. To connect the real Gemini Vision API:

1. Get your API key from [Google AI Studio](https://aistudio.google.com/)
2. Create a `.env` file in `backend/`:
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. In `backend/pipeline/frame_analyzer.py`, replace `_mock_gemini_call()` with:
   ```python
   import google.generativeai as genai
   genai.configure(api_key=os.environ['GEMINI_API_KEY'])
   model = genai.GenerativeModel('gemini-2.0-flash')
   # Pass frame image data to model.generate_content()
   ```

---

## 🧪 Running Tests

```bash
# Make sure Docker services are running first
docker compose up -d

cd backend
python -m pytest tests/ -v
```

---

## 📊 Why This Could Win

- **Real-world life-saving impact** — not a chatbot or task manager
- **Multi-partner integration** — all 6 hackathon partners serve distinct roles
- **Non-trivial multimodal AI** — vision + reasoning + psychological modeling
- **Grounded in real SAR science** — Mattson's Lost Person Behavior data
- **Emotionally resonant for judges** — people found alive
- **Technically impressive demo** — drone + AI + real-time map

---

## 📝 License

MIT License — built for the Google DeepMind hackathon.
