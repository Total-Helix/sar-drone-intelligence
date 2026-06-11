'use strict';

// =====================
// CONSTANTS & CONFIG
// =====================
// Determine API URLs dynamically (supports both local dev and Docker Nginx proxy)
const isLocalDev = window.location.port === '3000' || window.location.href.includes('file://');
const API = isLocalDev ? 'http://localhost:8000' : window.location.origin;
const WS_BASE = isLocalDev 
  ? 'ws://localhost:8000/ws' 
  : (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/ws';
const HEATMAP_CELL_SIZE = 0.005; // degrees (~0.5km at mid-latitudes)
const DEFAULT_LAT = 47.5;
const DEFAULT_LNG = -121.8;
const DEFAULT_ZOOM = 13;

// Psychological profile defaults per subject type
const PSYCH_PROFILES = {
  hiker: {
    found_alive_rate: 0.96, avg_distance_km: 3.1, spiral_prob: 0.23,
    behavior: 'Hikers typically stay on or near trails when lost. They often backtrack along the original route and tend to seek high ground for orientation. Most signal for help actively and remain mobile.',
    advice: [
      'Search primary trail corridors and junctions first',
      'Check high viewpoints — hikers climb to orient',
      'Follow water downstream — natural evacuation route',
      'Search within 3km radius from last known point',
      'Look for directional clues like broken branches',
    ],
    urgency: 'MODERATE',
  },
  child: {
    found_alive_rate: 0.97, avg_distance_km: 1.2, spiral_prob: 0.15,
    behavior: 'Children under 6 rarely travel far. They typically hide when called, seek shelter, and move toward familiar sounds. Often found in dense cover or near water sources within 1.5km of LKP.',
    advice: [
      'Search concentric rings — children rarely go far',
      'Check hidden spots: logs, bushes, culverts',
      'Use familiar voices and pets in the search',
      'Check any water sources immediately',
      'Avoid large searcher groups — can frighten child',
    ],
    urgency: 'CRITICAL',
  },
  dementia: {
    found_alive_rate: 0.82, avg_distance_km: 4.5, spiral_prob: 0.61,
    behavior: 'Dementia patients travel with determined purpose to imagined destinations. They cross terrain barriers others avoid and often travel along linear features. High fatality risk from exposure.',
    advice: [
      'Check former home / workplace addresses first',
      'Focus on roads and paths — linear travel pattern',
      'Search drainages and creek beds urgently',
      'Time-critical: exposure risk within hours',
      'Use scent tracking dogs immediately',
    ],
    urgency: 'CRITICAL',
  },
  hunter: {
    found_alive_rate: 0.93, avg_distance_km: 5.2, spiral_prob: 0.31,
    behavior: 'Hunters have survival skills and equipment. They typically stay put after dark, use signals, and attempt self-rescue. May be injured or pinned down. Check game trails and hunting stands.',
    advice: [
      'Search game trails and hunting corridors',
      'Listen for gunshots — hunters signal with 3 shots',
      'Check for camouflaged shelters near tree stands',
      'Search creek bottoms during evening hours',
      'Use aerial assets at dawn for visual signal',
    ],
    urgency: 'MODERATE',
  },
};

// =====================
// STATE
// =====================
const state = {
  missionId: null,
  ws: null,
  frameCount: 0,
  detectionCount: 0,
  waypointCount: 0,
  showHeatmap: true,
  showRoute: true,
  showDetections: true,
  arizeInterval: null,
};

// =====================
// MAP REFERENCES
// =====================
let map = null;
let heatmapLayer = null;
let routeLayer = null;
let detectionLayer = null;
let droneTrailLayer = null;
let droneMarker = null;
let lkpMarker = null;
let lkpCircle = null;
const droneTrailPoints = [];

// =====================
// INIT
// =====================
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadPastMissions();
  pollArize();
  bindEvents();
  addLog('info', '🛰️', 'SAR AI Agent initialized', 'Ready to launch drone mission.');
});

// =====================
// MAP INITIALIZATION
// =====================
function initMap() {
  map = L.map('map', {
    center: [DEFAULT_LAT, DEFAULT_LNG],
    zoom: DEFAULT_ZOOM,
    zoomControl: true,
    attributionControl: true,
  });

  // Dark CartoDB tiles
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);

  // Layer groups
  heatmapLayer   = L.layerGroup().addTo(map);
  routeLayer     = L.layerGroup().addTo(map);
  detectionLayer = L.layerGroup().addTo(map);
  droneTrailLayer = L.layerGroup().addTo(map);

  // Last Known Position marker
  const lkpIcon = L.divIcon({
    className: '',
    html: `<div style="
      width:16px;height:16px;border-radius:50%;
      background:#ffd166;border:3px solid #fff;
      box-shadow:0 0 16px rgba(255,209,102,0.9),0 0 32px rgba(255,209,102,0.4);
      cursor:pointer;
    "></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
  });
  lkpMarker = L.marker([DEFAULT_LAT, DEFAULT_LNG], { icon: lkpIcon })
    .addTo(map)
    .bindPopup(`
      <b style="color:#ffd166">⚠ Last Known Position</b><br>
      Lat: ${DEFAULT_LAT}, Lng: ${DEFAULT_LNG}<br>
      <span style="color:#8892a4;font-size:10px;">Search origin</span>
    `);

  // Search radius circle (faint)
  lkpCircle = L.circle([DEFAULT_LAT, DEFAULT_LNG], {
    radius: 3000,
    color: 'rgba(255,107,53,0.3)',
    fillColor: 'rgba(255,107,53,0.03)',
    fillOpacity: 1,
    weight: 1,
    dashArray: '6, 4',
  }).addTo(map);
}

// =====================
// EVENT BINDINGS
// =====================
function bindEvents() {
  document.getElementById('mission-form').addEventListener('submit', onStartMission);

  // ── Time fields: auto-populate search start + live elapsed display ──────
  const searchStartEl = document.getElementById('search-start-time');
  const entryTimeEl   = document.getElementById('entry-time');
  const elapsedValEl  = document.getElementById('elapsed-value');
  const elapsedWinEl  = document.getElementById('elapsed-window');

  function toLocalDatetimeValue(date) {
    // Returns 'YYYY-MM-DDTHH:MM' for datetime-local input
    const pad = n => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  function hourToWindow(h) {
    if (h >= 5  && h < 8)  return 'dawn';
    if (h >= 8  && h < 12) return 'morning';
    if (h >= 12 && h < 17) return 'afternoon';
    if (h >= 17 && h < 20) return 'dusk';
    return 'night';
  }

  function updateElapsed() {
    const entry  = entryTimeEl.value   ? new Date(entryTimeEl.value)   : null;
    const search = searchStartEl.value ? new Date(searchStartEl.value) : new Date();
    if (!entry) { elapsedValEl.textContent = '—'; elapsedWinEl.textContent = ''; return; }
    const diffMs   = Math.max(0, search - entry);
    const totalMin = Math.floor(diffMs / 60000);
    const h = Math.floor(totalMin / 60);
    const m = totalMin % 60;
    elapsedValEl.textContent = `${h}h ${m}m`;
    const win = hourToWindow(search.getHours());
    elapsedWinEl.textContent = win;
    elapsedWinEl.className = `elapsed-window ${win}`;
  }

  // Set search-start to current local time on load
  searchStartEl.value = toLocalDatetimeValue(new Date());
  updateElapsed();

  entryTimeEl.addEventListener('change', updateElapsed);
  searchStartEl.addEventListener('change', updateElapsed);
  // Tick every minute to keep search-start current if not modified
  setInterval(() => {
    if (!searchStartEl.dataset.userEdited) {
      searchStartEl.value = toLocalDatetimeValue(new Date());
      updateElapsed();
    }
  }, 60000);
  searchStartEl.addEventListener('input', () => { searchStartEl.dataset.userEdited = '1'; });
  // ─────────────────────────────────────────────────────────────────────────

  document.getElementById('btn-refresh-missions').addEventListener('click', () => {
    loadPastMissions();
  });

  document.getElementById('btn-clear-log').addEventListener('click', () => {
    document.getElementById('log-feed').innerHTML = '<div class="log-empty">Log cleared.</div>';
  });

  document.getElementById('btn-show-heatmap').addEventListener('click', (e) => {
    state.showHeatmap = !state.showHeatmap;
    e.currentTarget.classList.toggle('active', state.showHeatmap);
    if (state.showHeatmap) { map.addLayer(heatmapLayer); }
    else { map.removeLayer(heatmapLayer); }
  });

  document.getElementById('btn-show-route').addEventListener('click', (e) => {
    state.showRoute = !state.showRoute;
    e.currentTarget.classList.toggle('active', state.showRoute);
    if (state.showRoute) { map.addLayer(routeLayer); }
    else { map.removeLayer(routeLayer); }
  });

  document.getElementById('btn-show-detections').addEventListener('click', (e) => {
    state.showDetections = !state.showDetections;
    e.currentTarget.classList.toggle('active', state.showDetections);
    if (state.showDetections) { map.addLayer(detectionLayer); }
    else { map.removeLayer(detectionLayer); }
  });
}

// =====================
// START MISSION
// =====================
async function onStartMission(e) {
  e.preventDefault();
  const btn = document.getElementById('btn-launch');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Launching...';

  resetUI();

  const subjectType   = document.getElementById('subject-type').value;
  const subjectName   = document.getElementById('subject-name').value.trim() || 'Unknown';
  const subjectAge    = parseInt(document.getElementById('subject-age').value) || null;
  const subjectGender = document.getElementById('subject-gender').value || null;
  const circumstance  = document.getElementById('circumstance').value;
  const nationality   = document.getElementById('subject-nationality').value.trim() || null;
  const eduLevel      = document.getElementById('edu-level').value || null;
  const eduField      = document.getElementById('edu-field').value || null;
  const subjectJob    = document.getElementById('subject-job').value || null;
  const missingHours  = parseFloat(document.getElementById('missing-hours').value) || 2.0;
  const terrainType   = document.getElementById('terrain-type').value;
  const notes         = document.getElementById('notes').value.trim();
  const lkpLat        = parseFloat(document.getElementById('lkp-lat').value) || DEFAULT_LAT;
  const lkpLng        = parseFloat(document.getElementById('lkp-lng').value) || DEFAULT_LNG;

  // Update map marker dynamically
  if (lkpMarker) {
    lkpMarker.setLatLng([lkpLat, lkpLng]);
    lkpMarker.getPopup().setContent(`
      <b style="color:#ffd166">⚠ Last Known Position</b><br>
      Lat: ${lkpLat}, Lng: ${lkpLng}<br>
      <span style="color:#8892a4;font-size:10px;">Search origin</span>
    `);
  }
  if (lkpCircle) {
    lkpCircle.setLatLng([lkpLat, lkpLng]);
  }
  map.setView([lkpLat, lkpLng], map.getZoom(), {animate: true, duration: 1.0});

  // Time fields — convert local datetime-local values to ISO 8601 strings
  const entryRaw  = document.getElementById('entry-time').value;
  const searchRaw = document.getElementById('search-start-time').value;
  const entryISO  = entryRaw  ? new Date(entryRaw).toISOString()  : null;
  const searchISO = searchRaw ? new Date(searchRaw).toISOString() : new Date().toISOString();

  const missionBody = {
    subject_type: subjectType,
    subject_name: subjectName,
    last_known_lat: lkpLat,
    last_known_lng: lkpLng,
    reported_missing_hours: missingHours,
    terrain_type: terrainType,
    notes: notes,
    subject_age: subjectAge,
    subject_gender: subjectGender,
    circumstance: circumstance,
    nationality: nationality,
    education_level: eduLevel,
    education_field: eduField,
    subject_job: subjectJob,
    entry_time: entryISO,
    search_start_time: searchISO,
  };

  try {
    // 1. Create mission
    const createResp = await fetch(`${API}/api/mission/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(missionBody),
    });
    if (!createResp.ok) throw new Error(`Mission create failed: ${createResp.status}`);
    const createData = await createResp.json();
    state.missionId = createData.mission_id;

    setStatus('initializing');
    addLog('info', '🚀', `Mission created`, `ID: ${state.missionId.substring(0, 12)}... — Subject: ${subjectName}`);

    // 2. Show behavioral profile from psych data
    updatePsychCard(subjectType, null);

    // 3. Connect WebSocket
    connectWebSocket(state.missionId);

    // 4. Fetch similar cases
    loadSimilarCases(subjectType, terrainType);

    // 5. Run pipeline
    const pipeResp = await fetch(`${API}/api/mission/${state.missionId}/run-pipeline`, {
      method: 'POST',
    });
    if (!pipeResp.ok) throw new Error(`Pipeline start failed: ${pipeResp.status}`);

    addLog('info', '🛸', 'Pipeline running', `Analyzing drone footage with AI vision...`);

  } catch (err) {
    console.error('Mission start error:', err);
    setStatus('error');
    addLog('high', '❌', 'Mission start failed', err.message);
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🚁</span> Launch Drone Mission';
  }
}

// =====================
// WEBSOCKET
// =====================
function connectWebSocket(missionId) {
  if (state.ws) {
    try { state.ws.close(); } catch {}
    state.ws = null;
  }

  const url = `${WS_BASE}/${missionId}`;
  const ws = new WebSocket(url);
  state.ws = ws;

  ws.onopen = () => {
    addLog('info', '📡', 'WebSocket connected', `Live feed active — ${url}`);
  };

  ws.onclose = (e) => {
    addLog('info', '🔌', 'WebSocket closed', `Code: ${e.code}`);
  };

  ws.onerror = () => {
    addLog('high', '⚠', 'WebSocket error', 'Connection to drone feed failed');
    setStatus('error');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleWsMessage(msg);
    } catch (err) {
      console.warn('WS parse error:', err, event.data);
    }
  };
}

function handleWsMessage(msg) {
  switch (msg.type) {

    case 'status_update':
      setStatus(msg.status || 'running');
      if (typeof msg.progress === 'number') {
        setProgress(msg.progress, msg.message || '');
      } else if (msg.message) {
        addLog('info', '⚡', msg.message, '');
      }
      break;

    case 'frame_analyzed':
      state.frameCount = msg.frame_number || state.frameCount + 1;
      updateStat('stat-frames', state.frameCount);
      animateDrone(msg.drone_lat, msg.drone_lng);

      if (Array.isArray(msg.detections) && msg.detections.length > 0) {
        msg.detections.forEach(det => {
          state.detectionCount++;
          addDetectionMarker(det);
          const sig = det.significance || 'low';
          addLog(
            sig,
            trackTypeIcon(det.track_type),
            `[Frame ${msg.frame_number}/${msg.total_frames}] ${formatTrackType(det.track_type)} — ${sig.toUpperCase()}`,
            `${det.description} · Confidence: ${pct(det.confidence)} · ${formatTerrain(msg.terrain_type)}`
          );
        });
        updateStat('stat-detections', state.detectionCount);
      } else {
        // No detections this frame
        addLog('info', '🌲', `[Frame ${msg.frame_number}/${msg.total_frames}] ${formatTerrain(msg.terrain_type)}`, msg.summary || 'No detections.');
      }
      break;

    case 'heatmap_updated':
      if (Array.isArray(msg.cells)) renderHeatmap(msg.cells);
      if (msg.behavioral_insights) {
        updatePsychCard(null, msg.behavioral_insights);
      }
      break;

    case 'route_updated':
      if (Array.isArray(msg.waypoints)) {
        renderRoute(msg.waypoints);
        state.waypointCount = msg.waypoints.length;
        updateStat('stat-waypoints', state.waypointCount);
        renderWaypointList(msg.waypoints);
        addLog('info', '📍', `Route updated`, `${msg.waypoints.length} waypoints generated`);
      }
      break;

    case 'pipeline_complete': {
      setStatus('completed');
      setProgress(1.0, '');
      const s = msg.summary || {};
      addLog('info', '✅', 'Mission pipeline complete!',
        `${s.frames_analyzed || 0} frames · ${s.total_detections || 0} detections · ${s.waypoints_generated || 0} waypoints`
      );
      renderSummary(s);
      loadPastMissions();
      pollArize();
      document.getElementById('btn-launch').disabled = false;
      document.getElementById('btn-launch').innerHTML = '<span class="btn-icon">🚁</span> Launch Drone Mission';
      break;
    }

    case 'error':
      setStatus('error');
      addLog('high', '❌', 'Pipeline error', msg.message || 'Unknown error');
      document.getElementById('btn-launch').disabled = false;
      document.getElementById('btn-launch').innerHTML = '<span class="btn-icon">🚁</span> Launch Drone Mission';
      break;

    default:
      console.log('Unknown WS message type:', msg.type, msg);
  }
}

// =====================
// MAP RENDERING
// =====================
function animateDrone(lat, lng) {
  if (typeof lat !== 'number' || typeof lng !== 'number') return;

  const droneHtml = `
    <div style="position:relative;width:28px;height:28px;">
      <div style="
        width:28px;height:28px;border-radius:50%;
        background:radial-gradient(circle, rgba(255,107,53,0.9) 0%, rgba(255,107,53,0.4) 70%);
        display:flex;align-items:center;justify-content:center;
        font-size:16px;
        box-shadow:0 0 20px rgba(255,107,53,0.8), 0 0 40px rgba(255,107,53,0.3);
        z-index:2;position:relative;
      ">🚁</div>
      <div style="
        position:absolute;top:0;left:0;
        width:28px;height:28px;border-radius:50%;
        border:2px solid rgba(255,107,53,0.7);
        animation:drone-pulse 1.5s ease-out infinite;
      "></div>
    </div>
  `;

  const droneIcon = L.divIcon({
    className: '',
    html: droneHtml,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });

  if (droneMarker) {
    droneMarker.setLatLng([lat, lng]);
    droneMarker.setIcon(droneIcon);
  } else {
    droneMarker = L.marker([lat, lng], { icon: droneIcon, zIndexOffset: 1000 }).addTo(map);
  }

  // Add trail dot
  droneTrailPoints.push([lat, lng]);
  L.circleMarker([lat, lng], {
    radius: 2.5,
    color: 'rgba(255,107,53,0.0)',
    fillColor: 'rgba(255,107,53,0.5)',
    fillOpacity: 0.5,
    weight: 0,
  }).addTo(droneTrailLayer);

  // Draw trail polyline
  if (droneTrailPoints.length > 1) {
    droneTrailLayer.eachLayer(l => {
      if (l instanceof L.Polyline && !(l instanceof L.CircleMarker)) {
        droneTrailLayer.removeLayer(l);
      }
    });
    L.polyline(droneTrailPoints, {
      color: 'rgba(255,107,53,0.35)',
      weight: 1.5,
      dashArray: '4,4',
    }).addTo(droneTrailLayer);
  }
}

function renderHeatmap(cells) {
  heatmapLayer.clearLayers();
  if (!cells || cells.length === 0) return;

  const half = HEATMAP_CELL_SIZE / 2;

  cells.forEach(cell => {
    const p = typeof cell.probability === 'number' ? cell.probability : 0;
    const color = probabilityColor(p);
    const baseOpacity = 0.10 + p * 0.60;

    const rect = L.rectangle(
      [
        [cell.lat - half, cell.lng - half],
        [cell.lat + half, cell.lng + half],
      ],
      {
        color: 'transparent',
        fillColor: color,
        fillOpacity: baseOpacity,
        weight: 0,
        className: 'heatmap-cell',
      }
    ).addTo(heatmapLayer);

    const detScore = cell.detection_score !== undefined ? `${(cell.detection_score * 100).toFixed(0)}%` : 'N/A';
    rect.bindPopup(`
      <b>Zone ${cell.zone_id}</b><br>
      Probability: <b style="color:${color}">${(p * 100).toFixed(1)}%</b><br>
      Terrain: ${formatTerrain(cell.terrain_type || '')}<br>
      Detection score: ${detScore}
    `);
  });
}

function renderRoute(waypoints) {
  routeLayer.clearLayers();
  if (!waypoints || waypoints.length === 0) return;

  const latlngs = waypoints.map(w => [w.lat, w.lng]);

  // Dashed route line
  L.polyline(latlngs, {
    color: '#4a90d9',
    weight: 2.5,
    opacity: 0.75,
    dashArray: '10, 7',
    lineJoin: 'round',
  }).addTo(routeLayer);

  // Subtle glow line underneath
  L.polyline(latlngs, {
    color: '#4a90d9',
    weight: 6,
    opacity: 0.10,
  }).addTo(routeLayer);

  // Waypoint markers
  waypoints.forEach((wp) => {
    const priority = (wp.priority || 'medium').toLowerCase();
    const color = priorityColor(priority);
    const order = wp.order !== undefined ? wp.order : '?';

    const icon = L.divIcon({
      className: '',
      html: `
        <div style="
          width:28px;height:28px;border-radius:50%;
          background:${color}22;border:2px solid ${color};
          display:flex;align-items:center;justify-content:center;
          font-size:11px;font-weight:700;color:${color};
          font-family:monospace;
          box-shadow:0 0 12px ${color}55;
        ">${order}</div>
      `,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
    });

    L.marker([wp.lat, wp.lng], { icon })
      .addTo(routeLayer)
      .bindPopup(`
        <b style="color:${color}">WP${order} — Zone ${wp.zone_id}</b><br>
        Priority: <b style="color:${color}">${priority.toUpperCase()}</b><br>
        Probability: ${pct(wp.probability)}<br>
        Terrain: ${formatTerrain(wp.terrain_type || '')}<br>
        Est. search time: ${wp.estimated_search_time_min || '?'} min<br>
        <span style="color:#8892a4;font-size:10px;">${wp.reason || ''}</span>
      `);
  });

  // Fly to route bounds
  if (latlngs.length > 1) {
    const bounds = L.latLngBounds(latlngs);
    setTimeout(() => map.fitBounds(bounds.pad(0.3)), 250);
  }
}

function addDetectionMarker(det) {
  const loc = det.location;
  if (!loc || typeof loc.lat !== 'number' || typeof loc.lng !== 'number') return;

  const sig = det.significance || 'low';
  const color = significanceColor(sig);
  const icon = trackTypeIcon(det.track_type);

  const divIcon = L.divIcon({
    className: '',
    html: `
      <div style="
        width:12px;height:12px;border-radius:50%;
        background:${color};border:2px solid rgba(255,255,255,0.8);
        box-shadow:0 0 10px ${color}, 0 0 20px ${color}55;
        opacity:0.92;
      "></div>
    `,
    iconSize: [12, 12],
    iconAnchor: [6, 6],
  });

  const ageStr = det.estimated_age_hours !== undefined ? `${det.estimated_age_hours}h ago` : 'Unknown age';

  L.marker([loc.lat, loc.lng], { icon: divIcon })
    .addTo(detectionLayer)
    .bindPopup(`
      <b style="color:${color}">${icon} ${formatTrackType(det.track_type)}</b><br>
      Confidence: <b style="color:${color}">${pct(det.confidence)}</b><br>
      Significance: ${sig.toUpperCase()}<br>
      ${det.description || ''}<br>
      <span style="color:#8892a4;font-size:10px;">Est. age: ${ageStr}</span>
    `);
}

// =====================
// PROBABILITY COLOR
// =====================
function probabilityColor(p) {
  if (p >= 0.75) return '#ef4444';   // red — very high
  if (p >= 0.50) return '#ff6b35';   // orange — high
  if (p >= 0.25) return '#ffd166';   // yellow — medium
  return '#00d4aa';                  // teal — low
}

function priorityColor(priority) {
  switch (priority) {
    case 'critical': return '#ef4444';
    case 'high':     return '#ff6b35';
    case 'medium':   return '#ffd166';
    default:         return '#00d4aa';
  }
}

function significanceColor(sig) {
  switch (sig) {
    case 'high':   return '#ff6b35';
    case 'medium': return '#ffd166';
    default:       return '#00d4aa';
  }
}

// =====================
// UI HELPERS
// =====================
function resetUI() {
  state.frameCount = 0;
  state.detectionCount = 0;
  state.waypointCount = 0;
  droneTrailPoints.length = 0;

  updateStat('stat-frames', 0);
  updateStat('stat-detections', 0);
  updateStat('stat-waypoints', 0);

  heatmapLayer.clearLayers();
  routeLayer.clearLayers();
  detectionLayer.clearLayers();
  droneTrailLayer.clearLayers();

  if (droneMarker) {
    try { map.removeLayer(droneMarker); } catch {}
    droneMarker = null;
  }

  document.getElementById('log-feed').innerHTML = '<div class="log-empty">Awaiting drone data...</div>';
  document.getElementById('waypoints-list').innerHTML = '<p class="empty-msg">Route not yet generated.</p>';
  document.getElementById('cases-list').innerHTML = '<p class="empty-msg">Loading similar cases...</p>';

  const summaryCard = document.getElementById('summary-card');
  if (summaryCard) summaryCard.style.display = 'none';

  setProgress(0, '');
}

function setStatus(s) {
  const pill = document.getElementById('status-pill');
  if (!pill) return;
  pill.className = `status-pill ${s}`;
  pill.textContent = s.toUpperCase();
}

function setProgress(fraction, message) {
  const container = document.getElementById('progress-container');
  const bar       = document.getElementById('progress-bar');
  const text      = document.getElementById('progress-text');
  if (!container || !bar || !text) return;

  const pctVal = Math.min(100, Math.round(fraction * 100));
  container.style.display = fraction > 0 ? 'flex' : 'none';
  bar.style.setProperty('--progress', `${pctVal}%`);
  text.textContent = `${pctVal}%`;
}

function updateStat(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val;
  el.classList.remove('updated');
  void el.offsetWidth; // reflow
  el.classList.add('updated');
}

function addLog(level, icon, main, sub) {
  const feed = document.getElementById('log-feed');
  if (!feed) return;

  // Remove placeholder
  const empty = feed.querySelector('.log-empty');
  if (empty) empty.remove();

  const now = new Date();
  const timeStr = now.toTimeString().slice(0, 8);

  const entry = document.createElement('div');
  const normalizedLevel = ['high', 'medium', 'low'].includes(level) ? level : 'info';
  entry.className = `log-entry ${normalizedLevel}`;

  const badgeHtml = normalizedLevel !== 'info'
    ? `<span class="log-badge ${normalizedLevel}">${normalizedLevel.toUpperCase()}</span>`
    : '';

  entry.innerHTML = `
    <span class="log-icon">${icon}</span>
    <span class="log-text">
      <div class="log-main">${escHtml(main)}</div>
      ${sub ? `<div class="log-sub">${escHtml(sub)}</div>` : ''}
    </span>
    ${badgeHtml}
    <span class="log-time">${timeStr}</span>
  `;

  feed.prepend(entry);

  // Keep log manageable
  const entries = feed.querySelectorAll('.log-entry');
  if (entries.length > 80) {
    entries[entries.length - 1].remove();
  }
}

// =====================
// BEHAVIORAL / PSYCH PROFILE
// =====================
function updatePsychCard(subjectType, backendInsights) {
  const card = document.getElementById('psych-card');
  if (!card) return;
  card.style.display = 'block';

  const profile = subjectType ? (PSYCH_PROFILES[subjectType] || PSYCH_PROFILES.hiker) : null;

  // Merge backend data with local defaults
  const aliveRate = backendInsights?.found_alive_rate ?? profile?.found_alive_rate ?? 0.9;
  const distance  = backendInsights?.effective_distance_km ?? backendInsights?.avg_distance_km ?? profile?.avg_distance_km ?? 3.0;
  const spiral    = backendInsights?.panic_spiral_prob ?? profile?.spiral_prob ?? 0.25;
  const behavior  = backendInsights?.typical_behavior ?? profile?.behavior ?? '';
  const advice    = backendInsights?.key_advice ?? profile?.advice ?? [];
  const urgency   = backendInsights?.urgency ?? profile?.urgency ?? 'MODERATE';

  // Update stats
  document.getElementById('ps-alive-rate').textContent = `${(aliveRate * 100).toFixed(0)}%`;
  document.getElementById('ps-distance').textContent   = `${Number(distance).toFixed(1)} km`;
  document.getElementById('ps-spiral').textContent     = `${(spiral * 100).toFixed(0)}%`;
  document.getElementById('psych-behavior').textContent = behavior;

  // Urgency badge
  const badge = document.getElementById('urgency-badge');
  badge.className = `urgency-badge ${urgency}`;
  badge.textContent = urgency;

  // Language barrier warning
  if (backendInsights?.language_barrier) {
    badge.title = '⚠️ Language barrier detected for this nationality';
  }

  // Advice items (now includes job, edu, nationality, historical advice)
  const adviceList = document.getElementById('advice-list');
  adviceList.innerHTML = '';
  (Array.isArray(advice) ? advice : []).slice(0, 10).forEach(tip => {
    const item = document.createElement('div');
    item.className = 'advice-item';
    // Highlight language barrier, job, factor notes, and historical data differently
    if (tip.startsWith('⚠️') || tip.startsWith('Age factor') || tip.startsWith('Gender') ||
        tip.startsWith('Education') || tip.startsWith('[') || tip.startsWith('Historical')) {
      item.style.borderLeftColor = 'var(--yellow)';
      item.style.color = 'var(--yellow)';
    }
    if (tip.startsWith('Historical data')) {
      item.style.borderLeftColor = 'var(--teal)';
      item.style.color = 'var(--teal)';
    }
    item.textContent = tip;
    adviceList.appendChild(item);
  });

  // Path prediction panel
  const pp = document.getElementById('path-prediction');
  const pred = backendInsights?.path_prediction;
  if (pp && pred) {
    pp.style.display = 'block';
    document.getElementById('pp-conf').textContent    = `${Math.round((pred.confidence || 0) * 100)}% confidence`;
    document.getElementById('pp-bearing').textContent = `${pred.bearing_deg}°`;
    document.getElementById('pp-direction').textContent = pred.cardinal_direction || '—';
    document.getElementById('pp-tracks').textContent  = `${pred.detection_count_used} track points`;
    const alert = document.getElementById('pp-alert');
    if (alert) alert.style.display = pred.direction_change_detected ? 'block' : 'none';
  } else if (pp) {
    pp.style.display = 'none';
  }

  // Historical context panel
  const histPanel = document.getElementById('historical-prediction');
  const hist = backendInsights?.historical_context;
  if (histPanel && hist && hist.has_data) {
    histPanel.style.display = 'block';
    document.getElementById('hist-count').textContent = `${hist.similar_cases_count} matching cases`;
    document.getElementById('hist-alive').textContent = `${(hist.found_alive_rate * 100).toFixed(0)}%`;
    document.getElementById('hist-dist').textContent  = `${hist.avg_distance_km} km`;
    document.getElementById('hist-time').textContent  = `${hist.avg_hours_to_find} hrs`;
    const histNote = document.getElementById('hist-note');
    if (histNote && hist.historical_note) {
      histNote.style.display = 'block';
      histNote.textContent = hist.historical_note;
    }
  } else if (histPanel) {
    histPanel.style.display = 'none';
  }
}

// =====================
// SIMILAR CASES (Elastic)
// =====================
async function loadSimilarCases(subjectType, terrain) {
  const list = document.getElementById('cases-list');
  if (!list) return;
  list.innerHTML = '<p class="empty-msg">Searching case database...</p>';

  try {
    const url = `${API}/api/cases/similar?subject_type=${encodeURIComponent(subjectType)}&terrain=${encodeURIComponent(terrain)}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    renderCases(data.cases || []);
  } catch (err) {
    list.innerHTML = `<p class="empty-msg">Could not load cases.<br><span style="font-size:9px;color:#4a5568">${err.message}</span></p>`;
  }
}

function renderCases(cases) {
  const list = document.getElementById('cases-list');
  if (!list) return;

  if (!cases.length) {
    list.innerHTML = '<p class="empty-msg">No similar cases found.</p>';
    return;
  }

  list.innerHTML = '';
  cases.slice(0, 6).forEach(c => {
    const outcome = (c.outcome || 'unknown').toLowerCase().replace(' ', '_');
    const outcomeLbl = c.outcome || 'Unknown';
    const factors = Array.isArray(c.contributing_factors) ? c.contributing_factors : [];

    const item = document.createElement('div');
    item.className = 'case-item';
    item.innerHTML = `
      <div class="ci-header">
        <span class="ci-id">${escHtml(c.case_id || c.id || 'CASE-???')}</span>
        <span class="ci-outcome ${outcome}">${escHtml(outcomeLbl)}</span>
      </div>
      <div class="ci-area">${escHtml(c.area || c.location || 'Unknown area')}</div>
      <div class="ci-meta">
        ${c.subject_age ? `Age: ${c.subject_age}` : ''}
        ${c.duration_hours ? ` · ${c.duration_hours}h missing` : ''}
        ${c.distance_km ? ` · Found ${c.distance_km}km away` : ''}
      </div>
      <div class="ci-factors">
        ${factors.slice(0, 4).map(f => `<span class="ci-factor">${escHtml(f)}</span>`).join('')}
      </div>
    `;
    list.appendChild(item);
  });
}

// =====================
// WAYPOINT LIST
// =====================
function renderWaypointList(waypoints) {
  const list = document.getElementById('waypoints-list');
  if (!list) return;
  list.innerHTML = '';

  waypoints.forEach((wp, idx) => {
    const priority = (wp.priority || 'medium').toLowerCase();
    const probVal  = typeof wp.probability === 'number' ? wp.probability : 0;
    const probStr  = pct(probVal);

    const item = document.createElement('div');
    item.className = 'wp-item';
    item.innerHTML = `
      <div class="wp-num ${priority}">${wp.order !== undefined ? wp.order : idx + 1}</div>
      <div class="wp-body">
        <div class="wp-zone">Zone ${escHtml(String(wp.zone_id || '?'))}</div>
        <div class="wp-reason">${escHtml(wp.reason || '')}</div>
        <div class="wp-meta">${formatTerrain(wp.terrain_type || '')} · ${wp.estimated_search_time_min || '?'} min</div>
      </div>
      <div class="wp-prob ${priority}">${probStr}</div>
    `;

    // Click to fly to waypoint on map
    item.style.cursor = 'pointer';
    item.addEventListener('click', () => {
      if (typeof wp.lat === 'number' && typeof wp.lng === 'number') {
        map.flyTo([wp.lat, wp.lng], 15, { duration: 1 });
      }
    });

    list.appendChild(item);
  });
}

// =====================
// MISSION SUMMARY
// =====================
function renderSummary(summary) {
  const card = document.getElementById('summary-card');
  const body = document.getElementById('summary-body');
  if (!card || !body) return;

  const bi = summary.behavioral_insights || {};
  const items = [
    { label: '📸 Frames analyzed',      val: summary.frames_analyzed      ?? '—' },
    { label: '👣 Total detections',     val: summary.total_detections      ?? '—' },
    { label: '📍 Waypoints generated',  val: summary.waypoints_generated   ?? '—' },
    { label: '🔴 Found alive rate',     val: bi.found_alive_rate ? `${(bi.found_alive_rate*100).toFixed(0)}%` : '—' },
    { label: '📏 Avg. distance',        val: bi.avg_distance_km ? `${bi.avg_distance_km} km` : '—' },
    { label: '⚠ Urgency level',        val: bi.urgency ?? '—' },
  ];

  body.innerHTML = '';
  items.forEach(({ label, val }) => {
    const row = document.createElement('div');
    row.className = 'summary-stat';
    row.innerHTML = `
      <span class="ss-label">${escHtml(label)}</span>
      <span class="ss-val">${escHtml(String(val))}</span>
    `;
    body.appendChild(row);
  });

  card.style.display = 'block';
}

// =====================
// PAST MISSIONS
// =====================
async function loadPastMissions() {
  const list = document.getElementById('missions-list');
  if (!list) return;

  try {
    const resp = await fetch(`${API}/api/missions`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const missions = Array.isArray(data) ? data : (data.missions || []);

    if (!missions.length) {
      list.innerHTML = '<p class="empty-msg">No missions yet.</p>';
      return;
    }

    list.innerHTML = '';
    missions.slice().reverse().slice(0, 12).forEach(m => {
      const status   = (m.status || 'idle').toLowerCase();
      const subject  = m.subject_name || m.subject_type || 'Unknown';
      const terrain  = formatTerrain(m.terrain_type || '');
      const created  = m.created_at ? new Date(m.created_at).toLocaleString() : '';

      const item = document.createElement('div');
      item.className = `mission-item${m.mission_id === state.missionId ? ' active' : ''}`;
      item.innerHTML = `
        <div class="mi-subject">${escHtml(subject)}</div>
        <div class="mi-meta">${escHtml(terrain)} · ${escHtml(created)}</div>
        <div class="mi-status ${status}">${status.toUpperCase()}</div>
      `;

      item.addEventListener('click', () => loadMissionDetails(m.mission_id));
      list.appendChild(item);
    });

  } catch (err) {
    list.innerHTML = `<p class="empty-msg">Backend offline.<br><span style="font-size:9px">${err.message}</span></p>`;
  }
}

async function loadMissionDetails(missionId) {
  if (!missionId) return;
  try {
    const [missionResp, heatmapResp, routeResp] = await Promise.all([
      fetch(`${API}/api/mission/${missionId}`),
      fetch(`${API}/api/mission/${missionId}/heatmap`),
      fetch(`${API}/api/mission/${missionId}/route`),
    ]);

    if (missionResp.ok) {
      const m = await missionResp.json();
      setStatus(m.status || 'idle');
      if (m.subject_type) updatePsychCard(m.subject_type, null);
    }
    if (heatmapResp.ok) {
      const hd = await heatmapResp.json();
      if (Array.isArray(hd.cells)) renderHeatmap(hd.cells);
    }
    if (routeResp.ok) {
      const rd = await routeResp.json();
      if (Array.isArray(rd.waypoints)) {
        renderRoute(rd.waypoints);
        renderWaypointList(rd.waypoints);
        updateStat('stat-waypoints', rd.waypoints.length);
      }
    }

    // Highlight in list
    document.querySelectorAll('.mission-item').forEach(el => el.classList.remove('active'));
    state.missionId = missionId;
    addLog('info', '📋', `Loaded mission ${missionId.substring(0, 8)}`, 'Map updated with mission data.');

  } catch (err) {
    addLog('high', '❌', 'Failed to load mission details', err.message);
  }
}

// =====================
// ARIZE STATS POLL
// =====================
async function pollArize() {
  try {
    const resp = await fetch(`${API}/api/arize/stats`);
    if (!resp.ok) return;
    const data = await resp.json();
    const el = document.getElementById('arize-pred-count');
    if (el) {
      el.textContent = data.total_predictions ?? data.prediction_count ?? '—';
    }
  } catch {}

  // Schedule next poll in 10s
  setTimeout(pollArize, 10000);
}

// =====================
// FORMAT HELPERS
// =====================
function formatTrackType(t) {
  return (t || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatTerrain(t) {
  return (t || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function pct(val) {
  if (typeof val !== 'number') return '—';
  return `${(val * 100).toFixed(0)}%`;
}

function trackTypeIcon(trackType) {
  const t = (trackType || '').toLowerCase();
  if (t.includes('footprint') || t.includes('foot')) return '👣';
  if (t.includes('shelter') || t.includes('camp'))   return '⛺';
  if (t.includes('clothing') || t.includes('garment')) return '👕';
  if (t.includes('person') || t.includes('human'))   return '🧍';
  if (t.includes('fire') || t.includes('smoke'))     return '🔥';
  if (t.includes('blood') || t.includes('injury'))   return '🩸';
  if (t.includes('vehicle') || t.includes('car'))    return '🚗';
  if (t.includes('trail') || t.includes('path'))     return '🛤️';
  return '🔍';
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
