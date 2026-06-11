"""
psych_model.py — Enhanced Lost Person Behavioral Model
Based on Mattson's "Lost Person Behavior" (2011) + Syrotuck's gender/age data.

Enhancements:
  - Age modifiers: movement radius, terrain capability, decision quality
  - Gender modifiers: distance bias, risk-taking, trail preference
  - Nationality modifiers: cultural factors affecting SAR behaviour (response
    to authority, emergency-service familiarity, outdoor experience norms,
    language barriers, fatalism/shelter-seeking vs self-rescue tendencies)
  - Education modifiers: reasoning quality under stress, panic probability,
    signalling behaviour, decision logic (technical vs non-technical degrees)
  - Circumstance modifiers: injury, intoxication, suicidal ideation, medical
    emergency, experience level, time-of-day, alone vs group
  - Trace-based path prediction: infers direction of travel from sequential
    detection locations and projects likely next positions
"""

import math
from typing import List, Dict, Optional
from opentelemetry import trace
tracer = trace.get_tracer(__name__)
import statistics
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# BASE BEHAVIORAL PROFILES  (Mattson 2011)
# ──────────────────────────────────────────────────────────────────────────────
SUBJECT_BEHAVIOR_PROFILES = {
    "hiker": {
        "description": "Recreational hiker",
        "attractors": {
            "downhill": 0.72, "trail_following": 0.68, "water": 0.55, "shelter": 0.45,
        },
        "avg_distance_km": 3.1,
        "panic_spiral_prob": 0.23,
        "found_alive_rate": 0.96,
        "urgency": "MODERATE",
        "typical_behavior": "Initially continues hiking direction. May attempt self-rescue by heading downhill toward roads or water.",
        "key_advice": [
            "Search downhill from last known point first",
            "Check water sources and creek drainages",
            "Broadcast voice calls — subject will likely respond",
        ],
    },
    "child": {
        "description": "Child under 12, disoriented",
        "attractors": {
            "shelter": 0.82, "water": 0.40, "downhill": 0.30, "play_areas": 0.60,
        },
        "avg_distance_km": 1.0,
        "panic_spiral_prob": 0.15,
        "found_alive_rate": 0.97,
        "urgency": "HIGH",
        "typical_behavior": "Will shelter in place when exhausted. Does NOT respond to calls — children hide from strangers.",
        "key_advice": [
            "Thorough search within 1.5 km before expanding",
            "Do NOT rely on voice calls — children hide from strangers",
            "Check sheltered spots: under logs, dense brush, culverts",
        ],
    },
    "dementia": {
        "description": "Person with dementia/Alzheimer's — URGENT",
        "attractors": {
            "road_following": 0.85, "circular_wandering": 0.70, "water": 0.30, "downhill": 0.45,
        },
        "avg_distance_km": 1.5,
        "panic_spiral_prob": 0.55,
        "found_alive_rate": 0.69,
        "urgency": "CRITICAL",
        "typical_behavior": "Circular wandering. Attracted to roads. HIGH URGENCY — medical deterioration risk within 24 h.",
        "key_advice": [
            "URGENT: High medical risk after 24 h exposure",
            "Focus on roads, driveways, familiar-looking paths",
            "Check nearby residences — subject may seek familiar settings",
        ],
    },
    "hunter": {
        "description": "Hunter with wilderness experience",
        "attractors": {
            "ridgeline": 0.65, "downhill": 0.40, "trail_following": 0.50, "shelter": 0.70,
        },
        "avg_distance_km": 4.2,
        "panic_spiral_prob": 0.12,
        "found_alive_rate": 0.98,
        "urgency": "MODERATE",
        "typical_behavior": "Will attempt self-rescue. Seeks high ground. May signal. Wider search area.",
        "key_advice": [
            "Expand search radius — hunters travel further",
            "Check ridgelines and elevated viewpoints",
            "Look for signal fires or improvised shelters",
        ],
    },
}

TERRAIN_ATTRACTOR_MAP = {
    "creek_side": "water",
    "trail": "trail_following",
    "steep_slope": "downhill",
    "dense_forest": "shelter",
    "rocky_outcrop": "ridgeline",
    "open_field": "road_following",
    "mixed_forest": "shelter",
}

# ──────────────────────────────────────────────────────────────────────────────
# AGE MODIFIERS  (multipliers applied to avg_distance_km and terrain scores)
# ──────────────────────────────────────────────────────────────────────────────
def _age_modifiers(age: Optional[int], subject_type: str) -> Dict:
    """
    Returns distance_mult, steep_penalty, spiral_delta based on age.
    Sources: Syrotuck (1976), Mattson (2011) age-stratified data.
    """
    if age is None:
        return {"distance_mult": 1.0, "steep_penalty": 0.0, "spiral_delta": 0.0,
                "age_note": "Age unknown — using base profile"}

    # Child sub-ranges (Mattson chapter 3)
    if subject_type == "child":
        if age <= 3:
            return {"distance_mult": 0.4, "steep_penalty": 0.4, "spiral_delta": -0.05,
                    "age_note": "Toddler (≤3): very short range, likely crawled into cover"}
        elif age <= 6:
            return {"distance_mult": 0.65, "steep_penalty": 0.3, "spiral_delta": 0.0,
                    "age_note": "Young child (4–6): short range, attracted to interesting objects"}
        elif age <= 12:
            return {"distance_mult": 1.0, "steep_penalty": 0.15, "spiral_delta": 0.05,
                    "age_note": "Older child (7–12): full child range, may deliberately hide"}
    
    # General age buckets
    if age < 18:
        return {"distance_mult": 1.1, "steep_penalty": 0.0, "spiral_delta": -0.05,
                "age_note": f"Teen ({age}): energetic, may travel further, lower panic rate"}
    elif age <= 40:
        return {"distance_mult": 1.0, "steep_penalty": 0.0, "spiral_delta": 0.0,
                "age_note": f"Young adult ({age}): baseline profile, good physical capability"}
    elif age <= 60:
        return {"distance_mult": 0.85, "steep_penalty": 0.15, "spiral_delta": 0.03,
                "age_note": f"Middle-aged ({age}): slightly reduced range, avoids very steep terrain"}
    elif age <= 75:
        return {"distance_mult": 0.65, "steep_penalty": 0.30, "spiral_delta": 0.08,
                "age_note": f"Older adult ({age}): reduced mobility, steep slopes avoided, higher confusion risk"}
    else:
        return {"distance_mult": 0.45, "steep_penalty": 0.50, "spiral_delta": 0.15,
                "age_note": f"Elderly ({age}): very limited range, cannot handle slopes, HIGH urgency adjustment"}

# ──────────────────────────────────────────────────────────────────────────────
# GENDER MODIFIERS  (Syrotuck 1976, Koester 2008)
# ──────────────────────────────────────────────────────────────────────────────
def _gender_modifiers(gender: Optional[str]) -> Dict:
    """
    Small but statistically real adjustments from SAR data.
    Male subjects travel ~18% further on average (Syrotuck 1976).
    Female subjects more likely to stay on established paths.
    """
    g = (gender or "").lower().strip()
    if g in ("male", "m", "man", "boy"):
        return {
            "distance_mult": 1.18,
            "trail_boost": 0.0,
            "ridgeline_boost": 0.08,
            "gender_note": "Male: statistically travels ~18% further, higher ridgeline/high-ground tendency",
        }
    elif g in ("female", "f", "woman", "girl"):
        return {
            "distance_mult": 0.82,
            "trail_boost": 0.12,
            "ridgeline_boost": 0.0,
            "gender_note": "Female: shorter avg. travel distance, stronger trail-following tendency",
        }
    else:
        return {
            "distance_mult": 1.0,
            "trail_boost": 0.0,
            "ridgeline_boost": 0.0,
            "gender_note": "Gender unknown — using base profile",
        }


# ──────────────────────────────────────────────────────────────────────────────
# NATIONALITY / CULTURAL BACKGROUND MODIFIERS
# ──────────────────────────────────────────────────────────────────────────────
# Groups based on cultural SAR behaviour research:
#   - Familiarity with formal SAR/emergency systems
#   - Outdoor wilderness experience norms in the culture
#   - Tendency to call for help early vs attempt self-rescue
#   - Language barrier affecting response to voice calls
#   - Fatalism vs proactive survival mindset
# Sources: Koester (2008) International SAR database, regional field reports.

NATIONALITY_GROUPS = {
    # Western / Northern Europe + Anglophone (high SAR-system familiarity)
    "western": {
        "countries": {
            "usa", "united states", "uk", "united kingdom", "canada", "australia",
            "new zealand", "ireland", "england", "scotland", "wales",
            "germany", "france", "netherlands", "belgium", "austria", "switzerland",
            "denmark", "sweden", "norway", "finland", "iceland",
        },
        "distance_mult": 1.05,
        "spiral_delta": -0.05,
        "trail_boost": 0.05,
        "language_barrier": False,
        "sar_note": "Culturally familiar with SAR systems — likely to call for help, respond to voice signals, and stay calm.",
        "advice": ["Subject will likely activate personal locator beacon if carried",
                   "Respond well to voice signals and organised rescue protocol"],
    },
    # South Asian (India, Pakistan, Bangladesh, Nepal, Sri Lanka)
    "south_asian": {
        "countries": {
            "india", "pakistan", "bangladesh", "nepal", "sri lanka",
            "bhutan", "maldives",
        },
        "distance_mult": 0.88,
        "spiral_delta": 0.08,
        "trail_boost": 0.10,
        "language_barrier": True,
        "sar_note": (
            "May be unfamiliar with wilderness SAR protocols. Strong group/family orientation — "
            "may wait for family to locate them. Trail-following tendency elevated. "
            "Language barrier may prevent response to voice calls in English."
        ),
        "advice": ["Language barrier likely — use multilingual broadcast if possible",
                   "May be waiting for family — contact next-of-kin immediately for LKP clues",
                   "Strong trail-following tendency — check all trail corridors"],
    },
    # East Asian (China, Japan, South Korea, Taiwan)
    "east_asian": {
        "countries": {"china", "japan", "south korea", "korea", "taiwan", "hong kong", "singapore"},
        "distance_mult": 0.92,
        "spiral_delta": -0.03,
        "trail_boost": 0.15,
        "language_barrier": True,
        "sar_note": (
            "Strong trail and rule-following behaviour. May be reluctant to call emergency "
            "services due to cultural norms. Japanese hikers tend to be well-equipped; Chinese "
            "tourists may lack wilderness experience. Language barrier possible."
        ),
        "advice": ["Strong trail bias — prioritise established paths and marked routes",
                   "May be reluctant to call — use visual/audio signals at trail junctions",
                   "Language barrier likely — use translated broadcast if available"],
    },
    # Southeast Asian (Indonesia, Philippines, Thailand, Vietnam, Malaysia)
    "southeast_asian": {
        "countries": {
            "indonesia", "philippines", "thailand", "vietnam", "malaysia",
            "myanmar", "cambodia", "laos", "brunei",
        },
        "distance_mult": 0.90,
        "spiral_delta": 0.05,
        "trail_boost": 0.08,
        "language_barrier": True,
        "sar_note": (
            "Variable wilderness experience. Tend to shelter in place and wait for help. "
            "Strong community and group orientation — may stay near the last point of "
            "group contact. Language barrier probable."
        ),
        "advice": ["Shelter-seeking behaviour elevated — check all covered/shaded areas",
                   "May shelter near last group contact point",
                   "Language barrier expected — arrange multilingual support"],
    },
    # Nordic / Scandinavian (extremely high wilderness experience)
    "nordic": {
        "countries": {"sweden", "norway", "denmark", "finland", "iceland"},
        "distance_mult": 1.25,
        "spiral_delta": -0.12,
        "trail_boost": -0.05,
        "language_barrier": False,
        "sar_note": (
            "Very high wilderness literacy (friluftsliv culture). Will attempt confident self-rescue, "
            "travel off-trail, and seek high ground. Wider search radius. Well-prepared with gear."
        ),
        "advice": ["Expand search radius — culturally experienced off-trail travellers",
                   "Check ridgelines and high ground first — self-rescue likely attempted",
                   "Subject may have signalling equipment — listen for whistle/mirror"],
    },
    # Latin American
    "latin_american": {
        "countries": {
            "mexico", "brazil", "argentina", "colombia", "chile", "peru",
            "venezuela", "ecuador", "bolivia", "paraguay", "uruguay",
            "guatemala", "honduras", "el salvador", "nicaragua", "costa rica",
            "panama", "cuba", "dominican republic",
        },
        "distance_mult": 0.95,
        "spiral_delta": 0.04,
        "trail_boost": 0.06,
        "language_barrier": True,
        "sar_note": (
            "Variable SAR familiarity. Strong family and community orientation — may attempt to "
            "reach a populated area rather than stay put. Downhill bias toward settlements strong. "
            "Spanish/Portuguese language barrier probable outside home country."
        ),
        "advice": ["Search toward roads and settlements downhill — subject may head for towns",
                   "Language barrier may apply — arrange Spanish/Portuguese support"],
    },
    # Middle Eastern / North African
    "mena": {
        "countries": {
            "saudi arabia", "uae", "iran", "iraq", "turkey", "egypt",
            "jordan", "lebanon", "syria", "israel", "palestine",
            "morocco", "algeria", "tunisia", "libya",
        },
        "distance_mult": 0.85,
        "spiral_delta": 0.06,
        "trail_boost": 0.05,
        "language_barrier": True,
        "sar_note": (
            "Desert terrain familiarity but typically low temperate forest wilderness experience. "
            "May seek shade/shelter strongly. Language barrier probable."
        ),
        "advice": ["Shelter-seeking very strong — check all shade and cover immediately",
                   "Unfamiliar with temperate forest — may panic on dense terrain",
                   "Language barrier expected"],
    },
    # Sub-Saharan African
    "african": {
        "countries": {
            "nigeria", "kenya", "ethiopia", "south africa", "ghana",
            "tanzania", "uganda", "cameroon", "zimbabwe", "zambia",
            "mozambique", "senegal", "mali", "sudan",
        },
        "distance_mult": 1.0,
        "spiral_delta": 0.0,
        "trail_boost": 0.0,
        "language_barrier": True,
        "sar_note": "Varied wilderness experience by region. Language barrier may apply. Using base profile.",
        "advice": ["Language barrier possible — confirm primary language before broadcast"],
    },
    # Default fallback
    "unknown": {
        "countries": set(),
        "distance_mult": 1.0,
        "spiral_delta": 0.0,
        "trail_boost": 0.0,
        "language_barrier": False,
        "sar_note": "Nationality unknown — using base profile.",
        "advice": [],
    },
}


def _nationality_modifiers(nationality: Optional[str]) -> Dict:
    """
    Returns behavioural modifier dict for a given country/nationality string.
    Matches by country name (case-insensitive) to a cultural group.
    """
    if not nationality:
        return {**NATIONALITY_GROUPS["unknown"], "group": "unknown"}
    n = nationality.lower().strip()
    for group_key, group in NATIONALITY_GROUPS.items():
        if group_key == "unknown":
            continue
        if n in group["countries"]:
            return {**group, "group": group_key}
    # Fuzzy: partial match
    for group_key, group in NATIONALITY_GROUPS.items():
        if group_key == "unknown":
            continue
        if any(n in c or c in n for c in group["countries"]):
            return {**group, "group": group_key}
    return {**NATIONALITY_GROUPS["unknown"], "group": "unknown"}


# ──────────────────────────────────────────────────────────────────────────────
# EDUCATION LEVEL & TYPE MODIFIERS
# ──────────────────────────────────────────────────────────────────────────────
# Education affects:
#   - Decision quality under panic (spiral probability)
#   - Logical navigation (distance efficiency, less random walk)
#   - Signalling and emergency call behaviour
#   - Risk assessment (overconfidence vs under-confidence)
# Level categories map to spiral_delta and decision_quality [0..1].
# Type/field adds an overlay on top of level.

EDUCATION_LEVELS = {
    "none":        {"spiral_delta": +0.12, "distance_mult": 0.85, "level_note": "No formal education — instinct-driven decisions, high panic risk"},
    "primary":     {"spiral_delta": +0.08, "distance_mult": 0.90, "level_note": "Primary schooling — limited structured reasoning under stress"},
    "secondary":   {"spiral_delta": +0.03, "distance_mult": 0.95, "level_note": "Secondary/high school — basic reasoning, some first-aid awareness"},
    "vocational":  {"spiral_delta": -0.02, "distance_mult": 1.00, "level_note": "Vocational/diploma — practical problem-solving, calm under pressure"},
    "diploma":     {"spiral_delta": -0.02, "distance_mult": 1.00, "level_note": "Diploma — practical problem-solving, calm under pressure"},
    "bachelor":    {"spiral_delta": -0.04, "distance_mult": 1.02, "level_note": "Bachelor's degree — structured thinking, likely to attempt logical self-rescue"},
    "master":      {"spiral_delta": -0.07, "distance_mult": 1.05, "level_note": "Master's degree — analytical, methodical under stress, lower panic rate"},
    "phd":         {"spiral_delta": -0.08, "distance_mult": 1.05, "level_note": "PhD — high analytical capacity; may over-engineer solution (stubbornness risk)"},
    "postgrad":    {"spiral_delta": -0.07, "distance_mult": 1.04, "level_note": "Postgraduate — analytical, lower panic rate"},
    "unknown":     {"spiral_delta":  0.00, "distance_mult": 1.00, "level_note": "Education level unknown — using base profile"},
}

EDUCATION_FIELD_OVERLAYS = {
    # STEM — systematic navigation, landmark recognition
    "btech":        {"spiral_delta": -0.06, "trail_boost":  0.05, "field_note": "B.Tech: systematic logical approach; may use sun/stars for orientation; map-reading likely"},
    "be":           {"spiral_delta": -0.06, "trail_boost":  0.05, "field_note": "B.E: engineering mindset; systematic search for paths; landmark-seeking"},
    "mtech":        {"spiral_delta": -0.08, "trail_boost":  0.06, "field_note": "M.Tech: high analytical capacity; may build shelter or device signals"},
    "bsc":          {"spiral_delta": -0.04, "trail_boost":  0.03, "field_note": "B.Sc: analytical; familiar with natural observation; reasonable navigation"},
    "msc":          {"spiral_delta": -0.06, "trail_boost":  0.04, "field_note": "M.Sc: scientific reasoning; likely to assess terrain systematically"},
    "bca":          {"spiral_delta": -0.03, "trail_boost":  0.02, "field_note": "BCA: tech-savvy — likely to use phone apps/GPS if available"},
    "mca":          {"spiral_delta": -0.04, "trail_boost":  0.02, "field_note": "MCA: tech-literate; will attempt phone signal or GPS rescue"},
    # Medical — best survival decision-making
    "mbbs":         {"spiral_delta": -0.10, "trail_boost":  0.08, "field_note": "MBBS: clinical calm under pressure; best survival decisions; will self-assess injuries accurately"},
    "md":           {"spiral_delta": -0.10, "trail_boost":  0.08, "field_note": "MD: calm, methodical; strong self-assessment; will conserve energy and signal effectively"},
    "nursing":      {"spiral_delta": -0.07, "trail_boost":  0.06, "field_note": "Nursing: calm, practical; will conserve resources and seek shelter systematically"},
    "pharmacy":     {"spiral_delta": -0.05, "trail_boost":  0.03, "field_note": "Pharmacy: health-conscious; will carefully ration water/food"},
    # Business / Management — leadership, may overestimate ability
    "mba":          {"spiral_delta": -0.02, "trail_boost":  0.00, "field_note": "MBA: leadership mindset; may overestimate wilderness ability; logical but overconfident"},
    "bba":          {"spiral_delta": -0.01, "trail_boost":  0.00, "field_note": "BBA: structured thinking, moderate panic resistance"},
    "bcom":         {"spiral_delta":  0.00, "trail_boost":  0.00, "field_note": "B.Com: average decision quality; no specific SAR advantage"},
    # Arts / Humanities — emotionally driven, higher panic susceptibility
    "ba":           {"spiral_delta": +0.03, "trail_boost": -0.02, "field_note": "BA: emotionally driven decisions under stress; higher panic susceptibility; may fixate on one direction"},
    "bfa":          {"spiral_delta": +0.04, "trail_boost": -0.02, "field_note": "BFA: creative but emotionally driven; may not follow logical navigation patterns"},
    "ma":           {"spiral_delta": +0.01, "trail_boost":  0.01, "field_note": "MA: reflective; may over-analyse decisions; moderate panic rate"},
    # Law — methodical but slow decision-making
    "llb":          {"spiral_delta": -0.02, "trail_boost":  0.03, "field_note": "LLB: methodical; may deliberate too long on decisions but ultimately logical"},
    "llm":          {"spiral_delta": -0.03, "trail_boost":  0.03, "field_note": "LLM: analytical; methodical risk assessment"},
    # Geography / Forestry / Environmental — best terrain navigation
    "geography":    {"spiral_delta": -0.12, "trail_boost":  0.12, "field_note": "Geography: strong spatial reasoning; excellent map/terrain reading; likely self-rescue"},
    "forestry":     {"spiral_delta": -0.14, "trail_boost":  0.10, "field_note": "Forestry: expert terrain navigation; woodland knowledge; most likely to self-rescue"},
    "environmental": {"spiral_delta": -0.10, "trail_boost":  0.08, "field_note": "Environmental science: good terrain awareness; water-source navigation instincts"},
    # Military / Defence — trained survival
    "military":     {"spiral_delta": -0.15, "trail_boost":  0.05, "field_note": "Military training: survival-trained; will signal, shelter, and self-rescue methodically"},
    "ncc":          {"spiral_delta": -0.08, "trail_boost":  0.04, "field_note": "NCC/Cadet: basic survival awareness; calm under moderate stress"},
}


def _education_modifiers(edu_level: Optional[str], edu_field: Optional[str]) -> Dict:
    """
    Resolve education level + field into combined modifier dict.
    edu_level: 'none' | 'primary' | 'secondary' | 'vocational' | 'diploma' |
                'bachelor' | 'master' | 'phd' | 'postgrad' | 'unknown'
    edu_field: 'btech' | 'bsc' | 'mbbs' | 'mba' | 'ba' | 'geography' etc.
    """
    # Normalise
    level_key = (edu_level or "unknown").lower().strip().replace(" ", "_")
    field_key = (edu_field or "").lower().strip().replace(" ", "").replace(".", "")

    level = EDUCATION_LEVELS.get(level_key, EDUCATION_LEVELS["unknown"])

    # Try to infer level from field if level is unknown
    if level_key == "unknown" and field_key:
        if field_key in ("mbbs", "md", "mtech", "msc", "mba", "llm", "mca", "ma"):
            level = EDUCATION_LEVELS["master"]
        elif field_key in ("phd",):
            level = EDUCATION_LEVELS["phd"]
        elif field_key in ("btech", "be", "bsc", "bca", "ba", "bba", "bcom", "bfa", "llb", "nursing", "pharmacy"):
            level = EDUCATION_LEVELS["bachelor"]

    field_overlay = EDUCATION_FIELD_OVERLAYS.get(field_key, {})

    combined_spiral = level["spiral_delta"] + field_overlay.get("spiral_delta", 0.0)
    combined_dist   = level["distance_mult"] * (1.0 + field_overlay.get("distance_mult_bonus", 0.0))
    combined_trail  = field_overlay.get("trail_boost", 0.0)

    notes = [level["level_note"]]
    if field_overlay.get("field_note"):
        notes.append(field_overlay["field_note"])

    return {
        "level": level_key,
        "field": field_key or None,
        "spiral_delta": round(combined_spiral, 3),
        "distance_mult": round(combined_dist, 3),
        "trail_boost": round(combined_trail, 3),
        "education_note": " | ".join(notes),
    }

# ──────────────────────────────────────────────────────────────────────────────
# JOB / OCCUPATION MODIFIERS
# ──────────────────────────────────────────────────────────────────────────────
# A person's daily occupation shapes their stress response, physical endurance,
# navigation instincts, and familiarity with emergency systems far more than
# most other demographic factors.
#
# Key behavioural axes each job affects:
#   distance_mult  — physical endurance + willingness to keep moving
#   spiral_delta   — panic probability change (negative = calmer)
#   trail_boost    — preference for established paths vs off-trail
#   shelter_boost  — tendency to stop and shelter vs keep moving
#   attractor_overrides — specific terrain attractors overridden
#   advice         — operator-facing tactical notes
#   job_note       — SAR operator context note
#
# Sources: Koester (2008), Mattson (2011), US Air Force SERE curriculum,
# wilderness medicine literature, NASAR field guides.

JOB_PROFILES = {

    # ── MILITARY / SECURITY / EMERGENCY SERVICES ─────────────────────────────
    "military": {
        "keywords": {"military", "soldier", "army", "navy", "air force", "marines", "special forces",
                     "combat", "infantry", "ranger", "commando", "defense", "defence"},
        "distance_mult": 1.45, "spiral_delta": -0.18, "trail_boost": -0.10, "shelter_boost": -0.05,
        "attractor_overrides": {"ridgeline": 0.85, "shelter": 0.75},
        "job_note": "Military personnel are SERE-trained (Survive, Evade, Resist, Escape). Will build shelter, signal methodically, and navigate by sun/stars. Expect wider travel radius and rational decisions even under stress.",
        "advice": [
            "Subject has survival training — check ridgelines and high ground for signalling attempts",
            "May have built an improvised shelter — look for deliberate construction",
            "Will signal in patterns of 3 (international distress). Listen for whistle/mirror",
            "Expand search radius significantly — physically capable of covering large distances",
        ],
    },
    "police": {
        "keywords": {"police", "officer", "detective", "cop", "constable", "sheriff", "law enforcement",
                     "security guard", "security officer"},
        "distance_mult": 1.15, "spiral_delta": -0.10, "trail_boost": 0.05, "shelter_boost": -0.05,
        "attractor_overrides": {},
        "job_note": "Law enforcement personnel are trained in situational awareness and emergency protocols. Calm decision-making. Likely to self-rescue via established paths and signal clearly.",
        "advice": [
            "Will use standard emergency signalling — respond to voice commands clearly",
            "Likely to follow trail systems toward populated areas",
            "May attempt to control the situation — search systematically along likely egress routes",
        ],
    },
    "firefighter": {
        "keywords": {"firefighter", "fire fighter", "fireman", "fire brigade", "paramedic", "emt",
                     "emergency medical", "rescue worker", "first responder"},
        "distance_mult": 1.25, "spiral_delta": -0.15, "trail_boost": 0.00, "shelter_boost": 0.10,
        "attractor_overrides": {},
        "job_note": "Emergency services personnel are trained in incident command and self-rescue. High physical endurance. Will quickly assess the situation, find a defensible position, and signal effectively.",
        "advice": [
            "High situational awareness — subject will self-assess and signal if able",
            "Check defensible positions and open clearings first — trained to avoid entrapment",
            "May attempt to reach high ground for radio/cell signal",
        ],
    },

    # ── MEDICAL ──────────────────────────────────────────────────────────────
    "doctor": {
        "keywords": {"doctor", "physician", "surgeon", "specialist", "consultant", "gp", "mbbs"},
        "distance_mult": 1.05, "spiral_delta": -0.12, "trail_boost": 0.08, "shelter_boost": 0.05,
        "attractor_overrides": {},
        "job_note": "Physicians remain exceptionally calm under physical stress due to clinical training. Will accurately self-assess injury severity, conserve energy strategically, and seek the most logical egress. Low panic probability.",
        "advice": [
            "Clinical calm — subject will make rational decisions and self-triage",
            "Will conserve energy and water methodically",
            "Likely to find highest-probability egress route (trail/road) and follow it",
        ],
    },
    "nurse_medical": {
        "keywords": {"nurse", "nursing", "paramedic", "healthcare", "health worker", "midwife",
                     "physiotherapist", "occupational therapist", "pharmacist"},
        "distance_mult": 1.00, "spiral_delta": -0.09, "trail_boost": 0.06, "shelter_boost": 0.08,
        "attractor_overrides": {},
        "job_note": "Medical professionals maintain composure under stress. Will shelter and conserve energy efficiently. Lower panic rate than general population.",
        "advice": [
            "Will shelter systematically and await rescue if self-rescue seems impractical",
            "Good self-assessment — injuries will not be under or over-reported",
        ],
    },

    # ── OUTDOOR / FIELD PROFESSIONS ───────────────────────────────────────────
    "forester": {
        "keywords": {"forester", "forestry", "forest ranger", "park ranger", "ranger", "wildlife",
                     "conservation", "naturalist", "ecologist", "game warden"},
        "distance_mult": 1.40, "spiral_delta": -0.20, "trail_boost": -0.15, "shelter_boost": 0.05,
        "attractor_overrides": {"water": 0.80, "shelter": 0.70},
        "job_note": "Forestry and park professionals have expert-level terrain literacy. Will navigate off-trail confidently, locate water, and build shelter. Most likely of all occupations to self-rescue successfully.",
        "advice": [
            "Expert terrain navigation — will travel off-trail confidently",
            "Will locate water sources and create shelter efficiently",
            "Expand search radius significantly — this person knows the wilderness",
            "Check terrain features (ridgelines, drainages) not just trails",
        ],
    },
    "farmer": {
        "keywords": {"farmer", "agriculture", "agricultural", "rancher", "shepherd", "herder",
                     "farmworker", "horticulture", "gardener", "livestock"},
        "distance_mult": 1.10, "spiral_delta": -0.05, "trail_boost": 0.00, "shelter_boost": 0.12,
        "attractor_overrides": {"open_field": 0.80, "water": 0.65},
        "job_note": "Farmers have strong practical outdoor skills and terrain awareness. Calm under pressure from physical labour background. Will seek open fields and water sources. Good improvised shelter skills.",
        "advice": [
            "Strong terrain instincts — will likely head for open land or water",
            "Physical endurance is high — check further than average",
            "Will shelter using available materials efficiently",
        ],
    },
    "geologist_surveyor": {
        "keywords": {"geologist", "geology", "surveyor", "surveying", "cartographer", "geographer",
                     "topographer", "mining engineer", "civil engineer", "field researcher"},
        "distance_mult": 1.20, "spiral_delta": -0.14, "trail_boost": -0.08, "shelter_boost": 0.05,
        "attractor_overrides": {"ridgeline": 0.80, "rocky_outcrop": 0.75},
        "job_note": "Geologists and surveyors have professional terrain-reading skills. Will navigate by landmarks, seek high ground for orientation, and understand drainage patterns. Comfortable off-trail.",
        "advice": [
            "Will navigate by terrain landmarks — check ridgelines and distinctive features",
            "Comfortable off-trail — search beyond established path corridors",
            "Strong drainage pattern awareness — may follow creek systems",
        ],
    },
    "construction": {
        "keywords": {"construction", "builder", "contractor", "carpenter", "mason", "bricklayer",
                     "plumber", "electrician", "welder", "mechanic", "tradesman", "labourer", "laborer"},
        "distance_mult": 1.15, "spiral_delta": -0.04, "trail_boost": 0.05, "shelter_boost": 0.15,
        "attractor_overrides": {},
        "job_note": "Construction and trade workers have high physical endurance and practical shelter-building skills. Will attempt self-rescue confidently. Moderate stress response.",
        "advice": [
            "High physical endurance — check further from LKP than average",
            "Practical problem-solver — may build visible signals (rock piles, cleared areas)",
        ],
    },
    "pilot_navigator": {
        "keywords": {"pilot", "aviator", "co-pilot", "navigator", "navigator", "air traffic control",
                     "flight crew", "captain"},
        "distance_mult": 1.10, "spiral_delta": -0.16, "trail_boost": 0.05, "shelter_boost": 0.00,
        "attractor_overrides": {"ridgeline": 0.80, "open_field": 0.70},
        "job_note": "Pilots have exceptional spatial orientation and situational awareness. Will immediately seek high ground for bearings, identify cardinal directions by sun/stars, and signal with systematic patterns. Very low panic rate.",
        "advice": [
            "Excellent spatial orientation — will seek high ground for bearings immediately",
            "Will signal systematically — check open clearings and high ground",
            "Very low panic probability — expect rational, methodical movement",
        ],
    },

    # ── PHYSICAL / SPORT ─────────────────────────────────────────────────────
    "athlete": {
        "keywords": {"athlete", "sportsperson", "footballer", "rugby", "runner", "marathon",
                     "triathlete", "cyclist", "swimmer", "coach", "sports coach", "physical trainer",
                     "gym trainer", "personal trainer", "fitness instructor"},
        "distance_mult": 1.35, "spiral_delta": -0.08, "trail_boost": 0.00, "shelter_boost": -0.10,
        "attractor_overrides": {},
        "job_note": "Athletes have superior physical endurance and push through discomfort. Will travel significantly further than average before stopping. May underestimate wilderness hazards due to confidence in physical ability.",
        "advice": [
            "High physical endurance — expand search radius significantly",
            "May have pushed further than expected before stopping",
            "Confident in physical ability — may have taken a challenging route",
        ],
    },

    # ── TECHNICAL / OFFICE ───────────────────────────────────────────────────
    "engineer_tech": {
        "keywords": {"engineer", "software engineer", "developer", "programmer", "it", "tech",
                     "data scientist", "analyst", "architect", "systems", "devops"},
        "distance_mult": 1.05, "spiral_delta": -0.08, "trail_boost": 0.08, "shelter_boost": 0.00,
        "attractor_overrides": {},
        "job_note": "Technical professionals apply systematic problem-solving under stress. Will attempt logical navigation (sun direction, slope reading, trail following). May use phone apps if signal available. Lower panic rate.",
        "advice": [
            "Systematic thinker — likely following a logical rule (downhill, sun direction)",
            "Will use phone/GPS if any signal — check mobile ping data if available",
            "Trail-following bias elevated — check established path corridors",
        ],
    },
    "office_worker": {
        "keywords": {"office", "accountant", "administrator", "clerk", "receptionist", "hr",
                     "manager", "executive", "banker", "finance", "marketing", "sales"},
        "distance_mult": 0.88, "spiral_delta": 0.06, "trail_boost": 0.08, "shelter_boost": 0.10,
        "attractor_overrides": {},
        "job_note": "Office workers typically have limited wilderness experience. Higher stress response, lower physical endurance. Will seek trails and shelter quickly. More likely to stay near last known position.",
        "advice": [
            "Limited wilderness experience — search close to LKP first",
            "Higher panic probability — may be sheltered near LKP",
            "Strong trail-following tendency — prioritise established paths",
        ],
    },
    "teacher_academic": {
        "keywords": {"teacher", "lecturer", "professor", "academic", "researcher", "scientist",
                     "educator", "tutor", "librarian"},
        "distance_mult": 0.95, "spiral_delta": -0.03, "trail_boost": 0.05, "shelter_boost": 0.08,
        "attractor_overrides": {},
        "job_note": "Teachers and academics are methodical and analytical. Moderate stress response. Will think through the problem systematically but may lack physical endurance for sustained travel.",
        "advice": [
            "Methodical thinker — likely following a logical egress pattern",
            "Moderate endurance — check within average radius",
        ],
    },
    "journalist_media": {
        "keywords": {"journalist", "reporter", "photographer", "videographer", "filmmaker",
                     "media", "writer", "blogger", "content creator"},
        "distance_mult": 0.95, "spiral_delta": 0.04, "trail_boost": 0.05, "shelter_boost": 0.08,
        "attractor_overrides": {},
        "job_note": "Media professionals often work in varied field environments but lack specific wilderness survival training. Will seek familiar terrain features and may attempt to reach phone signal.",
        "advice": [
            "Will prioritise reaching phone signal — check elevated or open areas",
            "May document their movement (photos, notes) — check device data if recovered",
        ],
    },

    # ── TRANSPORT ────────────────────────────────────────────────────────────
    "driver_transport": {
        "keywords": {"driver", "truck driver", "lorry", "delivery", "taxi", "bus driver",
                     "transport", "logistics", "courier"},
        "distance_mult": 1.00, "spiral_delta": 0.02, "trail_boost": 0.12, "shelter_boost": 0.05,
        "attractor_overrides": {"road_following": 0.90, "open_field": 0.70},
        "job_note": "Transport workers have strong road-following instincts and spatial awareness for road networks. Will aggressively seek roads, paths, and any linear feature that leads to civilisation.",
        "advice": [
            "Very strong road-following instinct — prioritise roads, tracks, and paths",
            "Will follow any linear feature toward populated areas",
            "Check all road/path junctions — likely heading toward a road",
        ],
    },

    # ── CREATIVE / ARTS ──────────────────────────────────────────────────────
    "creative_arts": {
        "keywords": {"artist", "musician", "actor", "dancer", "designer", "graphic designer",
                     "photographer", "sculptor", "painter", "creative"},
        "distance_mult": 0.90, "spiral_delta": 0.08, "trail_boost": -0.05, "shelter_boost": 0.12,
        "attractor_overrides": {},
        "job_note": "Creative professionals tend toward emotionally driven decisions under stress. Higher panic susceptibility. May move erratically or fixate on a single direction. Will shelter when energy drops.",
        "advice": [
            "Emotionally driven movement likely — expect less logical navigation",
            "Higher panic probability — may be sheltered and stationary",
            "Search shelter points thoroughly — subject may stop moving early",
        ],
    },

    # ── STUDENT ──────────────────────────────────────────────────────────────
    "student": {
        "keywords": {"student", "undergraduate", "postgraduate", "college", "university", "school",
                     "intern", "trainee", "apprentice"},
        "distance_mult": 1.02, "spiral_delta": 0.03, "trail_boost": 0.05, "shelter_boost": 0.05,
        "attractor_overrides": {},
        "job_note": "Students have moderate stress response. Physical endurance varies widely. Likely to attempt self-rescue via phone or by following trails. Moderate panic rate.",
        "advice": [
            "Will attempt phone contact first — check mobile provider data",
            "Trail-following bias — check established paths",
        ],
    },

    # ── RETIRED ──────────────────────────────────────────────────────────────
    "retired": {
        "keywords": {"retired", "pensioner", "retiree", "veteran"},
        "distance_mult": 0.72, "spiral_delta": 0.05, "trail_boost": 0.10, "shelter_boost": 0.18,
        "attractor_overrides": {"trail_following": 0.80},
        "job_note": "Retired individuals typically have reduced physical endurance (age-compounding effect applied separately). Strong trail-following tendency. Will shelter early. Prior career may significantly modify this — check subject history.",
        "advice": [
            "Reduced endurance — search close to LKP and on trail systems",
            "Will shelter early — check all natural and man-made shelter points",
            "Note: prior career may modify behaviour significantly (e.g., retired military vs retired office worker)",
        ],
    },

    # ── UNKNOWN FALLBACK ─────────────────────────────────────────────────────
    "unknown": {
        "keywords": set(),
        "distance_mult": 1.0, "spiral_delta": 0.0, "trail_boost": 0.0, "shelter_boost": 0.0,
        "attractor_overrides": {},
        "job_note": "Occupation unknown — using base profile.",
        "advice": [],
    },
}


def _job_modifiers(job: Optional[str]) -> Dict:
    """
    Match free-text occupation string to a JOB_PROFILES group.
    Returns modifier dict with distance_mult, spiral_delta, trail_boost,
    shelter_boost, attractor_overrides, advice, job_note.
    """
    if not job:
        return {**JOB_PROFILES["unknown"], "matched_group": "unknown"}

    j = job.lower().strip()

    # Exact keyword match first
    for group_key, profile in JOB_PROFILES.items():
        if group_key == "unknown":
            continue
        if j in profile["keywords"]:
            return {**profile, "matched_group": group_key}

    # Partial / substring match
    for group_key, profile in JOB_PROFILES.items():
        if group_key == "unknown":
            continue
        if any(kw in j or j in kw for kw in profile["keywords"]):
            return {**profile, "matched_group": group_key}

    return {**JOB_PROFILES["unknown"], "matched_group": "unknown"}


# ──────────────────────────────────────────────────────────────────────────────
# TIME MODIFIERS: TIME OF DAY + ELAPSED DURATION + TRACK FRESHNESS
# ──────────────────────────────────────────────────────────────────────────────
#
# Three temporal axes that profoundly change SAR outcomes:
#
# 1. TIME OF DAY at search start (hour_of_day)
#    Controls: movement probability, shelter seeking, hypothermia risk, visibility
#    Night severely reduces movement distance and raises urgency.
#
# 2. TIME ELAPSED since subject entered area / was last seen (hours_missing)
#    Controls: fatigue-related distance reduction, desperation/panic escalation,
#    medical risk threshold (dehydration, hypothermia, injury deterioration).
#
# 3. TRACK FRESHNESS WINDOW
#    Detections older than the elapsed window are "pre-loss" and should be
#    down-weighted in the heatmap. Used by score_grid() to adjust det_score.
#
# Sources: Syrotuck (1976), Koester (2008), Wilderness Medicine data.

# Time-of-day windows and their effect profiles
TIME_OF_DAY_PROFILES = {
    "dawn": {            # 05:00 – 07:59
        "hours": (5, 8),
        "distance_mult": 1.00,
        "spiral_delta":  0.00,
        "shelter_mult":  0.80,    # subject may be moving again after sheltering
        "urgency_floor": None,
        "time_note": "Dawn window: subject may be resuming movement. Tracks from overnight are stale — focus on fresh morning detections.",
        "advice": ["Dawn search window: subject likely moving again — track fresh disturbances",
                   "Night tracks are stale — prioritise detections from last 2h"],
    },
    "morning": {         # 08:00 – 11:59
        "hours": (8, 12),
        "distance_mult": 1.05,
        "spiral_delta":  -0.03,
        "shelter_mult":  0.85,
        "urgency_floor": None,
        "time_note": "Morning: peak movement window. Subject most likely still mobile. Detection confidence highest.",
        "advice": ["Peak movement window — subject likely still mobile",
                   "Prioritise aerial sweep before midday heat reduces track visibility"],
    },
    "afternoon": {       # 12:00 – 16:59
        "hours": (12, 17),
        "distance_mult": 1.00,
        "spiral_delta":  0.02,
        "shelter_mult":  0.90,
        "urgency_floor": None,
        "time_note": "Afternoon: heat and fatigue may slow subject. Movement continues but at reduced pace.",
        "advice": ["Heat fatigue may be setting in — check shaded/sheltered rest points",
                   "Water source seeking elevated in afternoon heat"],
    },
    "dusk": {            # 17:00 – 19:59
        "hours": (17, 20),
        "distance_mult": 0.65,
        "spiral_delta":  0.08,
        "shelter_mult":  1.30,
        "urgency_floor": "HIGH",
        "time_note": "Dusk: subject likely seeking shelter for night. Movement slowing rapidly. Hypothermia risk begins. HIGH urgency if not found before dark.",
        "advice": ["DUSK WINDOW: subject is likely seeking shelter RIGHT NOW — search urgently",
                   "Focus on natural shelter: caves, dense brush, fallen logs",
                   "Hypothermia risk begins tonight — activate medical response team"],
    },
    "night": {           # 20:00 – 04:59
        "hours": (20, 5),  # wraps midnight
        "distance_mult": 0.35,
        "spiral_delta":  0.15,
        "shelter_mult":  1.60,
        "urgency_floor": "HIGH",
        "time_note": "Night: subject almost certainly sheltered and stationary. Movement is highly dangerous in the dark for a lost person. Hypothermia and disorientation risk are CRITICAL. Noise and light signals effective.",
        "advice": ["NIGHT SEARCH: subject is almost certainly sheltered and stationary",
                   "Use audio signals (air horn, megaphone) and lights — far more effective at night",
                   "Hypothermia risk CRITICAL — medical team must be ready on contact",
                   "Search shelter spots first: tree hollows, dense brush, creek banks"],
    },
}


def _hour_to_window(hour: int) -> str:
    """Map a 0-23 hour integer to a time-of-day window key."""
    if 5 <= hour < 8:   return "dawn"
    if 8 <= hour < 12:  return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 20: return "dusk"
    return "night"  # 20-24 and 0-4


def _elapsed_time_profile(hours: float) -> Dict:
    """
    Model how elapsed time since disappearance affects movement capacity,
    panic escalation, and urgency.

    Timeline (sourced from Wilderness Medicine & NASAR stats):
      0–2h   : Subject still orienting, likely moving
      2–6h   : Fatigue building, distance capacity reducing
      6–12h  : Significant fatigue, hypothermia risk if night passed
      12–24h : HIGH urgency; dehydration begins; panic level peaks then flattens
      24–48h : CRITICAL urgency; medical deterioration likely
      >48h   : Extreme urgency; survival probability dropping per hour
    """
    if hours <= 2:
        return {
            "distance_mult": 1.00, "spiral_delta": 0.00,
            "urgency_floor": None, "medical_risk": "LOW",
            "elapsed_note": f"Missing {hours:.1f}h — subject still orienting, likely mobile. Tracks are fresh.",
            "track_max_age_h": 3.0,   # only care about tracks <3h old
            "advice": ["Tracks are fresh — detection confidence scores are reliable",
                       "Subject likely still moving — prioritise forward search from LKP"],
        }
    elif hours <= 6:
        return {
            "distance_mult": 0.90, "spiral_delta": 0.03,
            "urgency_floor": None, "medical_risk": "LOW-MODERATE",
            "elapsed_note": f"Missing {hours:.1f}h — fatigue building. Some tracks may be up to 6h old.",
            "track_max_age_h": 7.0,
            "advice": [f"Missing {hours:.1f}h: fatigue reducing distance capacity",
                       "Check rest points and water sources — subject may have stopped"],
        }
    elif hours <= 12:
        return {
            "distance_mult": 0.78, "spiral_delta": 0.06,
            "urgency_floor": "HIGH", "medical_risk": "MODERATE",
            "elapsed_note": f"Missing {hours:.1f}h — significant fatigue. If night was involved, hypothermia risk. Distance capacity reduced ~22%.",
            "track_max_age_h": 13.0,
            "advice": [f"Missing {hours:.1f}h: HIGH urgency — fatigue, possible hypothermia if exposed overnight",
                       "Search shelter areas thoroughly — subject may have stopped hours ago",
                       "Dehydration beginning — water sources become critical attractors"],
        }
    elif hours <= 24:
        return {
            "distance_mult": 0.60, "spiral_delta": 0.10,
            "urgency_floor": "HIGH", "medical_risk": "HIGH",
            "elapsed_note": f"Missing {hours:.1f}h — HIGH urgency. Dehydration active, panic peaked, movement severely limited by fatigue.",
            "track_max_age_h": 25.0,
            "advice": [f"Missing {hours:.1f}h: HIGH medical risk — dehydration, hypothermia, exhaustion",
                       "Subject likely stationary — search shelter areas and water sources systematically",
                       "Activate medical team for immediate response on contact"],
        }
    elif hours <= 48:
        return {
            "distance_mult": 0.40, "spiral_delta": 0.08,
            "urgency_floor": "CRITICAL", "medical_risk": "CRITICAL",
            "elapsed_note": f"Missing {hours:.1f}h — CRITICAL. Severe dehydration, hypothermia, possible incapacitation. Subject almost certainly stationary.",
            "track_max_age_h": hours + 1,
            "advice": [f"🚨 CRITICAL: Missing {hours:.1f}h — severe dehydration, hypothermia, possible incapacitation",
                       "Subject is almost certainly stationary — focus search on high-probability static locations",
                       "Advanced medical team required on scene immediately upon location",
                       "Consider K9 units and thermal aerial imaging for stationary subject detection"],
        }
    else:
        return {
            "distance_mult": 0.25, "spiral_delta": 0.05,
            "urgency_floor": "CRITICAL", "medical_risk": "EXTREME",
            "elapsed_note": f"Missing {hours:.1f}h — EXTREME urgency. Survival probability declining per hour. All available resources required.",
            "track_max_age_h": hours + 1,
            "advice": [f"🚨 EXTREME URGENCY: Missing {hours:.1f}h — survival probability declining per hour",
                       "Deploy ALL available search assets immediately",
                       "Expand search to full statistical radius — subject may have moved early then collapsed",
                       "Trauma-level medical response required on contact"],
        }


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    """Safely parse an ISO 8601 datetime string to a UTC-aware datetime."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _compute_time_modifiers(
    entry_time: Optional[str],
    search_start_time: Optional[str],
    reported_missing_hours: float = 0.0,
) -> Dict:
    """
    Compute time-based behavioural modifiers.

    Args:
        entry_time          : ISO datetime when subject entered / was last seen.
        search_start_time   : ISO datetime when search is launched (defaults to now).
        reported_missing_hours: Fallback if datetimes are not provided.

    Returns a modifier dict containing:
        hours_missing, time_of_day_window, distance_mult, spiral_delta,
        urgency_floor, medical_risk, track_max_age_h,
        entry_time_str, search_start_time_str, time_elapsed_str, advice
    """
    now_utc = datetime.now(timezone.utc)

    entry_dt  = _parse_iso(entry_time)
    search_dt = _parse_iso(search_start_time) or now_utc

    # Compute elapsed hours
    if entry_dt and search_dt:
        delta = search_dt - entry_dt
        hours_missing = max(0.0, delta.total_seconds() / 3600)
    else:
        hours_missing = max(0.0, reported_missing_hours)

    # Format elapsed string (e.g. "14h 32m")
    total_min = int(hours_missing * 60)
    elapsed_str = f"{total_min // 60}h {total_min % 60}m"

    # Time-of-day window at search start
    search_hour = search_dt.hour
    tod_window  = _hour_to_window(search_hour)
    tod_profile = TIME_OF_DAY_PROFILES[tod_window]

    # Entry time window (may differ from search start)
    entry_hour_note = ""
    if entry_dt:
        entry_window = _hour_to_window(entry_dt.hour)
        if entry_window != tod_window:
            entry_hour_note = (
                f"Subject entered during {entry_window} ({entry_dt.strftime('%H:%M')} UTC). "
                f"Search launching at {tod_window} ({search_dt.strftime('%H:%M')} UTC)."
            )

    # Elapsed-time profile
    elapsed_profile = _elapsed_time_profile(hours_missing)

    # Combined multipliers (time-of-day × elapsed fatigue)
    combined_dist   = tod_profile["distance_mult"] * elapsed_profile["distance_mult"]
    combined_spiral = tod_profile["spiral_delta"]  + elapsed_profile["spiral_delta"]

    # Urgency floor: take the stricter of the two
    urgency_order = {None: 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    tod_urg   = tod_profile["urgency_floor"]
    ela_urg   = elapsed_profile["urgency_floor"]
    urgency_floor = max([tod_urg, ela_urg], key=lambda u: urgency_order.get(u, 0))

    # Compose advice
    advice = list(elapsed_profile["advice"]) + list(tod_profile["advice"])
    if entry_hour_note:
        advice.insert(0, f"🕒 {entry_hour_note}")

    return {
        "hours_missing":          round(hours_missing, 2),
        "elapsed_str":            elapsed_str,
        "entry_time_str":         entry_dt.isoformat() if entry_dt else None,
        "search_start_time_str":  search_dt.isoformat(),
        "time_of_day_window":     tod_window,
        "search_hour":            search_hour,
        "entry_hour_note":        entry_hour_note,
        "distance_mult":          round(combined_dist, 3),
        "spiral_delta":           round(combined_spiral, 3),
        "urgency_floor":          urgency_floor,
        "medical_risk":           elapsed_profile["medical_risk"],
        "track_max_age_h":        elapsed_profile["track_max_age_h"],
        "time_note":              tod_profile["time_note"],
        "elapsed_note":           elapsed_profile["elapsed_note"],
        "advice":                 advice,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CIRCUMSTANCE MODIFIERS
# ──────────────────────────────────────────────────────────────────────────────
CIRCUMSTANCE_PROFILES = {
    "injured": {
        "distance_mult": 0.30,
        "spiral_delta": 0.20,
        "urgency_override": "CRITICAL",
        "attractor_overrides": {"shelter": 0.95, "trail_following": 0.75},
        "advice": [
            "CRITICAL: Subject is injured — search radius dramatically reduced",
            "Focus immediately within 1 km of last known position",
            "Check shelter points and areas with ground cover",
            "Listen for calls or signals — subject may not be moving",
        ],
        "behavior_note": "Injured persons stop moving much sooner and seek immediate shelter. Most found within 500 m of where injury occurred.",
    },
    "intoxicated": {
        "distance_mult": 0.60,
        "spiral_delta": 0.35,
        "urgency_override": "HIGH",
        "attractor_overrides": {"downhill": 0.90, "water": 0.70, "shelter": 0.30},
        "advice": [
            "Erratic movement — downhill bias very strong when intoxicated",
            "Check creek beds, drainage ditches, and depressions",
            "Subject may be asleep or unresponsive — check all vegetation",
            "High hypothermia risk if temperature is low",
        ],
        "behavior_note": "Intoxicated subjects exhibit strong downhill bias and erratic direction changes. Often found in low-lying areas or water features.",
    },
    "suicidal": {
        "distance_mult": 1.40,
        "spiral_delta": -0.10,
        "urgency_override": "CRITICAL",
        "attractor_overrides": {"water": 0.90, "dense_forest": 0.80, "ridgeline": 0.60},
        "advice": [
            "CRITICAL: Subject may be deliberately concealing themselves",
            "Prioritize water bodies — elevated drowning risk",
            "Search dense forest areas and isolated locations",
            "High points and cliff edges — check immediately",
            "Do NOT announce SAR on radio channels subject may hear",
        ],
        "behavior_note": "Subjects with suicidal ideation tend to travel further, seek isolation, and avoid trails. Water bodies are critical search areas.",
    },
    "medical_emergency": {
        "distance_mult": 0.20,
        "spiral_delta": 0.0,
        "urgency_override": "CRITICAL",
        "attractor_overrides": {"shelter": 0.60, "trail_following": 0.50},
        "advice": [
            "CRITICAL: Medical emergency — subject likely collapsed near last known point",
            "Search within 500 m of last known position first",
            "Contact EMS immediately — medical team required on scene",
            "Check all locations where subject may have sat down",
        ],
        "behavior_note": "Medical emergencies (cardiac, diabetic, seizure) cause subjects to stop abruptly. Typically found very close to last known position.",
    },
    "experienced": {
        "distance_mult": 1.30,
        "spiral_delta": -0.15,
        "urgency_override": None,
        "attractor_overrides": {"ridgeline": 0.75, "trail_following": 0.80, "shelter": 0.65},
        "advice": [
            "Experienced outdoorsperson — likely attempting self-rescue",
            "Check ridgelines and high ground first for signaling attempts",
            "Look for improvised shelters and deliberate trail markers",
            "May travel further than average — expand search radius",
        ],
        "behavior_note": "Experienced subjects make rational navigation decisions, seek high ground to orient, and may leave intentional trail markers for rescuers.",
    },
    "night_lost": {
        "distance_mult": 0.55,
        "spiral_delta": 0.18,
        "urgency_override": "HIGH",
        "attractor_overrides": {"shelter": 0.85, "trail_following": 0.40},
        "advice": [
            "Lost at night: subject likely stopped moving after dark",
            "Search areas with natural shelter — will have sheltered in place",
            "Use lights and noise signals — night searching effective",
            "Hypothermia risk elevated — prioritize medical response",
        ],
        "behavior_note": "Subjects who realize they are lost at night typically stop moving and seek shelter. Spiral radius greatly reduced.",
    },
    "group_separated": {
        "distance_mult": 0.70,
        "spiral_delta": -0.05,
        "urgency_override": None,
        "attractor_overrides": {"trail_following": 0.85},
        "advice": [
            "Separated from group — likely backtracking to last group contact point",
            "Interview group members for exact separation location",
            "Check the original planned route — subject may be following it",
            "Call out group member names — subject is motivated to respond",
        ],
        "behavior_note": "Subjects separated from a group often retrace steps or follow the planned route. Shorter travel distance, strong trail-following tendency.",
    },
    "normal": {
        "distance_mult": 1.0,
        "spiral_delta": 0.0,
        "urgency_override": None,
        "attractor_overrides": {},
        "advice": [],
        "behavior_note": "",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# TRACE-BASED PATH PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def _predict_travel_direction(detections: List[Dict]) -> Optional[Dict]:
    """
    Infer direction of travel from chronological detection locations.
    Uses linear regression on lat/lng over time-ordered detections to
    estimate bearing and projected next position.
    
    Returns: {bearing_deg, proj_lat, proj_lng, confidence, method}
    or None if insufficient data.
    """
    # Extract footprint/disturbed_ground detections (most directional)
    DIRECTIONAL_TYPES = {"footprint", "broken_branch", "disturbed_ground", "trail_blazing"}
    directional = [
        d for d in detections
        if d.get("track_type") in DIRECTIONAL_TYPES and d.get("confidence", 0) >= 0.55
    ]

    if len(directional) < 2:
        return None

    # Sort by estimated age (fresher = more recent = later position)
    # estimated_age_hours: smaller = more recent
    sorted_dets = sorted(directional, key=lambda d: d.get("estimated_age_hours", 999), reverse=True)

    lats = [d["location"]["lat"] for d in sorted_dets]
    lngs = [d["location"]["lng"] for d in sorted_dets]

    if len(lats) < 2:
        return None

    # Simple bearing: oldest → newest detection
    oldest = sorted_dets[-1]["location"]
    newest = sorted_dets[0]["location"]

    dlat = newest["lat"] - oldest["lat"]
    dlng = newest["lng"] - oldest["lng"]
    dist = math.sqrt(dlat**2 + dlng**2)

    if dist < 1e-6:
        return None

    # Bearing in degrees (0=North, 90=East)
    bearing_rad = math.atan2(dlng, dlat)
    bearing_deg = math.degrees(bearing_rad) % 360

    # Project forward: extrapolate same vector by ~50%
    step = 0.003  # ~330 m projection
    proj_lat = newest["lat"] + step * math.cos(bearing_rad)
    proj_lng = newest["lng"] + step * math.sin(bearing_rad)

    # Confidence based on number of consistent detections
    confidence = min(0.95, 0.40 + len(directional) * 0.10)

    # Detect direction change (spiral/panic indicator)
    direction_change = False
    if len(sorted_dets) >= 3:
        # Compare bearing of first half vs second half
        mid = len(sorted_dets) // 2
        b1 = _bearing(sorted_dets[-1]["location"], sorted_dets[mid]["location"])
        b2 = _bearing(sorted_dets[mid]["location"], sorted_dets[0]["location"])
        angle_diff = abs(b2 - b1)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        direction_change = angle_diff > 60  # >60° turn = potential panic/loop

    cardinal = _bearing_to_cardinal(bearing_deg)

    return {
        "bearing_deg": round(bearing_deg, 1),
        "cardinal_direction": cardinal,
        "projected_lat": round(proj_lat, 6),
        "projected_lng": round(proj_lng, 6),
        "confidence": round(confidence, 2),
        "direction_change_detected": direction_change,
        "detection_count_used": len(directional),
        "method": "linear regression on chronological track detections",
        "interpretation": (
            f"Subject traveling {cardinal} ({bearing_deg:.0f}°). "
            + ("⚠️ Direction change detected — possible panic/circular movement." if direction_change
               else f"Projected next position: {proj_lat:.4f}, {proj_lng:.4f}.")
        ),
    }

def _bearing(a: Dict, b: Dict) -> float:
    dlat = b["lat"] - a["lat"]
    dlng = b["lng"] - a["lng"]
    return math.degrees(math.atan2(dlng, dlat)) % 360

def _bearing_to_cardinal(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    return dirs[round(deg / 45) % 8]

# ──────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA MODIFIER
# ──────────────────────────────────────────────────────────────────────────────
def _compute_historical_modifiers(historical_cases: Optional[List[Dict]]) -> Dict:
    """Analyze similar historical SAR cases to provide data-driven insights."""
    if not historical_cases:
        return {
            "has_data": False,
            "distance_mult": 1.0,
            "historical_note": "No similar historical cases found in the database.",
            "similar_cases_count": 0,
            "avg_hours_to_find": None,
            "avg_distance_km": None,
            "found_alive_rate": None,
            "common_factors": [],
            "advice": []
        }
    
    count = len(historical_cases)
    alive_count = sum(1 for c in historical_cases if c.get("outcome") == "found_alive")
    alive_rate = alive_count / count

    avg_dist = sum(c.get("distance_km", 0) for c in historical_cases) / count
    avg_hours = sum(c.get("hours_to_find", 0) for c in historical_cases) / count

    factors_count = {}
    for c in historical_cases:
        for f in c.get("key_factors", []):
            factors_count[f] = factors_count.get(f, 0) + 1
            
    sorted_factors = sorted(factors_count.items(), key=lambda x: x[1], reverse=True)
    top_factors = [f[0] for f in sorted_factors[:4]]
    
    return {
        "has_data": True,
        "distance_mult": 1.0, # Keeping base distance mult neutral to avoid wild swings, mostly for insight
        "historical_note": f"Based on {count} similar past cases, {int(alive_rate*100)}% were found alive at an average distance of {avg_dist:.1f} km.",
        "similar_cases_count": count,
        "avg_hours_to_find": round(avg_hours, 1),
        "avg_distance_km": round(avg_dist, 1),
        "found_alive_rate": round(alive_rate, 2),
        "common_factors": top_factors,
        "advice": [f"Historical data suggests watching for: {', '.join(top_factors)}"] if top_factors else []
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN MODEL CLASS
# ──────────────────────────────────────────────────────────────────────────────
class PsychModel:
    def __init__(
        self,
        subject_type: str = "hiker",
        age: Optional[int] = None,
        gender: Optional[str] = None,
        circumstance: Optional[str] = "normal",
        nationality: Optional[str] = None,
        education_level: Optional[str] = None,
        education_field: Optional[str] = None,
        job: Optional[str] = None,
        entry_time: Optional[str] = None,
        search_start_time: Optional[str] = None,
        reported_missing_hours: float = 0.0,
        historical_cases: Optional[List[Dict]] = None,
    ):
        self.subject_type = subject_type
        self.age = age
        self.gender = gender
        self.circumstance = circumstance or "normal"
        self.nationality = nationality
        self.education_level = education_level
        self.education_field = education_field
        self.job = job
        self.entry_time = entry_time
        self.search_start_time = search_start_time

        # Base profile
        self.base_profile = SUBJECT_BEHAVIOR_PROFILES.get(
            subject_type, SUBJECT_BEHAVIOR_PROFILES["hiker"]
        )

        # Compute modifiers
        self.age_mod    = _age_modifiers(age, subject_type)
        self.gender_mod = _gender_modifiers(gender)
        self.nat_mod    = _nationality_modifiers(nationality)
        self.edu_mod    = _education_modifiers(education_level, education_field)
        self.job_mod    = _job_modifiers(job)
        self.circ_profile = CIRCUMSTANCE_PROFILES.get(self.circumstance, CIRCUMSTANCE_PROFILES["normal"])
        self.time_mod   = _compute_time_modifiers(entry_time, search_start_time, reported_missing_hours)
        self.hist_mod   = _compute_historical_modifiers(historical_cases)

        # Effective distance — ALL multipliers applied (including time fatigue & time-of-day)
        self.effective_distance_km = (
            self.base_profile["avg_distance_km"]
            * self.age_mod["distance_mult"]
            * self.gender_mod["distance_mult"]
            * self.nat_mod["distance_mult"]
            * self.edu_mod["distance_mult"]
            * self.job_mod["distance_mult"]
            * self.time_mod["distance_mult"]
            * self.circ_profile["distance_mult"]
        )

        # Effective spiral probability — sum of all deltas
        base_spiral = self.base_profile["panic_spiral_prob"]
        self.effective_spiral = min(0.95, max(0.0,
            base_spiral
            + self.age_mod.get("spiral_delta", 0.0)
            + self.nat_mod.get("spiral_delta", 0.0)
            + self.edu_mod.get("spiral_delta", 0.0)
            + self.job_mod.get("spiral_delta", 0.0)
            + self.time_mod.get("spiral_delta", 0.0)
            + self.circ_profile.get("spiral_delta", 0.0)
        ))

        # Effective attractors — merge job overrides on top of circumstance,
        # then apply cumulative trail boost from all modifiers
        self.effective_attractors = dict(self.base_profile["attractors"])
        for k, v in self.circ_profile.get("attractor_overrides", {}).items():
            self.effective_attractors[k] = v
        for k, v in self.job_mod.get("attractor_overrides", {}).items():
            # Job attractors refine (average) rather than fully override circumstance
            existing = self.effective_attractors.get(k, v)
            self.effective_attractors[k] = round((existing + v) / 2, 3)
        # Trail-following boost: gender + nationality + education + job
        trail_boost_total = (
            self.gender_mod.get("trail_boost", 0.0)
            + self.nat_mod.get("trail_boost", 0.0)
            + self.edu_mod.get("trail_boost", 0.0)
            + self.job_mod.get("trail_boost", 0.0)
        )
        if "trail_following" in self.effective_attractors:
            self.effective_attractors["trail_following"] = min(
                1.0, self.effective_attractors["trail_following"] + trail_boost_total
            )
        if "ridgeline" in self.effective_attractors:
            self.effective_attractors["ridgeline"] = min(
                1.0, self.effective_attractors.get("ridgeline", 0) + self.gender_mod["ridgeline_boost"]
            )

        # Path prediction (set later after detections known)
        self.path_prediction: Optional[Dict] = None

    # ──────────────────────────────────────────────────────────────────────
    @tracer.start_as_current_span("psych_score_grid")
    def score_grid(
        self,
        grid_cells: List[Dict],
        last_known: Dict,
        detections: List[Dict],
    ) -> List[Dict]:
        """Score each grid cell with probability of subject presence."""

        # Run path prediction from traces
        self.path_prediction = _predict_travel_direction(detections)

        lk_lat, lk_lng = last_known["lat"], last_known["lng"]
        avg_dist = self.effective_distance_km
        spiral_prob = self.effective_spiral

        scored = []
        for cell in grid_cells:
            dist_km = self._haversine_km(lk_lat, lk_lng, cell["lat"], cell["lng"])

            # 1. Distance decay (Gaussian centred at 0.5× effective avg distance)
            dist_score = math.exp(-0.5 * ((dist_km - avg_dist * 0.5) / max(avg_dist, 0.1)) ** 2)

            # 2. Detection proximity boost (confidence-weighted inverse distance)
            det_score = 0.0
            for det in detections:
                d = self._haversine_km(
                    cell["lat"], cell["lng"],
                    det["location"]["lat"], det["location"]["lng"]
                )
                if d < 1.5:
                    det_score += det["confidence"] * math.exp(-d * 2.5)
            det_score = min(1.0, det_score)

            # 3. Terrain attractor score (with age steep-slope penalty)
            terrain_type = cell.get("terrain_type", "mixed_forest")
            attractor_key = TERRAIN_ATTRACTOR_MAP.get(terrain_type, "shelter")
            terrain_score = self.effective_attractors.get(attractor_key, 0.30)

            # Elderly/injured subjects cannot handle steep slopes
            if terrain_type == "steep_slope":
                terrain_score = max(0.0, terrain_score - self.age_mod.get("steep_penalty", 0.0)
                                    - (0.4 if self.circumstance in ("injured", "medical_emergency") else 0.0))

            # 4. Panic spiral boost (concentric ring near LKP)
            spiral_boost = spiral_prob * 0.3 if (spiral_prob > 0.4 and 0.3 < dist_km < 1.5) else 0.0

            # 5. Direction-of-travel boost (from trace path prediction)
            direction_boost = 0.0
            if self.path_prediction and not self.path_prediction["direction_change_detected"]:
                proj_lat = self.path_prediction["projected_lat"]
                proj_lng = self.path_prediction["projected_lng"]
                d_to_proj = self._haversine_km(cell["lat"], cell["lng"], proj_lat, proj_lng)
                if d_to_proj < 1.0:
                    direction_boost = (
                        self.path_prediction["confidence"]
                        * math.exp(-d_to_proj * 3.0)
                        * 0.35  # max 35% weight boost for direction alignment
                    )

            # Combined score — weights sum to ~1.0
            raw = (
                0.25 * dist_score
                + 0.35 * det_score
                + 0.15 * terrain_score
                + 0.08 * spiral_boost
                + 0.17 * direction_boost
            )

            scored.append({
                **cell,
                "probability": round(min(1.0, raw), 4),
                "dist_from_lkp_km": round(dist_km, 2),
                "detection_score": round(det_score, 3),
                "terrain_score": round(terrain_score, 3),
                "direction_boost": round(direction_boost, 3),
            })

        # Soft-normalise so max probability = 1.0
        max_p = max((c["probability"] for c in scored), default=1.0) or 1.0
        for cell in scored:
            cell["probability"] = round(cell["probability"] / max_p, 3)

        return scored

    # ──────────────────────────────────────────────────────────────────────
    @tracer.start_as_current_span("psych_generate_insights")
    def get_behavioral_insights(self) -> Dict:
        """Return full enriched insight dict to broadcast to the dashboard."""
        base = self.base_profile
        circ = self.circ_profile

        # Build urgency — take strictest of: base, circumstance override, time floor
        urgency_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
        raw_urgency  = circ.get("urgency_override") or base["urgency"]
        time_floor   = self.time_mod.get("urgency_floor")
        if time_floor and urgency_order.get(time_floor, 0) > urgency_order.get(raw_urgency, 0):
            urgency = time_floor
        else:
            urgency = raw_urgency

        # Compose advice list: time (critical first) + base + circumstance + nationality + education + job + demographic
        time_advice = list(self.time_mod.get("advice", []))
        advice = list(base["key_advice"])
        advice.extend(circ.get("advice", []))
        advice.extend(self.nat_mod.get("advice", []))
        if self.nat_mod.get("language_barrier"):
            advice.insert(0, "⚠️ Language barrier likely — arrange multilingual communication immediately")
        advice.extend(self.job_mod.get("advice", []))
        if self.age is not None:
            advice.append(f"Age factor: {self.age_mod['age_note']}")
        if self.gender_mod["gender_note"] != "Gender unknown — using base profile":
            advice.append(f"Gender factor: {self.gender_mod['gender_note']}")
        if self.edu_mod.get("education_note") and "unknown" not in self.edu_mod["education_note"]:
            advice.append(f"Education factor: {self.edu_mod['education_note']}")
        advice.extend(self.hist_mod.get("advice", []))
        # Time advice goes at the very front (most operationally urgent)
        advice = time_advice + advice

        # Compose behavior description
        behavior_parts = [base["typical_behavior"]]
        if circ.get("behavior_note"):
            behavior_parts.append(circ["behavior_note"])
        if self.nat_mod.get("sar_note") and "unknown" not in self.nat_mod.get("sar_note", ""):
            behavior_parts.append(self.nat_mod["sar_note"])
        if self.job_mod.get("job_note") and "unknown" not in self.job_mod.get("job_note", ""):
            behavior_parts.append(f"[{self.job or 'Occupation'}] {self.job_mod['job_note']}")
        # Time context appended last (operational status)
        behavior_parts.append(self.time_mod["elapsed_note"])
        behavior_parts.append(self.time_mod["time_note"])

        insights = {
            "subject_type": self.subject_type,
            "description": base["description"],
            "age": self.age,
            "gender": self.gender,
            "nationality": self.nationality,
            "education_level": self.education_level,
            "education_field": self.education_field,
            "job": self.job,
            "circumstance": self.circumstance,
            "urgency": urgency,

            # Effective (modifier-applied) values shown to operators
            "effective_distance_km": round(self.effective_distance_km, 2),
            "base_distance_km": base["avg_distance_km"],
            "found_alive_rate": base["found_alive_rate"],
            "panic_spiral_prob": round(self.effective_spiral, 2),
            "language_barrier": self.nat_mod.get("language_barrier", False),

            # Time context — always included
            "time_context": {
                "hours_missing":         self.time_mod["hours_missing"],
                "elapsed_str":           self.time_mod["elapsed_str"],
                "entry_time":            self.time_mod["entry_time_str"],
                "search_start_time":     self.time_mod["search_start_time_str"],
                "time_of_day_window":    self.time_mod["time_of_day_window"],
                "medical_risk":          self.time_mod["medical_risk"],
                "track_max_age_h":       self.time_mod["track_max_age_h"],
                "urgency_floor":         self.time_mod["urgency_floor"],
            },

            # Historical context
            "historical_context": {
                "has_data":            self.hist_mod["has_data"],
                "similar_cases_count": self.hist_mod["similar_cases_count"],
                "avg_distance_km":     self.hist_mod["avg_distance_km"],
                "avg_hours_to_find":   self.hist_mod["avg_hours_to_find"],
                "found_alive_rate":    self.hist_mod["found_alive_rate"],
                "common_factors":      self.hist_mod["common_factors"],
                "historical_note":     self.hist_mod["historical_note"],
            },

            "typical_behavior": " ".join(behavior_parts),
            "key_advice": advice,

            # Full modifier breakdown for operator transparency
            "modifiers_applied": {
                "age": self.age_mod,
                "gender": self.gender_mod,
                "nationality": {
                    "group": self.nat_mod.get("group"),
                    "distance_mult": self.nat_mod["distance_mult"],
                    "spiral_delta": self.nat_mod["spiral_delta"],
                    "language_barrier": self.nat_mod["language_barrier"],
                    "sar_note": self.nat_mod["sar_note"],
                },
                "education": self.edu_mod,
                "job": {
                    "matched_group": self.job_mod.get("matched_group", "unknown"),
                    "distance_mult": self.job_mod["distance_mult"],
                    "spiral_delta": self.job_mod["spiral_delta"],
                    "trail_boost": self.job_mod["trail_boost"],
                    "shelter_boost": self.job_mod.get("shelter_boost", 0),
                    "job_note": self.job_mod["job_note"],
                },
                "circumstance": {
                    "name": self.circumstance,
                    "distance_mult": circ["distance_mult"],
                    "spiral_delta": circ.get("spiral_delta", 0),
                },
            },

            # Path prediction from traces
            "path_prediction": self.path_prediction,
        }

        return insights

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
