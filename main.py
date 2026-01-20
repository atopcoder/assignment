from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Context Matcher")

# --- Data models ---
class MatchInput(BaseModel):
    host_element: str
    adjacent_element: str
    exposure: str

class MatchOutput(BaseModel):
    suggested_detail: str
    confidence: float
    reason: str

# --- Hardcoded sample details (in-memory) ---
SAMPLE_DETAILS: List[Dict[str, Any]] = [
    {
        "id": "D-001",
        "title": "External Wall–Slab Junction Waterproofing",
        "host": "external wall",
        "adjacent": "slab",
        "exposure": "external",
        "notes": "Typical waterproofing and junction details for above-grade external wall to slab."
    },
    {
        "id": "D-002",
        "title": "Internal Partition–Slab Recess",
        "host": "internal partition",
        "adjacent": "slab",
        "exposure": "internal",
        "notes": "Interior partition details where a slab recess occurs."
    },
    {
        "id": "D-003",
        "title": "Curtain Wall–CMU Sill Detail",
        "host": "curtain wall",
        "adjacent": "cmu",
        "exposure": "external",
        "notes": "Sill interface with masonry backup (CMU)."
    },
    {
        "id": "D-004",
        "title": "Roof Edge–Parapet Flashing",
        "host": "roof edge",
        "adjacent": "parapet",
        "exposure": "external",
        "notes": "Roof edge termination and flashing detail."
    },
    {
        "id": "D-005",
        "title": "Foundation Wall–Footing Junction",
        "host": "foundation wall",
        "adjacent": "footing",
        "exposure": "external",
        "notes": "Below-grade foundation wall to footing connection detail."
    }
]

# --- Simple normalization helpers ---
# Order matters: longer synonyms must come first to avoid partial replacements
NORMALIZATION_SYNONYMS = [
    ("exterior", "external"),
    ("interior", "internal"),
    ("concrete slab", "slab"),
    ("masonry", "cmu"),
    ("ext wall", "external wall"),
    ("int wall", "internal wall"),
]


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    # apply synonym replacements (order matters)
    for old, new in NORMALIZATION_SYNONYMS:
        if old in s:
            s = s.replace(old, new)
    return s

# --- Scoring logic ---
# weights define importance of each attribute (sums to 1.0)
WEIGHTS = {"host": 0.45, "adjacent": 0.35, "exposure": 0.20}


def score_match(input_normalized: Dict[str, str], detail: Dict[str, Any]) -> tuple[float, List[str]]:
    """Return a score between 0.0 and 1.0 and a list of textual reasons for matches/mismatches."""
    reasons: List[str] = []
    total_score = 0.0

    # host comparison
    host_input = input_normalized["host"]
    host_detail = normalize(detail.get("host", ""))
    if host_input == host_detail and host_input:
        total_score += WEIGHTS["host"] * 1.0
        reasons.append("Exact match on host")
    elif host_input and host_detail and (host_input in host_detail or host_detail in host_input):
        total_score += WEIGHTS["host"] * 0.6
        reasons.append("Partial match on host")
    else:
        reasons.append("Host differs")

    # adjacent comparison
    adj_input = input_normalized["adjacent"]
    adj_detail = normalize(detail.get("adjacent", ""))
    if adj_input == adj_detail and adj_input:
        total_score += WEIGHTS["adjacent"] * 1.0
        reasons.append("Exact match on adjacent element")
    elif adj_input and adj_detail and (adj_input in adj_detail or adj_detail in adj_input):
        total_score += WEIGHTS["adjacent"] * 0.6
        reasons.append("Partial match on adjacent element")
    else:
        reasons.append("Adjacent element differs")

    # exposure comparison
    exp_input = input_normalized["exposure"]
    exp_detail = normalize(detail.get("exposure", ""))
    if exp_input == exp_detail and exp_input:
        total_score += WEIGHTS["exposure"] * 1.0
        reasons.append("Exact match on exposure")
    else:
        # penalize mismatch but still allow for partial knowledge
        if exp_input and exp_detail:
            reasons.append(f"Exposure differs (input={exp_input}, detail={exp_detail})")
        else:
            reasons.append("Exposure unknown")

    # clamp between 0 and 1
    total_score = max(0.0, min(total_score, 1.0))
    return total_score, reasons


# --- API endpoints ---
@app.get("/details")
def list_details() -> List[Dict[str, Any]]:
    """Return the in-memory detail library (short-form)."""
    return [{"id": d["id"], "title": d["title"], "host": d["host"], "adjacent": d["adjacent"], "exposure": d["exposure"]} for d in SAMPLE_DETAILS]


@app.post("/match", response_model=MatchOutput)
def match_detail(payload: MatchInput) -> MatchOutput:
    """Match the input to the best sample detail and return a suggested detail with confidence and reason."""
    input_norm = {
        "host": normalize(payload.host_element),
        "adjacent": normalize(payload.adjacent_element),
        "exposure": normalize(payload.exposure),
    }

    # score every detail
    best = None
    best_score = -1.0
    best_reasons: List[str] = []
    for d in SAMPLE_DETAILS:
        s, reasons = score_match(input_norm, d)
        if s > best_score:
            best_score = s
            best = d
            best_reasons = reasons

    # interpret score into final confidence and reason
    if best is None:
        return MatchOutput(suggested_detail="No match found", confidence=0.0, reason="No samples available")

    # map raw score to a human-friendly confidence (we keep it simple)
    confidence = round(best_score, 2)

    # craft reason: prefer concise sentence
    if confidence >= 0.85:
        reason_text = "High confidence: " + "; ".join(best_reasons)
    elif confidence >= 0.5:
        reason_text = "Medium confidence: " + "; ".join(best_reasons)
    else:
        reason_text = "Low confidence: " + "; ".join(best_reasons)

    # if very low confidence, still return the best candidate but mark low
    return MatchOutput(
        suggested_detail=best["title"],
        confidence=confidence,
        reason=reason_text,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")