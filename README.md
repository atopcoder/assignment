# Context Matching Tool for AEC Professionals

A FastAPI service that suggests the most relevant construction detail based on context input.

## The Problem

AEC professionals (architects, engineers, BIM teams) need to find the right **construction detail** based on context:
- **Host element**: Primary building element (wall, roof, floor)
- **Adjacent element**: What it connects to (slab, parapet, foundation)
- **Exposure**: Internal vs external (determines waterproofing/fire needs)

## Assumptions

1. **Inputs are structured and short**, generally following consistent naming (e.g., "External Wall", "Slab", "External")
2. **Limited detail library**: Uses 5 hardcoded sample details for demonstration. A production system would connect to a database
3. **Text-based matching**: Matching is purely text-based with synonym normalization. No semantic/ML matching
4. **Single best match**: Returns only the top match, not a ranked list
5. **Confidence is a heuristic score** designed to be understandable and stable—not a statistically calibrated probability

## Matching Logic

### Scoring Approach

Construction details describe **junctions** between elements. Match by:

1. **Normalize** inputs (lowercase, map synonyms like "exterior"→"external", "masonry"→"cmu")
2. **Score** each field:
   - Exact match: 1.0
   - Partial match (substring): 0.6
   - No match: 0.0
3. **Weight** scores and sum: `host(0.45) + adjacent(0.35) + exposure(0.20) = confidence`
4. **Return** the detail with highest confidence score

### Why These Weights?

| Field    | Weight | Rationale |
|----------|--------|-----------|
| Host     | 45%    | Most defines the detail type (wall vs roof vs floor) |
| Adjacent | 35%    | Critical for junction specifics (slab vs parapet vs footing) |
| Exposure | 20%    | Affects finishing layers, not core detail structure |

### Sample Details

| ID    | Title                              | Host               | Adjacent | Exposure |
|-------|------------------------------------|--------------------|----------|----------|
| D-001 | External Wall–Slab Waterproofing   | external wall      | slab     | external |
| D-002 | Internal Partition–Slab Recess     | internal partition | slab     | internal |
| D-003 | Curtain Wall–CMU Sill Detail       | curtain wall       | cmu      | external |
| D-004 | Roof Edge–Parapet Flashing         | roof edge          | parapet  | external |
| D-005 | Foundation Wall–Footing Junction   | foundation wall    | footing  | external |

## Running the Service

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Start the Server

```bash
uv run uvicorn main:app --reload
```

Server runs at `http://localhost:8000`.

### Verify

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{"host_element":"External Wall","adjacent_element":"Slab","exposure":"External"}'
```

Expected response:
```json
{
  "suggested_detail": "External Wall–Slab Junction Waterproofing",
  "confidence": 1.0,
  "reason": "High confidence: Exact match on host; Exact match on adjacent element; Exact match on exposure"
}
```

### API Endpoints

#### `GET /details`
List all available construction details.

```bash
curl http://localhost:8000/details
```

#### `POST /match`
Find the best matching detail for given context.

#### Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Future Improvements

### Quick Wins (High Priority)

1. **Fuzzy string matching with RapidFuzz**: Replace substring matching to handle typos and variations
   ```python
   from rapidfuzz.distance import JaroWinkler

   # Handles typos: "parpaet" ≈ "parapet" (0.92 similarity)
   # Handles variations: "ext wall" ≈ "external wall" (0.85 similarity)
   score = JaroWinkler.normalized_similarity(input_str, target_str)
   ```

2. **Explicit NO_MATCH threshold**: Return `null` when confidence is too low instead of a misleading suggestion
   ```python
   MINIMUM_CONFIDENCE = 0.4
   if best_score < MINIMUM_CONFIDENCE:
       return {"suggested_detail": None, "reason": "No matching detail found"}
   ```

3. **Gap-based confidence calibration**: Use gap between best and second-best match for more meaningful confidence
   ```python
   gap = best_score - second_best_score
   if best_score >= 0.8 and gap >= 0.2:
       quality = "high"  # Clear winner
   elif best_score >= 0.6 and gap >= 0.1:
       quality = "medium"  # Good but has close alternatives
   ```

4. **Return top-N candidates**: Provide top 3 suggestions with confidence scores for better UX

### Medium Priority

5. **Construction taxonomy**: Leverage domain hierarchies for partial credit matching
   ```
   wall → external wall → curtain wall → unitized curtain wall
   ```
   - "curtain wall" matches "wall" with ~0.7 credit
   - "cmu" matches "masonry wall" through parent relationship

6. **BM25 ranking**: Better than substring matching for larger detail libraries
   ```python
   from rank_bm25 import BM25Okapi
   # Handles term frequency saturation and document length normalization
   ```

7. **Ensemble matching**: Combine multiple strategies with weighted voting
   | Strategy | Weight | Best For |
   |----------|--------|----------|
   | Exact | 1.0 | Perfect matches |
   | Taxonomy | 0.85 | Domain relationships |
   | Fuzzy | 0.75 | Typos, variations |
   | BM25 | 0.65 | Large libraries |

### Other Improvements

8. **Database integration**: Replace hardcoded details with a searchable database
9. **Testing**: Add unit tests for scoring logic and edge cases
10. **Production hardening**: Structured error responses, logging/metrics, rate limiting

### Dependencies for Improvements
```toml
dependencies = [
    "rapidfuzz>=3.0.0",
    "rank-bm25>=0.2.0",
]
```

## Advanced: Semantic Matching with Embeddings

For cases where text-based matching falls short (e.g., "Facade wall" won't match "External wall"), embeddings provide semantic understanding.

### Embeddings Approach

Use text embeddings to capture semantic meaning:

```python
from openai import OpenAI  # or sentence-transformers for local models

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# At startup: embed all details
for detail in DETAILS:
    detail["embedding"] = embed(
        f"{detail['host']} {detail['adjacent']} {detail['exposure']} {detail['notes']}"
    )

# At query time: embed input, find nearest neighbors
def match(input: MatchInput):
    query_embedding = embed(
        f"{input.host_element} {input.adjacent_element} {input.exposure}"
    )
    scores = [cosine_sim(query_embedding, d["embedding"]) for d in DETAILS]
    best_idx = argmax(scores)
    return DETAILS[best_idx], scores[best_idx]
```

### Vector DB Options

| DB | Use Case | Notes |
|----|----------|-------|
| **pgvector** | Already using Postgres | Just add extension |
| **Pinecone** | Managed, scales easily | SaaS cost |
| **Qdrant** | Self-hosted, fast | Good for on-prem |
| **ChromaDB** | Local/dev, simple API | Good for prototyping |

### Trade-offs

| Aspect | Current (Text) | Embeddings |
|--------|----------------|------------|
| Latency | <1ms | 50-200ms (API call) |
| Cost | Free | ~$0.0001/query |
| Explainability | Clear ("exact match on host") | Less transparent |
| Offline | Yes | Needs API or local model |
| Semantic matching | No | Yes ("Facade" ↔ "External") |

### Recommended: Hybrid Approach

Combine both methods for best results:

```python
def match(input: MatchInput):
    # 1. Try exact/weighted match first
    score, detail = weighted_match(input)
    if score >= 0.8:
        return detail, score, "exact"

    # 2. Fall back to semantic search for fuzzy queries
    detail, sim = vector_search(input)
    return detail, sim, "semantic"
```

This gives fast exact matches when possible, with semantic fallback for ambiguous queries.
