# Universal API Contract (Gaman AI)

The engine must be completely agnostic to the contents of the `state`. It treats the state simply as contextual grounding for the evaluation primitives.

## 1. The `noul` (Predicate) Primitive
**Input:**
```json
{
  "state": {"user_id": 123, "action": "delete_all"},
  "predicate": "Is this a destructive action?"
}
```
**Output:**
```json
{
  "primitive": "noul",
  "passed": true,
  "probability": 0.982,
  "latency_ms": 12.4
}
```

## 2. The `choice` (Routing) Primitive
**Input:**
```json
{
  "state": {"cpu_usage": 98, "memory_usage": 85},
  "options": ["scale_up", "scale_down", "do_nothing"]
}
```
**Output:**
```json
{
  "primitive": "choice",
  "selection": "scale_up",
  "confidence": 0.941,
  "margin": 0.882,
  "entropy": 0.231,
  "tier": "HIGH",
  "escalate_to_system2": false,
  "probabilities": {
    "scale_up": 0.941,
    "scale_down": 0.059,
    "do_nothing": 0.000
  },
  "latency_ms": 14.1
}
```

## 3. The `score` (Evaluator) Primitive
**Input:**
```json
{
  "state": {"review": "The product broke after two days of use."},
  "criterion": "Severity of hardware failure"
}
```
**Output:**
```json
{
  "primitive": "score",
  "value": 0.895,
  "latency_ms": 11.2
}
```

## 4. Enterprise Decision Contract & Robustness Invariants
When evaluating unstructured text payloads or routing decisions via `choice()` / `decide()`:
- **Decision Metadata:** Emits `selection`, calibrated `confidence`, normalized `margin` ($p_{(1)} - p_{(2)}$), Shannon `entropy` ($-\sum p_i \ln p_i$), confidence `tier` (`HIGH` $\ge 0.70$ with margin $\ge 0.35$, `MEDIUM` $\ge 0.50$, or `LOW`), and `escalate_to_system2` boolean trigger.
- **Pre-Tokenization Sanitization:** Automatically strips conversational boilerplate/signoffs and restructures adversative clauses (`instead`, `however`, `rather than`, counterfactuals) to eliminate the Sandwich Trap and align positional attention with authentic communicative intent.
- **Logit Regularization:** Zero-centers logits across options to preserve mathematical shift invariance, clamps extreme uncalibrated logit divergence to $[-8.0, 8.0]$, and dynamically expands temperature $T_{\text{eff}} = T \cdot (1.0 + 0.25 \cdot N_{\text{conflicts}})$ when candidate options exhibit lexical conflict.