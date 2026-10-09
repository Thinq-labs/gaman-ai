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