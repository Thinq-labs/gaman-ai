# Gaman AI

**A Deterministic, Sub-Millisecond System One AI Decision Engine.**

Generative Large Language Models (LLMs) are over-engineered for control-plane tasks. They are autoregressive, latency-bound, computationally expensive, and fundamentally non-deterministic. Gaman AI provides modern software architectures with a low-latency, strictly typed "System One" reflex layer. 

By discarding autoregressive text generation in favor of a highly optimized, encoder-only architecture, Gaman AI executes parallel evaluation passes over input states to produce calibrated, deterministic classifications.

## Core Capabilities

Gaman AI is designed to replace generative LLMs in critical control flows where speed, reliability, and structured outputs are non-negotiable.

*   **Semantic Routing (Discrete Choices):** Route incoming requests, payloads, or agentic actions to strict predefined categories in $O(1)$ temporal steps.
*   **Security & Guardrails (Strict Probabilities):** Execute sub-10ms binary classification (Yes/No) on inputs/outputs with calibrated epistemic uncertainty to prevent prompt injection, PII leakage, or malicious intent.
*   **RAG Context Scoring (Normalized Ranges):** Score and filter retrieval-augmented generation context windows using cross-encoder topology, outputting normalized scores to instantly discard irrelevant context.

## Architectural Overview

Gaman AI is built on a deep, narrow bidirectional transformer topology, compiled and optimized for edge-adjacent deployment.

### 1. The Inference Pipeline
*   **Non-Autoregressive:** Zero token generation. The engine computes the input state and evaluation criteria in a single forward pass using bidirectional attention.
*   **Typed Classification Heads:** Outputs are structurally guaranteed. The final network layers consist of specialized multi-layer perceptrons (MLPs) outputting raw logit vectors, continuous scalars, or binary probabilities—not generated JSON strings.
*   **Calibrated Confidence:** Logits are calibrated using Temperature Scaling, ensuring that the model's confidence scores reflect true statistical probabilities, enabling strict thresholding in application logic.

### 2. Execution & Deployment
*   **Runtime:** Compiled to ONNX and strictly quantized to INT8.
*   **Compute Footprint:** Requires <300MB of memory. Gaman AI is designed to run directly alongside application code (via ONNX Runtime, OpenVINO, or Rust-based ML runtimes) without relying on external GPU clusters or remote API calls.
*   **Latency:** Sub-10 millisecond execution on standard CPU hardware, capable of handling thousands of requests per second in high-tick-rate game loops, message queues, and event-driven microservices.

## Conceptual Usage

Gaman AI integrates directly into your backend architecture. (Note: API design is currently in Phase 3 development).

```python
from gaman import SystemOneEngine, RoutingSchema

# Load optimized ONNX runtime
engine = SystemOneEngine.load("gaman-core-int8.onnx", provider="cpu")

# Define strict routing criteria
schema = RoutingSchema(
    categories=["sql_query", "casual_chat", "system_command"]
)

# Execute deterministic evaluation (0 generation tokens)
result = engine.route(
    state="Drop all tables from the user database.",
    schema=schema
)

print(result.category) # Output: system_command
print(result.confidence) # Output: 0.992
```

## Contributing

We are currently building out the Phase 4 MVP open-source roadmap. Contributions focusing on ONNX quantization, Rust inference bindings, and custom loss functions (Focal Loss for classification heads) are welcome. Please refer to the open issues for current milestones.

## License

This project is licensed under the MIT License - see the LICENSE file for details.