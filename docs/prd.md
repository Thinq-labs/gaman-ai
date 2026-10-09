# Product Requirements Document (PRD): Gaman AI

## 1. Product Vision
Gaman AI is an open-source, local-first "System One" decision engine. Modern software architectures use LLMs for routing, policy enforcement, and logic checks—but generative models are too slow, expensive, and unpredictable for control-plane tasks. 

Gaman AI discards the autoregressive decoding loop entirely. It uses highly optimized, quantized bidirectional encoders to evaluate states (JSON/Text) in a single parallel forward pass, returning deterministic, strictly typed classifications. It brings machine learning to the edge, operating at the speed of a standard software function.

## 2. Target Audience
- **Backend & Systems Engineers:** Need sub-second routing for microservices and event-driven architectures.
- **DevSecOps:** Need instant, local policy evaluations (e.g., CI/CD pipeline scans) without sending proprietary code to cloud APIs.
- **Game Developers:** Need high-tick-rate decision engines for NPC behavior that won't bottleneck the main game loop.
- **Agentic AI Builders:** Need deterministic traffic cops for multi-agent DAGs (Directed Acyclic Graphs).

## 3. Core Primitives (The "What")
The engine must strictly support only three operations. It will NEVER generate free-form text.
1. **`choice` (Semantic Routing):** Given a state and $K$ options, output a discrete selection with a calibrated confidence score.
2. **`noul` (Predicate / Guardrail):** Given a state and a policy, output a strict Yes/No probability representing policy compliance.
3. **`score` (Context Evaluator):** Given a state and a criterion, output a continuous scalar bounded between $0.0$ and $1.0$.

## 4. Competitive Positioning
Gaman AI competes against OpenAI's Decisions API and TypeSafe's Jev AI by attacking their shared weakness: the network.
- **Zero Network Latency:** Runs directly in the application's memory space.
- **Absolute Privacy:** Processes all payloads locally; suitable for HIPAA/SOC2 compliant environments.
- **Zero Token Costs:** Eliminates API variable costs; executes entirely on local consumer hardware.

## 5. Non-Functional Requirements (Strict Constraints)
- **Latency Target:** Sub-20ms per evaluation on an average consumer CPU (e.g., Apple M-series or Intel i5). Sub-10ms with GPU acceleration.
- **Memory Footprint:** Static RAM usage must remain under 200MB (no KV-cache bloat).
- **Determinism:** The same input state MUST yield the exact same floating-point probability and choice every time. Zero stochastic variation.
- **Hardware Profile:** Must run natively on consumer hardware using ONNX Runtime. No dependency on high-end NVIDIA server GPUs for inference.

## 6. Success Metrics for MVP (v0.1)
1. Successfully parses a structured JSON payload natively.
2. Executes a `choice`, `noul`, and `score` operation via CLI or Python SDK.
3. Automatically falls back to CPU vector instructions if no GPU is detected, without crashing.
4. Processes the HackerRank Orchestrate `messages.csv` dataset iteratively with 100% schema compliance.