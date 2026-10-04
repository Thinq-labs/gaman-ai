# Neo AI

A minimalist, zero-dependency neural network and autograd engine built from scratch to visualize parameters, activations, and backpropagation in real time. 

Neo AI strips away the abstraction of modern machine learning frameworks. By implementing a custom scalar-level automatic differentiation engine and representing operations as a Directed Acyclic Graph (DAG), this project exposes the exact mathematical mechanics that power neural networks. It pairs a pure mathematical backend with an interactive frontend, allowing users to physically see how weights, biases, and gradients manipulate decision boundaries.

## Core Features

* **Zero-Framework Autograd:** A completely custom `Value` class that tracks scalar operations, stores local gradients, and dynamically builds a computational DAG for automatic differentiation.
* **Real-Time Backpropagation:** Step-by-step reverse topological traversal of the network graph, calculating the exact chain-rule derivatives for every node.
* **Interactive Architecture:** Adjust individual weights or biases in real time and instantly observe the cascading effects on hidden activations and the final loss function.
* **Visual Decision Boundaries:** A live 2D projection showing how the model classifies space and how that classification warps during the training loop.

## System Architecture

Neo AI is divided into three decoupled layers:

1. **The Core Engine:** The mathematical foundation. Handles scalar arithmetic, matrix dot products, activation functions (ReLU, Tanh), and the MSE/Binary Cross-Entropy loss calculations without relying on external ML libraries.
2. **The Tracing System:** The state manager. It serializes the computational DAG and records snapshots of all network parameters before and after optimizer steps, enabling time-scrubbing.
3. **The Visualization Interface:** The interactive surface. Renders the network topology and maps the underlying mathematical state to visual indicators (edge thickness for weight magnitude, node color for activation strength).

## Getting Started

### Prerequisites
* [Node.js](https://nodejs.org/) (v18 or higher)
* [Python 3.10+](https://www.python.org/) (if running the decoupled backend)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/neo-ai.git](https://github.com/yourusername/neo-ai.git)
   cd neo-ai
