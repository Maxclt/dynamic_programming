# Dynamic Programming Project

This project, **dynamic-programming**, is a Python-based implementation and exploration of dynamic programming techniques. It includes dependencies for machine learning, data manipulation, and visualization. 

## Project Information

- **Name**: `dynamic-programming`
- **Version**: `0.1.0`
- **Description**: Dynamic programming project
- **Author**: Maxime Coulet `<maxime.coulet@ensae.fr>`

## Project Overview

This repository implements the problem and provides tools for studying different estimation techniques, including:

- Bellman operators for dynamic programming
- Greedy methods for exploring policies and estimating the optimal value function and policy
- Neural network methods for function approximation

The project includes structured notebooks, source code, and utilities for experimentation.

---

## Repository Structure

### Notebooks

- **`notebooks/advanced_programming_project.ipynb`**: 
  The main notebook for studying the problem, running experiments, and visualizing results.

### Source Code

1. **`src/dynamic_programming/`**:
   - `bellman_operator.py`: Implements the Bellman operator for solving dynamic programming problems.
   - `epsilon_greedy.py`: Contains the greedy algorithm for estimating the optimal value function and policy.

2. **`src/neural_network/`**:
   - `nn_method.py`: Implements neural network methods for approximating the value function.

3. **`src/problem/`**:
   - `opti_problem.py`: Encapsulates the definition of the intertemporal optimization problem.

4. **`utils/`**:
   - Utility functions for logging, debugging, and auxiliary tasks.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd dynamic-programming

## Prerequisites

- Python version `>=3.12, <3.13` is required.
- Ensure you have `poetry` installed on your system. If not, install it using the following command:
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
