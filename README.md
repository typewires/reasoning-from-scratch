# Reasoning From Scratch — Learning Journey

My personal implementation and notes while working through Sebastian Raschka's [Build a Reasoning Model (From Scratch)](https://mng.bz/lZ5B).

## About

This repository documents my self-study of reasoning LLMs, following the book and code by Sebastian Raschka. The original code is available at [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch).

## Chapters

- [Chapter 1: Understanding Reasoning Models](https://github.com/typewires/reasoning-from-scratch/tree/main/chapter1)
- [Chapter 2: Generating Text with a Pre-trained LLM](https://github.com/typewires/reasoning-from-scratch/tree/main/chapter2)
- [Chapter 3: Evaluating Reasoning Models](https://github.com/typewires/reasoning-from-scratch/tree/main/chapter3)
- [Chapter 4: Improving Reasoning with Inference-Time Scaling](https://github.com/typewires/reasoning-from-scratch/tree/main/chapter4)
- Chapter 5: Inference-Time Scaling via Self-Refinement
- Chapter 6: Training Reasoning Models with Reinforcement Learning
- Chapter 7: Distilling Reasoning Models for Efficient Reasoning
- Chapter 8: Improving the Reasoning Pipeline and Future Directions

## My Notes & Experiments

*I'll add my own observations, experiments, and things I tried differently here as I work through the material.*

## Setup

### Quick Start (Recommended)

First, install [uv](https://docs.astral.sh/uv/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then run the project:
```bash
# Clone this repo
git clone https://github.com/typewires/reasoning-from-scratch.git
cd reasoning-from-scratch

# Run Jupyter Lab (uv handles virtual env + dependencies automatically)
uv run jupyter lab
```

### Traditional Setup (Alternative)
```bash
# Clone this repo
git clone https://github.com/typewires/reasoning-from-scratch.git
cd reasoning-from-scratch

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
jupyter lab
```

## Acknowledgments

All credit for the original code and methodology goes to [Sebastian Raschka](https://github.com/rasbt). This repo is for personal learning purposes.

## License

Original code is [Apache 2.0 licensed](https://github.com/rasbt/reasoning-from-scratch/blob/main/LICENSE).