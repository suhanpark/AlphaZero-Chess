# AlphaZero Chess

An AlphaZero-style chess engine implementation using PyTorch and Monte Carlo Tree Search (MCTS).

## Project Structure

```text
alphazero-chess/
├─ src/
│  ├─ features.py      # Board encoding and action mapping
│  ├─ env.py          # Chess environment wrapper
│  ├─ network.py      # Neural network architecture
│  ├─ mcts.py         # Monte Carlo Tree Search implementation
│  ├─ selfplay.py     # Self-play game generation
│  ├─ train.py        # Training loop
│  ├─ evaluate.py     # Model evaluation utilities
│  ├─ export_onnx.py  # ONNX model export
│  └─ cli_play.py     # Command-line play interface
├─ requirements.txt
└─ README.md
```

## Installation

```bash
conda create -n alphazero-chess python=3.12 -y
conda activate alphazero-chess
pip install -r requirements.txt
```

## Usage

### Training (on GPU)

```python
# Run training loop
python -m src.train
```

### Export to ONNX

```python
# Export trained model to ONNX format
python -m src.export_onnx
```

### Play against the AI

```bash
# Play via command line interface
python -m src.cli_play
```

## Features

- **18-channel board encoding**: 12 piece planes + side-to-move + castling rights + move parity
- **4096 action space**: Simple from×to encoding with auto-queen promotions
- **Residual neural network**: Configurable width and depth
- **MCTS with Dirichlet noise**: For exploration during self-play
- **ONNX export**: For deployment and inference optimization

## Notes

- Uses simplified 64×64 action space for v0 (promotions default to queen)
- Optimized for single-GPU training on Google Colab A100
- CLI play runs efficiently on CPU using ONNX inference

## For v1

- Implement multiprocessing for self-play
- Add explicit promotion handling
- Create evaluation tournaments
