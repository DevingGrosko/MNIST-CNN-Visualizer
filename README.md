# MNIST CNN Visualizer & Training Pipeline

Python/PyTorch project for training and inspecting a convolutional neural network on MNIST digits.

## Features

- Loads MNIST train/test splits with Torchvision and PyTorch `DataLoader`s.
- Defines a small CNN with convolution, ReLU, max pooling, and fully connected layers.
- Implements reusable training and evaluation loops with cross-entropy loss and Adam optimization.
- Includes probing utilities to inspect kernels, feature maps, activation changes, logits, and prediction probabilities.

## Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib
- tqdm

## Project Structure

- `ForwardStep.py` defines the CNN architecture.
- `LoadingData.py` loads MNIST data and creates batches.
- `Training.py` contains training and evaluation loops.
- `Probe.py` visualizes intermediate model behavior.
- `main.py` runs an exploratory probe workflow.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

MNIST data is downloaded locally by Torchvision and is intentionally excluded from the repository.

