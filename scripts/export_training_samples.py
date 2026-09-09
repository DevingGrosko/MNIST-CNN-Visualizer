"""Export a small, genuine MNIST training stream for the in-browser movie.

No retraining or checkpoint changes. Select the first 20 occurrences per digit.
"""
import json
from pathlib import Path
import numpy as np
from torchvision.datasets import MNIST

root = Path(__file__).resolve().parents[1]
dataset = MNIST(root / 'data', train=True, download=False)
counts = [0] * 10
samples = []
for index, (image, label) in enumerate(dataset):
    if counts[label] >= 20:
        continue
    samples.append({'id': index, 'label': label, 'pixels': np.asarray(image).reshape(-1).tolist()})
    counts[label] += 1
    if min(counts) == 20:
        break
result = {'split': 'train', 'selection': 'first 20 examples per digit in dataset order', 'samples': samples}
(root / 'web/assets/training-samples.json').write_text(json.dumps(result, separators=(',', ':')))
print(f'Exported {len(samples)} MNIST training examples.')
