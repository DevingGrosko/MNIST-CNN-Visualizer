# CNN, Explained — MNIST CNN Visualizer

An interactive teaching website built on the original Python/PyTorch MNIST project.

**[Open the live visualizer](https://devinggrosko.github.io/MNIST-CNN-Visualizer/)**

Choose a real MNIST test example, draw a digit, or load a local image. Follow its actual activations through the original network and inspect the math at every stage. All inference and sandbox learning happen locally in the browser; images are not uploaded.

## The interactive lab

- **12 lessons:** input normalization, Conv1, ReLU, max pooling, Conv2, ReLU, max pooling, flattening, dense/ReLU, logits, softmax, and learning.
- **Convolution inspector:** animated scanning, every output channel, keyboard-selectable cells, 3×3 input/weight/product matrices, zero padding, bias, and all eight Conv2 input-channel contributions.
- **Activation and pooling inspection:** before/after maps, exact selected values, and max-pool winner routing.
- **Full classifier inspection:** all 784 flattened activations, all 128 hidden neurons, and every connection for a selected hidden/output neuron.
- **Real predictions:** 50 held-out MNIST examples (first five per label), freehand drawing, local image loading, pixel editing, and all ten probability scores.
- **Learning:** switch between initial and trained checkpoints; choose a label and learning rate; run real full-network backpropagation and one SGD update; inspect loss, gradients, and a changed weight. Updates are temporary. Reset restores the checkpoint.
- **Field guide:** detailed explanations of channels, receptive fields, weight sharing, training, numerical stability, and model limitations.

The small network diagrams show a subset of connections for legibility. The inspector tables expose all connections for the selected neuron. Each heatmap rescales its color intensity independently; use numeric inspectors for comparisons. Matrices show rounded values while calculations retain full precision.

## Original architecture

The web implementation follows `ForwardStep.Forward` exactly:

```text
1×28×28 grayscale image, values in [0, 1]
Conv2d(1, 8, 3, stride=1, padding=1) → ReLU → MaxPool2d(2)
Conv2d(8, 16, 3, stride=1, padding=1) → ReLU → MaxPool2d(2)
Flatten: 16×7×7 → 784
Linear(784, 128) → ReLU → Linear(128, 10)
Softmax for display; raw logits for cross-entropy training
```

**103,018 trainable parameters.** Convolutions use PyTorch's cross-correlation convention, and tensors use channel-first ordering.

The original Python architecture, data loading, training loops, and Matplotlib probing code are preserved. The website extends the inspection ideas in `Probe.py` without requiring Python to serve the site.

## Run the website

Node.js 20 or newer is sufficient. There are no runtime or build dependencies to install.

```bash
npm run dev
# http://127.0.0.1:4173
npm test
npm run build
```

Equivalent commands: `node scripts/serve.mjs`, `node --test tests/*.test.mjs`, and `node scripts/build.mjs`.

`web/` contains the website source. `dist/` is generated static output suitable for GitHub Pages. Asset URLs are relative, so the repository subpath works without special configuration. The GitHub Actions workflow validates the code and publishes `main` through GitHub Pages. Pull requests run validation without deploying.

## Model provenance and reproduction

The committed checkpoint was trained from seed 42 with the original architecture, Adam at learning rate 0.001, batch size 128, and three epochs over all 60,000 MNIST training examples. Inputs use `ToTensor()` (pixel / 255), with no mean/std normalization. Evaluation covers all 10,000 MNIST test examples.

| Checkpoint | Test accuracy | Test cross-entropy |
| --- | ---: | ---: |
| Initial random weights | 7.91% | 2.30344 |
| Epoch 1 | 95.93% | 0.12443 |
| Epoch 2 | 97.76% | 0.07162 |
| Epoch 3 (exported) | 97.92% | 0.05994 |

The stored metadata includes these measurements. Initial and trained weights are exported as transparent JSON, rounded to eight decimal places. JavaScript uses double-precision arithmetic; PyTorch reference tensors use float32. Hardware/library differences can affect a retraining run despite the fixed seed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_export.py --epochs 3
npm test
```

The exporter imports the existing `Forward` class, downloads MNIST, trains, evaluates, and regenerates `web/assets/model.json`, `web/assets/samples.json`, and `tests/pytorch-reference.json`. No test images are used in that training run. The bundled samples are selected by test-set order, not by whether the model predicts them correctly.

The browser learning sandbox uses single-image **SGD** so its update equation is easy to inspect; the exported checkpoint uses **Adam**. Sandbox updates to a test example are educational and invalidate the original accuracy measurement for that modified model. The UI labels modified weights accordingly. Changing the image or target restarts the displayed loss trace; modified weights persist until Reset or a checkpoint switch.

This model recognizes handwritten digits only. Drawings and uploaded photos can differ from MNIST. A high softmax probability is not a guarantee or an out-of-distribution detector. Blank inputs are explicitly identified.

## Validation

`npm test` checks all intermediate forward tensors against independently computed PyTorch outputs for three test samples under both checkpoints. It also checks tensor dimensions, convolution padding and channel summation, pooling ties, extreme-logit numerical stability, parameter counts, and real loss reduction after an SGD step.

The full backward pass is checked by central finite differences on weights and biases from every trainable tensor and selected input pixels. That test adds tiny deterministic perturbations to avoid ambiguous max-pool ties and uses a sufficiently small step to stay within the same activation region.

The optional WebMCP tools `inspect_cnn` and `explore_cnn_sample` are feature-detected. They are not required for the lab. A supporting WebMCP runtime was not available during development, so their browser integration has not been verified. Interactive browser/visual QA has not been performed; validation covers numerical behavior, syntax, and the static build.

## Repository map

- `ForwardStep.py` — original model architecture.
- `LoadingData.py` — original MNIST data loaders and image inspection.
- `Training.py` — original training/evaluation loops with Adam.
- `Probe.py` — original Matplotlib inspection utilities.
- `main.py` — original exploratory probe script (its current loop trains zero batches).
- `web/engine.js` — transparent inference, loss, backward pass, and SGD.
- `web/lessons.js` — numerical inspectors and teaching content.
- `web/app.js` — live input, drawing, state, and network rendering.
- `web/render.js`, `web/guide.js`, `web/styles.css` — graphics, field guide, and responsive presentation.
- `scripts/train_export.py` — reproducible training and browser export.
- `.github/workflows/pages.yml` — tests, static build, and GitHub Pages publication.

## References

- [Stanford CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [MNIST dataset](https://yann.lecun.org/exdb/mnist/)
