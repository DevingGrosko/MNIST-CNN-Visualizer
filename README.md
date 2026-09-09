# CNN, Explained — MNIST CNN Visualizer

An interactive textbook built on the original Python/PyTorch MNIST project, with separate pages for the network overview, twelve lessons, and field notes.

**[Open the live visualizer](https://devinggrosko.github.io/MNIST-CNN-Visualizer/)**

Choose a real MNIST test example, draw a digit, or load a local image. Follow its actual activations through the original network and inspect the math at every stage. All inference and sandbox learning happen locally in the browser; images are not uploaded.

## The interactive lab

- **14 real pages:** an overview, twelve chapter URLs with previous/next navigation, and separate field notes. Internal navigation retains the current image and model, supports browser Back/Forward, and direct lesson links work on GitHub Pages. Reloading a document starts a fresh model session.
- **12 lessons:** input normalization, Conv1, ReLU, max pooling, Conv2, ReLU, max pooling, flattening, dense/ReLU, logits, softmax, and learning.
- **Worked convolution:** side-by-side input/filter/output diagrams, linked cell selections, a nine-term arithmetic replay, running totals, zero padding, bias, and an explicit eight-channel ledger for Conv2.
- **Activation and pooling inspection:** before/after maps, a live ReLU function plot with local derivative, exact selected values, and a diagram of max-pool gradient routing.
- **Worked classifier:** select any connection in a true-weight neuron diagram; compare signed contributions, including the aggregated remainder; see all 128 neurons and every connection in an open table. Four aligned softmax columns trace logits, shifting, exponentiation, and normalization.
- **Real predictions:** 50 held-out MNIST examples (first five per label), freehand drawing, local image loading, pixel editing, and all ten probability scores.
- **Training movie:** a twelve-scene forward/backward/update cycle with play, pause, single-scene stepping, and 0.5–4× speed. Start untrained, stop after one update, then play to ten or keep adding ten more. All gradients and SGD updates are real.
- **Matrix wall:** eight first-layer kernels with numbers, all 128 second-layer kernel slices, complete dense/output weight textures, every bias, gradient matrices, and all feature-map channels. A wide view keeps the four trainable layers together. Warm flashes highlight large changes in each panel.
- **Exact update inspection:** click any parameter to inspect old weights, derivatives, signed changes, and new weights in four aligned numeric matrices. Inspect every spatial contribution to a convolution derivative or the actual dense-layer gradient product. Compare the same parameter before training, after one update, and after ten.
- **Training playback:** repeat one image to make learning visible, or stream 200 genuine MNIST training examples. Review saved checkpoints without mutating the live model. Each update shows paired before/after loss for the same image. Checkpoints 0, 1, 10 and the latest 20 updates remain available; the next ten updates can be played indefinitely. Updates stay in memory until a reset or reload.
- **Field guide:** detailed explanations of channels, receptive fields, weight sharing, training, numerical stability, and model limitations.

The site uses the Botanical palette: a dark brown header, clay accents, lavender navigation, and burgundy highlights around light reading surfaces. Shared tokens drive the page, SVG diagrams, heatmaps, numeric matrices, and training flashes. Burgundy represents positive values, lavender represents negative values, clay highlights weight updates, and dark outlines indicate selection. The small network diagrams and six-connection dense diagrams show a subset of connections for legibility; all omitted contributions are still included in the calculated result. The inspector tables expose all connections for the selected neuron. The training movie holds each weight/activation scale fixed across the before/after pair; other heatmaps rescale independently. Use numeric inspectors for exact comparisons. Dense textures fit the whole matrix into the available space; click or use the flat-index selector to inspect any underlying cell. On narrow screens, the matrix wall scrolls horizontally. Matrices show rounded values while calculations retain full precision.

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

`web/` contains the shared website source. `web/pages.js` declares fourteen routes; `scripts/pages.mjs` generates route-specific documents with titles, descriptions, and initial page views. `dist/` contains fourteen real HTML documents suitable for GitHub Pages, including `conv1.html`, `dense.html`, `softmax.html`, and `learning.html`. The local development server uses the same page renderer. Production pages reference a content-addressed `_app/<revision>/` directory containing styles, all JavaScript modules, and model data. This prevents a fresh document from mixing old cached scripts with new styles. Asset URLs remain relative, so the repository subpath works without special configuration. The local server keeps unversioned URLs with `no-store` caching. The GitHub Actions workflow validates the code and publishes `main` through GitHub Pages. Pull requests run validation without deploying.

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
python scripts/export_training_samples.py
npm test
```

The exporter imports the existing `Forward` class, downloads MNIST, trains, evaluates, and regenerates `web/assets/model.json`, `web/assets/samples.json`, and `tests/pytorch-reference.json`. No test images are used in that training run. The bundled samples are selected by test-set order, not by whether the model predicts them correctly.

The browser learning sandbox uses single-image **SGD** so its update equation is easy to inspect; the exported checkpoint uses **Adam**. Sandbox updates to a test example are educational and invalidate the original accuracy measurement for that modified model. The UI labels modified weights accordingly. Changing the image, target, training mode, or rate begins a new trace from the current weights. The movie’s Reset and Watch buttons restore random weights. The model switch restores the exported initial/trained weights. Timeline checkpoint buttons are read-only views and do not change the live run. The optional training stream is exported separately from the train split (first 20 examples of each label in dataset order), and is not used to measure test accuracy.

This model recognizes handwritten digits only. Drawings and uploaded photos can differ from MNIST. A high softmax probability is not a guarantee or an out-of-distribution detector. Blank inputs are explicitly identified.

## Validation

`npm test` checks all intermediate forward tensors against independently computed PyTorch outputs for three test samples under both checkpoints. It also checks tensor dimensions, convolution padding and channel summation, pooling ties, extreme-logit numerical stability, parameter counts, and real loss reduction after an SGD step.

Worked-visualization tests check that every convolution term and channel sum reconciles, signed dense contribution groups include the full dot product, all four softmax columns match inference, and the illustrated chain rule agrees with full backpropagation. Route tests cover all fourteen documents and repository-subpath resolution. Training tests check the commit boundary, ten-step equivalence to direct SGD, loss reduction, immutable checkpoint review, bounded snapshot retention, the training stream, and all 1,224 convolution-weight derivative sums plus representative derivatives from every bias/dense tensor. The Conv2 mosaic is checked for an exact, bijective mapping to all 1,152 weights. Release-asset tests verify cache keys change with source content, all page assets use one release directory, and relative imports/data URLs resolve within it. There are 31 numerical, route, and release tests.

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
- `web/math-views.js`, `web/math-trace.js` — linked mathematical diagrams and tested arithmetic records.
- `web/training-session.js` — numerical training frames, checkpoints, gradient operands, and playback state.
- `web/training-player.js`, `web/training.css` — the matrix wall, movie controls, and exact numeric inspectors.
- `scripts/export_training_samples.py` — export the 200-image training stream.
- `web/pages.js`, `scripts/pages.mjs` — chapter URLs and static page generation.
- `web/app.js` — live input, drawing, state, and network rendering.
- `web/palette.js` — the Botanical colour tokens, canvas colours, and contrast-aware numeric labels.
- `web/render.js`, `web/guide.js`, `web/styles.css` — graphics, field guide, and responsive presentation.
- `scripts/train_export.py` — reproducible training and browser export.
- `.github/workflows/pages.yml` — tests, static build, and GitHub Pages publication.

## References

- [Stanford CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [MNIST dataset](https://yann.lecun.org/exdb/mnist/)
