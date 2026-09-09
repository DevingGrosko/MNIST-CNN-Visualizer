"""Train the original Forward architecture and export transparent browser weights.

Run from repository root: .venv/bin/python scripts/train_export.py --epochs 3
Training uses only the MNIST training split; evaluation uses all 10,000 test digits.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ForwardStep import Forward


def weights(model):
    state = model.state_dict()
    names = {'conv1Weight': 'conv_block.0.weight', 'conv1Bias': 'conv_block.0.bias',
             'conv2Weight': 'conv_block.3.weight', 'conv2Bias': 'conv_block.3.bias',
             'fc1Weight': 'classifier.1.weight', 'fc1Bias': 'classifier.1.bias',
             'fc2Weight': 'classifier.3.weight', 'fc2Bias': 'classifier.3.bias'}
    return {name: [round(v, 8) for v in state[key].detach().cpu().flatten().tolist()]
            for name, key in names.items()}


def run(epochs):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train = MNIST(ROOT / 'data', train=True, download=True, transform=ToTensor())
    test = MNIST(ROOT / 'data', train=False, download=True, transform=ToTensor())
    loader = DataLoader(train, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test, batch_size=256)
    model = Forward().to(device)
    initial = weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []

    def evaluate():
        model.eval()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                z = model(x)
                correct += int((z.argmax(1) == y).sum())
                loss += float(loss_fn(z, y)) * len(y)
        return {'accuracy': correct / len(test), 'testLoss': loss / len(test)}

    history.append({'epoch': 0, **evaluate()})
    print(json.dumps(history[-1]), flush=True)
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        loss_sum = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
        history.append({'epoch': epoch, 'trainLoss': loss_sum / len(train), **evaluate()})
        print(json.dumps({**history[-1], 'seconds': round(time.time() - start, 1)}), flush=True)

    model.eval()
    out = ROOT / 'web' / 'assets'
    out.mkdir(parents=True, exist_ok=True)
    meta = {'architecture': 'ForwardStep.Forward', 'seed': 42, 'epochs': epochs,
            'optimizer': 'Adam', 'learningRate': 0.001, 'batchSize': 128,
            'trainingExamples': len(train), 'testExamples': len(test),
            'parameters': sum(p.numel() for p in model.parameters()), 'history': history,
            'preprocessing': 'ToTensor: grayscale uint8 / 255; no mean/std normalization',
            'source': 'https://github.com/DevingGrosko/MNIST-CNN-Visualizer/blob/main/ForwardStep.py'}
    (out / 'model.json').write_text(json.dumps({'meta': meta, 'initial': initial, 'trained': weights(model)}, separators=(',', ':')))
    samples, counts = [], [0] * 10
    for index, (x, label) in enumerate(test):
        if counts[label] >= 5:
            continue
        samples.append({'id': index, 'label': label, 'pixels': (x.flatten()*255).round().int().tolist()})
        counts[label] += 1
        if min(counts) == 5:
            break
    (out / 'samples.json').write_text(json.dumps(samples, separators=(',', ':')))
    # Independent PyTorch results for numeric parity at every stage, for both snapshots.
    fixtures = []
    for snapshot, state in [('trained', model.state_dict()), ('initial', None)]:
        probe = Forward().to('cpu')
        if state is not None:
            probe.load_state_dict({k: v.cpu() for k, v in state.items()})
        else:
            keys = list(probe.state_dict())
            exported = list(initial.values())
            probe.load_state_dict({k: torch.tensor(v).reshape(probe.state_dict()[k].shape) for k, v in zip(keys, exported)})
        for sample in [samples[0], samples[1], samples[2]]:
            x = torch.tensor(sample['pixels'], dtype=torch.float32).view(1,1,28,28)/255
            stages = {'input': x}
            with torch.no_grad():
                for name, layer in zip(['conv1','relu1','pool1','conv2','relu2','pool2'], probe.conv_block):
                    x = layer(x); stages[name] = x
                x = x.flatten(1); stages['flatten'] = x
                x = probe.classifier[1](x); stages['dense'] = x
                x = torch.relu(x); stages['hidden'] = x
                x = probe.classifier[3](x); stages['logits'] = x
                stages['softmax'] = torch.softmax(x,dim=1)
            fixtures.append({'snapshot': snapshot, 'sampleId': sample['id'],
                             'stages': {k: v.flatten().tolist() for k,v in stages.items()}})
    (ROOT/'tests'/'pytorch-reference.json').write_text(json.dumps(fixtures, separators=(',', ':')))
    print(f'Exported model, {len(samples)} test samples, and {len(fixtures)} parity fixtures.', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    run(parser.parse_args().epochs)
