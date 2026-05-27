import torch
from torch import nn
from tqdm import tqdm


class Training:
    def __init__(self, model, train_dataloader, test_dataloader, device, lr=0.001):
        self.device = device

        # Model
        self.model = model.to(device)

        # Data
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # Loss & optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self):
        self.model.train()
        train_loss = 0.0

        # Loads one batch of data (32 images) at a time, up to 60,000 images
        # So loops 1800 ish times (60,000 / 32)
        for X, y in self.train_dataloader:
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            logits = self.model(X)
            loss = self.loss_fn(logits, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(self.train_dataloader)

    def test_step(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                test_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()

        accuracy = correct / len(self.test_dataloader.dataset)
        return test_loss / len(self.test_dataloader), accuracy

    def fit(self, epochs):
        for epoch in tqdm(range(epochs)):
            train_loss = self.train_step()
            test_loss, test_acc = self.test_step()

            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )