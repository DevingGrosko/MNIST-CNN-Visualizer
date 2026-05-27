import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LoadingData:
    """
    Handles:
    - loading MNIST train/test datasets
    - creating DataLoaders
    - inspecting and visualizing samples
    """
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

        # Transform converts images to tensors with values in [0, 1]
        self.transform = transforms.ToTensor()

        # Training dataset
        self.train_data = MNIST(
            root="./data",
            train=True,          # load training split
            download=True,
            transform=self.transform
        )

        # Test dataset
        self.test_data = MNIST(
            root="./data",
            train=False,         # load test split (different files)
            download=True,
            transform=self.transform
        )

        # DataLoader for training data
        self.train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True         # shuffle each epoch for better training
        )

        # DataLoader for test data
        self.test_dataloader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False        # no need to shuffle during evaluation
        )

    def get_image(self, index: int):
        """
        Returns a single (image, label) pair from the training dataset.
        """
        image, label = self.train_data[index]
        return image, label

    def show_image(self, image: torch.Tensor, label: int):
        """
        Visualizes a single MNIST image tensor.
        """
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()

    def load_premade_batch(self, dataloader, index):
        for i, (images, labels) in enumerate(dataloader):
            if i == index:
                plt.figure(figsize=(8, 8))
                for j in range(len(images)):
                    plt.subplot(4, 8, j + 1)
                    plt.imshow(images[j].squeeze(), cmap="gray")
                    plt.title(str(labels[j].item()))
                    plt.axis("off")
                plt.tight_layout()
                plt.show()
                return
        raise IndexError("Batch index out of range")

    def get_custom_batch(self,dataset, indices):
        """
        Returns a set of custom images at specific indices of my choosing
        """
        images = []
        labels = []
        for idx in indices:
            img, lbl = dataset[idx]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    def getShapeOfBatch(self,train_dataloader):
        """Prints shape of a batch from the training data"""
        images, labels = next(iter(train_dataloader))

        print("Images tensor:")
        print("  shape:", images.shape)
        print("  dtype:", images.dtype)

        print("\nLabels tensor:")
        print("  shape:", labels.shape)
        print("  dtype:", labels.dtype)

    def batch_stats(self, dataloader):
        """
        Prints min, max, mean, std of a single batch
        """
        images, _ = next(iter(dataloader))

        print("Min pixel value:", images.min().item())
        print("Max pixel value:", images.max().item())
        print("Mean pixel value:", images.mean().item())
        print("Std pixel value:", images.std().item())

    def number_Of_Batches_In_Epoch(self, dataloader):
        print("Number of batches:", len(dataloader))
        print("Batch size:", dataloader.batch_size)

