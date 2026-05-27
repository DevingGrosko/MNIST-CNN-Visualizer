import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F


class Probe:
    """
    Runs a fixed image through the model and records
    intermediate weights, activations, and outputs.
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        # Explicit layer references (for clarity + control)
        self.conv1 = self.model.conv_block[0]
        self.relu1 = self.model.conv_block[1]
        self.pool1 = self.model.conv_block[2]

        self.conv2 = self.model.conv_block[3]
        self.relu2 = self.model.conv_block[4]
        self.pool2 = self.model.conv_block[5]

    def _show(self):
        plt.show()  # blocks until you close the window

    # -----------------------
    # Image inspection
    # -----------------------
    def show_image_tensor(self, image, label):
        image = image.detach().cpu()

        print("Image tensor:")
        print(image)
        print("Shape:", image.shape)
        print("Min / Max:", image.min().item(), image.max().item())

        plt.figure()
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        self._show()

    # -----------------------
    # Forward probe
    # -----------------------
    def run(self, image):
        """
        Runs a single image through the network step by step.
        """
        self.model.eval()
        image = image.unsqueeze(0).to(self.device)  # [1, 1, 28, 28]

        with torch.no_grad():
            # Conv1
            z1 = self.conv1(image)
            a1 = self.relu1(z1)
            p1 = self.pool1(a1)

            # Conv2
            z2 = self.conv2(p1)
            a2 = self.relu2(z2)
            p2 = self.pool2(a2)

            # Classifier
            logits = self.model.classifier(p2)
            probs = F.softmax(logits, dim=1)

        return {
            "conv1_pre_relu": z1.detach().cpu(),
            "conv1_post_relu": a1.detach().cpu(),
            "conv1_post_pool": p1.detach().cpu(),
            "conv2_pre_relu": z2.detach().cpu(),
            "conv2_post_relu": a2.detach().cpu(),
            "conv2_post_pool": p2.detach().cpu(),
            "logits": logits.detach().cpu(),
            "probs": probs.detach().cpu(),
        }

    # -----------------------
    # Conv1 inspection
    # -----------------------
    def show_conv1_kernel(self, kernel_index):
        kernel = self.conv1.weight[kernel_index, 0].detach().cpu()

        print(f"Conv1 Kernel {kernel_index}:")
        print(kernel)

        plt.figure()
        plt.imshow(kernel, cmap="gray")
        plt.title(f"Conv1 Kernel {kernel_index}")
        plt.colorbar()
        self._show()

    def show_conv1_activation(self, activations, kernel_index):
        fmap = activations[0, kernel_index]

        plt.figure()
        plt.imshow(fmap, cmap="gray")
        plt.title(f"Conv1 Activation – Kernel {kernel_index}")
        plt.colorbar()
        self._show()

    # -----------------------
    # Conv2 inspection
    # -----------------------
    def show_conv2_kernel_stack(self, kernel_index):
        weights = self.conv2.weight[kernel_index].detach().cpu()  # [8, 3, 3]

        plt.figure(figsize=(8, 4))
        for i in range(weights.shape[0]):
            plt.subplot(2, 4, i + 1)
            plt.imshow(weights[i], cmap="gray")
            plt.title(f"Slice {i}")
            plt.axis("off")

        plt.suptitle(f"Conv2 Kernel {kernel_index} (8 input slices)")
        self._show()

    def show_conv2_activation(self, activations, kernel_index):
        fmap = activations[0, kernel_index]

        plt.figure()
        plt.imshow(fmap, cmap="gray")
        plt.title(f"Conv2 Activation – Kernel {kernel_index}")
        plt.colorbar()
        self._show()

    # -----------------------
    # Output inspection
    # -----------------------
    def show_output(self, logits, probs):
        print("\nLogits:")
        print(logits)

        print("\nProbabilities:")
        print(probs)

        pred = probs.argmax(dim=1).item()
        conf = probs.max(dim=1).values.item()

        print(f"\nPredicted class: {pred}")
        print(f"Confidence: {conf:.4f}")

    def show_relu_and_pool_effects(
            self,
            pre_relu,
            post_relu,
            post_pool,
            kernel_index,
            layer_name="Conv"
    ):
        """
        Shows effects of Conv → ReLU → MaxPool
        Works for ANY convolutional layer
        """

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(pre_relu[0, kernel_index], cmap="gray")
        axes[0].set_title("Pre-ReLU")
        axes[0].axis("off")

        axes[1].imshow(post_relu[0, kernel_index], cmap="gray")
        axes[1].set_title("Post-ReLU")
        axes[1].axis("off")

        axes[2].imshow(post_pool[0, kernel_index], cmap="gray")
        axes[2].set_title("Post-MaxPool")
        axes[2].axis("off")

        plt.suptitle(f"{layer_name} Kernel {kernel_index} — ReLU & Pool Effects")
        self._show()

    def show_conv2_inputs(self, conv1_pooled):
        """
        Shows the 8 Conv1 feature maps that feed into Conv2
        """
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))

        for i in range(8):
            ax = axes[i // 4, i % 4]
            ax.imshow(conv1_pooled[0, i], cmap="gray")
            ax.set_title(f"Conv1 fmap {i}")
            ax.axis("off")

        plt.suptitle("8 Conv1 Feature Maps → Conv2 Input")
        self._show()

    def show_conv2_channel_contributions(self, conv1_pooled, kernel_index):
        """
        Shows how ONE Conv2 kernel combines 8 Conv1 feature maps
        """
        weights = self.conv2.weight[kernel_index].detach().cpu()  # [8, 3, 3]

        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        summed = torch.zeros_like(conv1_pooled[0, 0])

        for i in range(8):
            # single-channel convolution
            response = F.conv2d(
                conv1_pooled[:, i:i + 1],
                weights[i:i + 1].unsqueeze(1),
                padding=1
            )[0, 0]

            summed += response

            ax = axes[i // 4, i % 4]
            ax.imshow(response, cmap="gray")
            ax.set_title(f"Slice {i} response")
            ax.axis("off")

        axes[2, 1].imshow(summed, cmap="gray")
        axes[2, 1].set_title("Summed Conv2 Output")
        axes[2, 1].axis("off")

        # hide unused slots
        axes[2, 0].axis("off")
        axes[2, 2].axis("off")
        axes[2, 3].axis("off")

        plt.suptitle(f"Conv2 Kernel {kernel_index} — Channel Contributions")
        self._show()

    def show_all_conv2_maps(self, pre_relu, post_relu, post_pool):
        """
        Shows all 16 Conv2 feature maps:
        - pre-ReLU
        - post-ReLU
        - post-MaxPool
        """
        stages = [
            ("Conv2 Pre-ReLU", pre_relu),
            ("Conv2 Post-ReLU", post_relu),
            ("Conv2 Post-MaxPool", post_pool),
        ]

        for title, tensor in stages:
            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            for i in range(16):
                ax = axes[i // 4, i % 4]
                ax.imshow(tensor[0, i], cmap="gray")
                ax.set_title(f"Channel {i}")
                ax.axis("off")

            plt.suptitle(title)
            self._show()

    def show_logit_contribution(self, pooled_conv2, class_index):
        fc1 = self.model.classifier[1]  # Linear(784 → 128)
        fc2 = self.model.classifier[3]  # Linear(128 → 10)

        flat = pooled_conv2.view(1, -1).to(self.device)

        with torch.no_grad():
            hidden = F.relu(fc1(flat))
            weights = fc2.weight[class_index]
            contribution = hidden * weights

        plt.figure(figsize=(8, 3))
        plt.plot(contribution.detach().cpu().numpy())
        plt.title(f"Contribution to logit {class_index}")
        self._show()

    def show_flattened_vector(self, pooled_conv2):
        flat = pooled_conv2.view(-1)

        print("\nFlattened vector info:")
        print("Shape:", flat.shape)
        print("First 40 values:", flat[:40])
        print("Min / Max / Mean:",
              flat.min().item(),
              flat.max().item(),
              flat.mean().item())

        plt.figure(figsize=(10, 3))
        plt.plot(flat.detach().cpu().numpy())
        plt.title("Flattened Conv2 Output (784 values)")
        self._show()



