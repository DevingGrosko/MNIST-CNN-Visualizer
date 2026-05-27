import torch
from LoadingData import LoadingData
from ForwardStep import Forward
from Probe import Probe

torch.manual_seed(42)
# -----------------------
# Device
# -----------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------
# Data
# -----------------------
data_module = LoadingData(batch_size=32)

# FIXED probe image (from TEST set, never trained on)
probe_image, probe_label = data_module.test_data[0]

print("Probe label:", probe_label)
print("Probe image shape:", probe_image.shape)

# -----------------------
# Model (FRESH, UNTRAINED)
# -----------------------
model = Forward().to(device)

# IMPORTANT: do NOT create Training()
# IMPORTANT: do NOT call fit()
# This model has random weights

# ============================================================
print("\n==============================")
print("TRAINING — 10 BATCHES")
print("==============================")

model.train()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Gets the data in batches and prepares to be iterated through
train_iter = iter(data_module.train_dataloader)

for i in range(0):
    train_images, train_labels = next(train_iter)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)

    optimizer.zero_grad()
    train_logits = model(train_images)
    loss = criterion(train_logits, train_labels)
    loss.backward()
    optimizer.step()

    print(f"Batch {i+1} loss:", loss.item())

# -----------------------
# Probe
# -----------------------
probe = Probe(model, device)

# -----------------------
# Run probe pass (NO TRAINING)
# -----------------------
outputs = probe.run(probe_image)

probs = outputs["probs"][0]  # shape: [10]

print("\nClass probabilities after 50 batches:")
for i, p in enumerate(probs):
    print(f"Class {i}: {p.item():.4f}")

# ---- Conv1 mechanics ----
probe.show_relu_and_pool_effects(
    outputs["conv1_pre_relu"],
    outputs["conv1_post_relu"],
    outputs["conv1_post_pool"],
    kernel_index=0,
    layer_name="Conv1"
)

probe.show_relu_and_pool_effects(
    outputs["conv1_pre_relu"],
    outputs["conv1_post_relu"],
    outputs["conv1_post_pool"],
    kernel_index=2,
    layer_name="Conv1"
)


# ---- Conv2 inputs ----
probe.show_conv2_inputs(outputs["conv1_post_pool"])

# ---- Conv2 channel combination ----
probe.show_conv2_channel_contributions(
    outputs["conv1_post_pool"],
    kernel_index=0
)
probe.show_conv2_channel_contributions(
    outputs["conv1_post_pool"],
    kernel_index=1
)

# Conv2 full inspection
probe.show_all_conv2_maps(
    outputs["conv2_pre_relu"],
    outputs["conv2_post_relu"],
    outputs["conv2_post_pool"]
)

# ---- Final output ----
probe.show_output(outputs["logits"], outputs["probs"])


