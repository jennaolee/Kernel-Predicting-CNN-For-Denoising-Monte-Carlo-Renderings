"""
Kernel Predicting Convolutional Network (KPCN)
Replication of:
Bako et al., Kernel-Predicting Convolutional Networks for Denoising Monte Carlo Renderings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Model Definition
# ============================================================

class KPCN(nn.Module):
    def __init__(self, in_channels=3, kernel_size=21, hidden_features=100):
        """
        Args:
            in_channels: number of input channels (RGB = 3)
            kernel_size: spatial size of predicted kernel (default 21x21)
            hidden_features: number of feature maps per conv layer (paper uses 100)
        """
        super(KPCN, self).__init__()

        self.kernel_size = kernel_size
        k = kernel_size

        layers = []

        # First convolution layer
        layers.append(nn.Conv2d(in_channels, hidden_features, kernel_size=5, padding=2))
        layers.append(nn.ReLU(inplace=True))

        # Remaining 7 hidden layers (total = 8 conv layers)
        for _ in range(7):
            layers.append(nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding=2))
            layers.append(nn.ReLU(inplace=True))

        self.feature_extractor = nn.Sequential(*layers)

        # Final layer predicts per-pixel kernel weights (k*k channels)
        self.kernel_predictor = nn.Conv2d(hidden_features, k * k, kernel_size=1)

    def forward(self, noisy_image):
        """
        Args:
            noisy_image: (B, 3, H, W)

        Returns:
            denoised_image: (B, 3, H, W)
        """

        B, C, H, W = noisy_image.shape
        k = self.kernel_size

        # Extract features
        features = self.feature_extractor(noisy_image)

        # Predict per-pixel kernels
        kernels = self.kernel_predictor(features)  # (B, k*k, H, W)

        # Softmax normalization across kernel weights
        kernels = F.softmax(kernels, dim=1)

        # Extract k×k patches from noisy image
        patches = F.unfold(noisy_image, kernel_size=k, padding=k // 2)
        patches = patches.view(B, C, k * k, H, W)

        # Apply predicted kernels
        kernels = kernels.unsqueeze(1)  # (B, 1, k*k, H, W)
        denoised = (kernels * patches).sum(dim=2)

        return denoised


# ============================================================
# Training Utilities
# ============================================================

def create_model(device="cuda"):
    model = KPCN(in_channels=3, kernel_size=21, hidden_features=100)
    return model.to(device)


def create_optimizer(model, lr=1e-5):
    return torch.optim.Adam(model.parameters(), lr=lr)


def create_loss():
    # Paper reports L1 performs best
    return nn.L1Loss()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Example Training Step (for debugging/demo)
# ============================================================

def train_step(model, optimizer, loss_fn, noisy, target):
    model.train()
    optimizer.zero_grad()

    output = model(noisy)
    loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()

    return loss.item()


def validation_step(model, loss_fn, noisy, target):
    model.eval()
    with torch.no_grad():
        output = model(noisy)
        loss = loss_fn(output, target)
    return loss.item()


# ============================================================
# Minimal Test Run
# ============================================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(device)
    optimizer = create_optimizer(model)
    loss_fn = create_loss()

    print("Model created")
    print("Total trainable parameters:", count_parameters(model))

    # Dummy data test (for sanity check)
    noisy = torch.randn(2, 3, 128, 128).to(device)
    target = torch.randn(2, 3, 128, 128).to(device)

    loss = train_step(model, optimizer, loss_fn, noisy, target)
    print("Training loss:", loss)

    val_loss = validation_step(model, loss_fn, noisy, target)
    print("Validation loss:", val_loss)