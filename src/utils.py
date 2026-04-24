"""Shared helpers: data loading, image grids, seeding, EMA."""

import os
import copy
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_mnist_loaders(batch_size=128, data_dir="./data", num_workers=2):
    """Create MNIST training and test data loaders.

    Args:
        batch_size: Number of samples per batch.
        data_dir: Directory used to cache the dataset.
        num_workers: Number of worker processes for loading.

    Returns:
        A tuple `(train_loader, test_loader)` with images in [-1, 1].
    """
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    train_ds = torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, test_loader


def save_image_grid(images, path, nrow=8, value_range=(-1, 1)):
    """Save a tensor batch as an image grid.

    Args:
        images: Image tensor batch.
        path: Output path for the saved image.
        nrow: Number of images per row.
        value_range: Value range used for clamping and normalization.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    grid = torchvision.utils.make_grid(
        images.clamp(*value_range), nrow=nrow, normalize=True, value_range=value_range
    )
    plt.figure(figsize=(nrow, max(1, len(images) // nrow)))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def set_seed(seed):
    """Set random seeds for reproducible torch runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999):
        """Create an EMA shadow copy of a model.

        Args:
            model: Model to track.
            decay: Exponential decay factor.
        """
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters from the current model weights."""
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        """Run forward pass with EMA parameters."""
        return self.shadow(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def state_dict(self):
        """Return the EMA model state dict."""
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        """Load EMA model parameters from a state dict."""
        self.shadow.load_state_dict(sd)
