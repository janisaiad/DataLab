"""FID, Precision and Recall computed in a 128-d MNIST-classifier feature space."""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from scipy import linalg

from src.utils import get_mnist_loaders, set_seed
from src.sample import load_baseline, load_mcl, generate_baseline, generate_mcl


class MNISTClassifier(nn.Module):
    """LeNet-style classifier; 128-d features are taken from `feat`."""

    def __init__(self):
        """Initialise the classifier."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.feat = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        """Run classification and return logits plus features."""
        h = self.conv(x).flatten(1)
        f = self.feat(h)
        return self.head(f), f


def train_classifier(device, epochs=5, batch_size=256):
    """Train or load a cached MNIST feature extractor.

    Args:
        device: Torch device used for training and inference.
        epochs: Number of epochs if training is needed.
        batch_size: Batch size used for training.

    Returns:
        A trained classifier in eval mode.
    """
    cache_path = "checkpoints/shared/mnist_classifier.pt"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        clf = MNISTClassifier().to(device)
        clf.load_state_dict(torch.load(cache_path, map_location=device, weights_only=True))
        clf.eval()
        return clf

    train_loader, test_loader = get_mnist_loaders(batch_size)
    clf = MNISTClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        clf.train()
        for imgs, labels in train_loader:
            logits, _ = clf(imgs.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            opt.zero_grad(); loss.backward(); opt.step()

        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                logits, _ = clf(imgs.to(device))
                correct += (logits.argmax(1) == labels.to(device)).sum().item()
                total += labels.shape[0]
        print(f"  Classifier epoch {ep}: acc={correct/total:.4f}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(clf.state_dict(), cache_path)
    clf.eval()
    return clf


@torch.no_grad()
def extract_features(clf, images, batch_size=512, device="cpu"):
    """Extract classifier features from image tensors.

    Args:
        clf: Feature extractor model.
        images: Input images.
        batch_size: Batch size for feature extraction.
        device: Device used for inference.

    Returns:
        A NumPy array of extracted features.
    """
    clf.eval()
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    feats = []
    for (batch,) in loader:
        _, f = clf(batch.to(device))
        feats.append(f.cpu())
    return torch.cat(feats).numpy()


def compute_fid(feats_real, feats_gen):
    """Compute Frechet distance between two feature sets.

    Args:
        feats_real: Features from real images.
        feats_gen: Features from generated images.

    Returns:
        The FID value as a float.
    """
    mu_r, mu_g = feats_real.mean(0), feats_gen.mean(0)
    d = feats_real.shape[1]
    eps = 1e-6
    sigma_r = np.cov(feats_real, rowvar=False) + np.eye(d) * eps
    sigma_g = np.cov(feats_gen, rowvar=False) + np.eye(d) * eps

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def _knn_distance(feats, k=5):
    """Return the k-th nearest-neighbor distance for each sample."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(feats)
    dists, _ = nn.kneighbors(feats)
    return dists[:, -1]


def compute_precision_recall(feats_real, feats_gen, k=5):
    """Compute manifold-based precision and recall.

    Args:
        feats_real: Features from real images.
        feats_gen: Features from generated images.
        k: Number of nearest neighbors.

    Returns:
        A tuple `(precision, recall)`.
    """
    from sklearn.neighbors import NearestNeighbors

    radii_real = _knn_distance(feats_real, k)

    nn_real = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(feats_real)
    dists_gen_to_real, idxs = nn_real.kneighbors(feats_gen)
    dists_gen_to_real = dists_gen_to_real[:, 0]
    precision = float((dists_gen_to_real <= radii_real[idxs[:, 0]]).mean())

    radii_gen = _knn_distance(feats_gen, k)

    nn_gen = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(feats_gen)
    dists_real_to_gen, idxs2 = nn_gen.kneighbors(feats_real)
    dists_real_to_gen = dists_real_to_gen[:, 0]
    recall = float((dists_real_to_gen <= radii_gen[idxs2[:, 0]]).mean())

    return precision, recall


def evaluate(real_images, gen_images, device="cpu", k=5):
    """Evaluate generated images against real images.

    Args:
        real_images: Real reference images.
        gen_images: Generated images to evaluate.
        device: Device used for feature extraction.
        k: Number of nearest neighbors for precision/recall.

    Returns:
        A dictionary with `fid`, `precision`, and `recall`.
    """
    clf = train_classifier(device)
    feats_real = extract_features(clf, real_images, device=device)
    feats_gen = extract_features(clf, gen_images, device=device)

    fid = compute_fid(feats_real, feats_gen)
    prec, rec = compute_precision_recall(feats_real, feats_gen, k=k)
    return {"fid": fid, "precision": prec, "recall": rec}


def main():
    """Entry point for standalone sample evaluation."""
    p = argparse.ArgumentParser(description="Evaluate generated samples")
    p.add_argument("--samples_pt", default=None,
                   help="Path to .pt file with generated images (B,1,28,28) in [-1,1]")
    p.add_argument("--checkpoint", default=None, help="Generate on-the-fly from ckpt")
    p.add_argument("--mode", choices=["baseline", "mcl"], default="baseline")
    p.add_argument("--strategy", default="single_expert")
    p.add_argument("--expert_id", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--solver", default="euler")
    p.add_argument("--k", type=int, default=5, help="k for precision/recall")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", default=None, help="Optional path to save metrics as JSON")
    args = p.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    if args.samples_pt:
        gen_images = torch.load(args.samples_pt, weights_only=True)
    elif args.checkpoint:
        print("Generating samples ...")
        if args.mode == "baseline":
            model, a = load_baseline(args.checkpoint, device)
            gen_images = generate_baseline(
                model, args.num_samples, args.num_steps, device,
                a["sigma_min"], a["sigma_max"], args.solver, args.seed,
            )
        else:
            experts, K, a = load_mcl(args.checkpoint, device)
            gen_images = generate_mcl(
                experts, K, strategy=args.strategy, expert_id=args.expert_id,
                num_samples=args.num_samples, num_steps=args.num_steps,
                device=device, sigma_min=a["sigma_min"], sigma_max=a["sigma_max"],
                solver=args.solver, seed=args.seed,
            )
        gen_images = gen_images.cpu()
    else:
        raise ValueError("Provide --samples_pt or --checkpoint")

    _, test_loader = get_mnist_loaders(batch_size=512)
    real_images = torch.cat([x for x, _ in test_loader])[:args.num_samples]

    metrics = evaluate(real_images, gen_images, device=str(device), k=args.k)
    print(f"\n{'='*40}")
    print(f"  FID:       {metrics['fid']:.2f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"{'='*40}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved -> {args.out_json}")


if __name__ == "__main__":
    main()
