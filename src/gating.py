"""Train the routing gating network on winner labels from the MCL experts."""

import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import ScoreNet, GatingNet
from src.diffusion import sample_sigma_train, add_noise
from src.utils import get_mnist_loaders, set_seed
from src.sample import load_mcl


@torch.no_grad()
def collect_winners(experts, K, loader, sigma_min, sigma_max, device, num_batches=None):
    """Collect winner labels predicted by expert reconstruction loss.

    Args:
        experts: Sequence of trained expert models.
        K: Number of experts.
        loader: Data loader that yields training batches.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        device: Torch device used for computation.
        num_batches: Optional limit on processed batches.

    Returns:
        A tuple `(x_t, sigma, winners)` stored on CPU tensors.
    """
    x_ts, sigmas, winners_all = [], [], []
    for i, (images, _) in enumerate(tqdm(loader, desc="Collecting winners")):
        if num_batches is not None and i >= num_batches:
            break
        x_0 = images.to(device)
        B = x_0.shape[0]
        sigma = sample_sigma_train(B, sigma_min, sigma_max, device)
        x_t, eps = add_noise(x_0, sigma)

        losses = []
        for k in range(K):
            eps_pred = experts[k](x_t, sigma)
            loss_k = (eps_pred - eps).pow(2).sum(dim=(1, 2, 3))
            losses.append(loss_k)
        losses = torch.stack(losses, dim=1)
        w = losses.argmin(dim=1)

        x_ts.append(x_t.cpu())
        sigmas.append(sigma.cpu())
        winners_all.append(w.cpu())

    return torch.cat(x_ts), torch.cat(sigmas), torch.cat(winners_all)


def train_gating(args):
    """Train a gating network to predict expert winners.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device)
    set_seed(args.seed)

    experts, K, mcl_args = load_mcl(args.mcl_ckpt, device)
    sigma_min, sigma_max = mcl_args["sigma_min"], mcl_args["sigma_max"]

    train_loader, _ = get_mnist_loaders(args.batch_size)

    print(f"Collecting winner labels from {K} experts ...")
    x_t, sigma, winners = collect_winners(
        experts,
        K,
        train_loader,
        sigma_min,
        sigma_max,
        device,
        num_batches=args.collect_batches,
    )
    dataset = torch.utils.data.TensorDataset(x_t, sigma, winners)
    gating_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    gating_net = GatingNet(K=K).to(device)
    optimizer = torch.optim.Adam(gating_net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(
        f"Training gating network  |  params: "
        f"{sum(p.numel() for p in gating_net.parameters()):,}"
    )

    for epoch in range(1, args.epochs + 1):
        gating_net.train()
        total_loss, correct, total = 0.0, 0, 0

        for xt_b, sig_b, w_b in tqdm(gating_loader, desc=f"Gating epoch {epoch}"):
            xt_b, sig_b, w_b = xt_b.to(device), sig_b.to(device), w_b.to(device)
            logits = gating_net(xt_b, sig_b)
            loss = criterion(logits, w_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xt_b.shape[0]
            correct += (logits.argmax(1) == w_b).sum().item()
            total += xt_b.shape[0]

        acc = correct / total
        print(f"  -> loss: {total_loss / total:.4f}  acc: {acc:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"gating_K{K}.pt")
    torch.save(
        {"gating_net": gating_net.state_dict(), "K": K, "args": vars(args)}, path
    )
    print(f"Gating network saved -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mcl_ckpt", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--collect_batches",
        type=int,
        default=100,
        help="Number of data batches used to collect winner labels",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="checkpoints")
    args = p.parse_args()
    train_gating(args)
