"""
Small CNN training demo with TensorBoard + optional Weights & Biases logging.

Usage:
  pip install torch torchvision tensorboard wandb
  # Set WANDB_API_KEY in env or run `wandb login` once
  python tb_cnn_demo.py               # runs with wandb logging enabled by default
  tensorboard --logdir runs --port 6006 --bind_all
"""

import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dep
    wandb = None  # type: ignore


class TinyCNN(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_synthetic_dataset(num_samples: int = 2000):
    # Fake "image" data: 1x28x28 with random labels
    x = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(x, y)


def train(
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    use_wandb: bool = True,
    wandb_project: str = "tb-cnn-demo",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = make_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logdir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=logdir)

    run = None
    if use_wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed; pip install wandb or set use_wandb=False")
        run = wandb.init(
            project=wandb_project,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": device,
            },
        )
        wandb.watch(model, log_freq=100)

    global_step = 0
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("loss/train_step", loss.item(), global_step)
            if run:
                wandb.log(
                    {"loss/train_step": loss.item(), "epoch": epoch, "step": global_step},
                    step=global_step,
                )
            global_step += 1

        avg_loss = running_loss / len(loader)
        writer.add_scalar("loss/train_epoch", avg_loss, epoch)
        if run:
            wandb.log({"loss/train_epoch": avg_loss, "epoch": epoch}, step=global_step)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    writer.add_text("meta", f"device={device}")
    writer.close()
    print(f"Logs written to {logdir}")
    if run:
        print(f"W&B run URL: {run.url}")
        run.finish()
    return logdir


if __name__ == "__main__":
    # Disable wandb by setting USE_WANDB=0
    use_wandb = os.getenv("USE_WANDB", "1") not in ("0", "false", "False")
    train(use_wandb=use_wandb)

