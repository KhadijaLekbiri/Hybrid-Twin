import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from pathlib import Path

from hybrid_twin.model.model import BiasLearner
from hybrid_twin.model.loss  import PhysicsInformedLoss
from hybrid_twin.config.loader import load_config

cfg = load_config(Path(__file__).resolve().parent.parent.parent.parent / "config.yaml")

DATA_DIR    = cfg["paths"]["DATA_DIR"]
OUTPUT_DIR  = cfg["paths"]["OUTPUT_DIR"];  OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_SIZE  = cfg["model"]["BATCH_SIZE"]
EPOCHS      = cfg["model"]["EPOCHS"]
LR          = cfg["model"]["LR"]
ALPHA       = cfg["model"]["ALPHA"]
BETA        = cfg["model"]["BETA"]
HIDDEN_SIZE = cfg["model"]["HIDDEN_SIZE"]
NUM_LAYERS  = cfg["model"]["NUM_LAYERS"]
DROPOUT     = cfg["model"]["DROPOUT"]
INPUT_SIZE  = cfg["model"]["INPUT_SIZE"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_seq_target(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim == 2:
        y = y[:, :, None]
    return y


class Trainer:
    """
    Manages the full training loop for BiasLearner.
    """

    def __init__(self, model: BiasLearner, criterion: PhysicsInformedLoss, cfg: dict):
        self.model     = model
        self.criterion = criterion
        self.cfg       = cfg
        self.optimiser = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode="min", factor=0.5, patience=5
        )

    def _load_data(self):
        X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
        y_train = _as_seq_target(
            np.load(DATA_DIR / "y_train.npy").astype(np.float32)
        ).reshape(X_train.shape[0], X_train.shape[1], 1)

        X_val = np.load(DATA_DIR / "X_val.npy").astype(np.float32)
        y_val = _as_seq_target(
            np.load(DATA_DIR / "y_val.npy").astype(np.float32)
        ).reshape(X_val.shape[0], X_val.shape[1], 1)

        print(f"X_train : {X_train.shape}   y_train : {y_train.shape}")
        print(f"X_val   : {X_val.shape}     y_val   : {y_val.shape}")

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
        return (
            TorchDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
            TorchDataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False),
        )

    def evaluate(self, loader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                total += self.criterion(self.model(X_batch), y_batch).item() * X_batch.size(0)
        return total / len(loader.dataset)

    def train(self) -> dict:
        train_loader, val_loader = self._load_data()

        print(f"\nDevice  : {DEVICE}")
        print(f"Model   : {sum(p.numel() for p in self.model.parameters()):,} parameters\n")

        best_val_loss = float("inf")
        history       = {"train": [], "val": []}

        for epoch in range(1, self.cfg["EPOCHS"] + 1):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                self.optimiser.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimiser.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            val_loss = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            history["train"].append(train_loss)
            history["val"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(OUTPUT_DIR / "best_bias_learner.pt", epoch, val_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{self.cfg['EPOCHS']} | "
                      f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
        np.save(OUTPUT_DIR / "loss_history.npy", history)
        return history

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        torch.save({
            "epoch":            epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss":         val_loss,
            "config": {
                "input_size":  INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers":  NUM_LAYERS,
                "dropout":     DROPOUT,
            },
        }, path)


# Backward-compat alias
def train():
    model     = BiasLearner(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = PhysicsInformedLoss(alpha=ALPHA, beta=BETA)
    trainer   = Trainer(model, criterion, cfg["model"])
    history   = trainer.train()
    return model, history


if __name__ == "__main__":
    train()
