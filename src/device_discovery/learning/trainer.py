from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from device_discovery.learning.feature_encoder import FeatureEncoder, load_observations
from device_discovery.learning.heuristic_classifier import DEVICE_CLASSES
from device_discovery.learning.mlp_classifier import MLPClassifier

LABEL_TO_INDEX = {label: index for index, label in enumerate(DEVICE_CLASSES)}


@dataclass
class TrainingConfig:
    """Configuration for MLP training."""

    data_dir: str = "data/generated/moderate"
    output_dir: str = "artifacts/mlp"
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 12
    hidden_dim: int = 64
    dropout: float = 0.1


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select CUDA when available, otherwise use CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device


def load_labels(csv_path: str | Path) -> list[str]:
    """Load device class labels from a labeled CSV file.

    Args:
        csv_path: Path to the training or validation CSV.

    Returns:
        Device class labels in the same row order as the CSV.

    Raises:
        ValueError: If the label column is missing or a label
            is not one of the supported device classes.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Missing label column: {csv_path}")

        labels = []

        for row in reader:
            label = row["label"]

            if label not in LABEL_TO_INDEX:
                raise ValueError(f"Unknown class label: {label!r}")

            labels.append(label)

    return labels


def build_tensors(csv_path: str | Path, encoder: FeatureEncoder) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode observations and labels into PyTorch tensors.

    Args:
        csv_path: Path to a labeled training or validation CSV.
        encoder: Feature encoder already fitted on training data.

    Returns:
        A tuple containing:
            - Feature tensor of shape (N, input_dim).
            - Target tensor of shape (N,).
    """
    observations = load_observations(csv_path)
    labels = load_labels(csv_path)

    if len(observations) != len(labels):
        raise ValueError("Observation and label counts do not match")

    if not observations:
        raise ValueError("Dataset is empty")

    features = torch.tensor(encoder.transform_many(observations), dtype=torch.float32)

    targets = torch.tensor([LABEL_TO_INDEX[label] for label in labels], dtype=torch.long)

    return features, targets


@torch.no_grad()
def evaluate(model: MLPClassifier, features: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Calculate cross-entropy loss and classification accuracy.

    The model and tensors must reside on the same device.

    Args:
        model: Trained or partially trained MLP.
        features: Encoded feature tensor.
        targets: Integer class labels.

    Returns:
        A tuple of (loss, accuracy).
    """
    model.eval()

    logits = model(features)
    loss = nn.functional.cross_entropy(logits, targets)

    predictions = logits.argmax(dim=1)

    accuracy = (predictions == targets).float().mean().item()

    return float(loss.item()), float(accuracy)


def fit_mlp(
    x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, input_dim: int, config: TrainingConfig, device: torch.device, verbose: bool = True
) -> tuple[MLPClassifier, int, float, list[dict[str, float]]]:
    """Train an MLP with early stopping on validation loss.

    Returns the model restored to its best checkpoint, the best epoch,
    the best validation loss and the per-epoch history.
    """
    # Keep training tensors on CPU for DataLoader.
    # Move each mini-batch to GPU during training.
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # Initialize neural network and optimizer
    model = MLPClassifier(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(x_train, y_train)

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed), pin_memory=(device.type == "cuda"))

    # Training metrics can be evaluated on GPU.
    x_train_eval = x_train.to(device)
    y_train_eval = y_train.to(device)

    # Training loop with validation and early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()

        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device, non_blocking=(device.type == "cuda"))

            batch_targets = batch_targets.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad()

            logits = model(batch_features)

            loss = criterion(logits, batch_targets)

            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(model, x_train_eval, y_train_eval)

        val_loss, val_accuracy = evaluate(model, x_val, y_val)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})

        if verbose:
            print(f"Epoch {epoch:03d} | " f"train_loss={train_loss:.4f} | " f"train_acc={train_accuracy:.4f} | " f"val_loss={val_loss:.4f} | " f"val_acc={val_accuracy:.4f}")

        # Save the checkpoint with the lowest validation loss.
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    # Restore the best model.
    model.load_state_dict(best_state)

    return model, best_epoch, best_val_loss, history


def train(config: TrainingConfig) -> dict[str, object]:
    """Train an MLP classifier using partially observed device data.

    This function implements the neural baseline training pipeline
    for the network device discovery experiments.

    The training pipeline consists of six steps:

    1. Load labeled training and validation observations from CSV files.
    2. Fit the feature encoder exclusively on training observations
       to prevent vocabulary leakage from validation or test data.
    3. Transform observations into fixed-dimensional feature tensors
       and convert device class labels into integer targets.
    4. Train the MLP classifier using CrossEntropyLoss and the
       Adam optimizer.
    5. Monitor validation loss and apply early stopping, retaining
       the model checkpoint with the lowest validation loss.
    6. Save the best model weights, feature encoder vocabulary,
       class mappings, training configuration, and training history.

    CUDA is used automatically when available. Otherwise, training
    falls back to CPU.

    The test dataset and hidden simulator ground truth are not
    accessed during training or model selection.

    Args:
        config: Training configuration containing dataset paths,
            output directory, seed, hyperparameters, and early
            stopping settings.

    Returns:
        A dictionary containing training configuration, dataset
        sizes, feature dimension, best epoch, validation metrics,
        checkpoint path, and complete training history.

    Raises:
        ValueError: If training parameters or dataset contents
            are invalid.
        FileNotFoundError: If a required CSV file is missing.
        RuntimeError: If training produces no valid checkpoint.

    Notes:
        - Feature vocabulary is learned from training data only.
        - Validation data does not update neural network weights.
        - The best checkpoint minimizes validation loss.
        - Softmax probabilities are not necessarily calibrated.
        - Robustness evaluation is performed separately in E1.
    """

    #  Validate configuration and initialize reproducibility

    if config.epochs <= 0:
        raise ValueError("epochs must be positive")

    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.patience <= 0:
        raise ValueError("patience must be positive")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    set_seed(config.seed)

    device = get_device()

    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"

    # Load observations and fit encoder on training data only
    train_observations = load_observations(train_path)

    encoder = FeatureEncoder().fit(train_observations)

    print(f"Feature dimensions: {encoder.n_features}")
    print(f"Firmware vocabulary: {encoder.firmware_values}")

    # Convert observations into tensors
    x_train, y_train = build_tensors(train_path, encoder)

    x_val, y_val = build_tensors(val_path, encoder)

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")

    model, best_epoch, best_val_loss, history = fit_mlp(x_train, y_train, x_val, y_val, input_dim=encoder.n_features, config=config, device=device)

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    _, best_val_accuracy = evaluate(model, x_val, y_val)

    # Save checkpoint and training summary
    checkpoint_path = output_dir / "model.pt"

    # Save weights on CPU for portable checkpoint loading.
    cpu_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    torch.save(
        {
            "model_state_dict": cpu_state_dict,
            "input_dim": encoder.n_features,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "classes": list(DEVICE_CLASSES),
            "firmware_values": list(encoder.firmware_values),
        },
        checkpoint_path,
    )

    summary = {
        "config": asdict(config),
        "device": str(device),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "input_dim": encoder.n_features,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }

    summary_path = output_dir / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print()
    print("=" * 50)
    print("Training completed")
    print(f"Device: {device}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 50)

    return summary


def main() -> None:
    """Run the MLP training pipeline from the command line."""
    parser = argparse.ArgumentParser(description=("Train the MLP baseline for " "neurosymbolic device discovery"))

    parser.add_argument("--data-dir", default="data/generated/moderate")

    parser.add_argument("--output-dir", default="artifacts/mlp")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--learning-rate", type=float, default=0.001)

    parser.add_argument("--patience", type=int, default=12)

    parser.add_argument("--hidden-dim", type=int, default=64)

    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    train(config)


if __name__ == "__main__":
    main()
