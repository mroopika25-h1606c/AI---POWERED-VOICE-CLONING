import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.dataset_loader import VoiceSpoofingDataLoader
from models.aasist import AASIST


def calculate_metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    predictions = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
    }

    try:
        metrics["auc_roc"] = roc_auc_score(
            labels,
            probabilities[:, 1],
        )
    except ValueError:
        metrics["auc_roc"] = 0.0

    return metrics


def run_training_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    for batch_number, (audio, labels) in enumerate(
        loader,
        start=1,
    ):
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(audio)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        total_loss += loss.item()

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

        if batch_number % 10 == 0:
            print(
                f"Training batch "
                f"{batch_number}/{len(loader)} "
                f"loss={loss.item():.4f}"
            )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    metrics["loss"] = total_loss / len(loader)

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    name="Validation",
):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    for batch_number, (audio, labels) in enumerate(
        loader,
        start=1,
    ):
        audio = audio.to(device)
        labels = labels.to(device)

        logits = model(audio)
        loss = criterion(logits, labels)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        total_loss += loss.item()

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

        if batch_number % 10 == 0:
            print(
                f"{name} batch "
                f"{batch_number}/{len(loader)}"
            )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    metrics["loss"] = total_loss / len(loader)

    return metrics


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1-score:  {metrics['f1'] * 100:.2f}%")
    print(f"ROC-AUC:   {metrics['auc_roc'] * 100:.2f}%")


def save_checkpoint(
    output_path,
    model,
    optimizer,
    epoch,
    metrics,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "label_mapping": {
            "bonafide": 0,
            "spoof": 1,
        },
    }

    torch.save(checkpoint, output_path)

    print(f"Checkpoint saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="LA_cpu",
    )

    parser.add_argument(
        "--metadata-dir",
        default="metadata",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    args = parser.parse_args()

    metadata_folder = Path(args.metadata_dir)

    train_csv = (
        metadata_folder
        / f"{args.dataset}_train.csv"
    )

    validation_csv = (
        metadata_folder
        / f"{args.dataset}_val.csv"
    )

    test_csv = (
        metadata_folder
        / f"{args.dataset}_test.csv"
    )

    for csv_path in [
        train_csv,
        validation_csv,
        test_csv,
    ]:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {csv_path}"
            )

    requested_device = args.device.lower()

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA is unavailable. Using CPU."
            )
            requested_device = "cpu"

    device = torch.device(requested_device)

    print(f"Device: {device}")
    print(f"Training CSV: {train_csv}")
    print(f"Validation CSV: {validation_csv}")
    print(f"Test CSV: {test_csv}")

    train_loader = VoiceSpoofingDataLoader(
        str(train_csv),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        cache_preprocessed=False,
    )

    validation_loader = VoiceSpoofingDataLoader(
        str(validation_csv),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cache_preprocessed=False,
    )

    test_loader = VoiceSpoofingDataLoader(
        str(test_csv),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        cache_preprocessed=False,
    )

    model = AASIST(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.00001,
    )

    checkpoints_folder = Path("checkpoints")
    checkpoints_folder.mkdir(exist_ok=True)

    latest_checkpoint = (
        checkpoints_folder / "latest.pth"
    )

    best_checkpoint = (
        checkpoints_folder / "best.pth"
    )

    best_validation_f1 = -1.0

    print(
        f"\nStarting training for "
        f"{args.epochs} epochs"
    )

    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 60)

        training_metrics = run_training_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            name="Validation",
        )

        print_metrics(
            "TRAINING RESULTS",
            training_metrics,
        )

        print_metrics(
            "VALIDATION RESULTS",
            validation_metrics,
        )

        save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            epoch,
            validation_metrics,
        )

        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                validation_metrics,
            )

            print("New best model saved")

    print("\nTraining completed")

    if best_checkpoint.exists():
        checkpoint = torch.load(
            best_checkpoint,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            f"Loaded best checkpoint from "
            f"epoch {checkpoint['epoch']}"
        )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        name="Testing",
    )

    print_metrics(
        "FINAL TEST RESULTS",
        test_metrics,
    )


if __name__ == "__main__":
    main()