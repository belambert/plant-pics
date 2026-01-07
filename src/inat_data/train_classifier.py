import os
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
import typer
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

import wandb

# Weights & Biases configuration
WANDB_PROJECT = "inat-vit-classifier"

# Random seed for reproducibility
RANDOM_SEED = 42

# Dataset split ratios (train/val/test)
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.6, 0.2, 0.2

app = typer.Typer()


@app.command()
def main(
    output_dir: str = typer.Option(
        "models/vit-inat-classifier",
        "--output-dir",
        "-o",
        help="Directory to save the trained model",
    ),
    model_name: str = typer.Option(
        # or : google/vit-large-patch16-224-in21k
        "google/vit-base-patch16-224-in21k",
        "--model",
        "-m",
        help="Pretrained ViT model to use",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        "-b",
        help="Batch size for training",
    ),
    learning_rate: float = typer.Option(
        2e-5,
        "--learning-rate",
        "-lr",
        help="Learning rate for optimizer",
    ),
    no_wandb: bool = typer.Option(
        False,
        "--no-wandb",
        help="Disable Weights & Biases logging",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        help="Limit total dataset size before splitting (useful for quick testing)",
    ),
):
    """
    Train a Vision Transformer (ViT) image classifier on iNaturalist data.

    This script loads the blambert/inat_sample_classes dataset from Hugging Face
    and fine-tunes a pretrained ViT model for plant species classification.
    Logs training metrics to Weights & Biases.
    """
    train_vit_classifier(
        output_dir=output_dir,
        model_name=model_name,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_wandb=not no_wandb,
        limit=limit,
    )


def get_device():
    """Detect which device (cuda/mps/cpu) is being used."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def print_model_parameters(model):
    """Print the number of parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")


def print_class_distribution(dataset, labels):
    """Print table showing class distribution across train/val/test splits."""
    console = Console()
    table = Table(
        title="\nClass Distribution", show_header=True, header_style="bold magenta"
    )
    table.add_column("Class", style="cyan")
    table.add_column("Train", justify="right", style="green")
    table.add_column("Val", justify="right", style="yellow")
    table.add_column("Test", justify="right", style="blue")
    table.add_column("Total", justify="right", style="bold")

    train_counts = Counter(dataset["train"]["class"])
    val_counts = Counter(dataset["val"]["class"])
    test_counts = Counter(dataset["test"]["class"])

    for label in sorted(labels):
        t, v, te = (
            train_counts.get(label, 0),
            val_counts.get(label, 0),
            test_counts.get(label, 0),
        )
        table.add_row(label, str(t), str(v), str(te), str(t + v + te))

    table.add_section()
    table.add_row(
        "TOTAL",
        str(len(dataset["train"])),
        str(len(dataset["val"])),
        str(len(dataset["test"])),
        str(len(dataset["train"]) + len(dataset["val"]) + len(dataset["test"])),
        style="bold",
    )

    console.print(table)


def train_vit_classifier(
    output_dir: str = "models/vit-inat-classifier",
    model_name: str = "google/vit-base-patch16-224",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    use_wandb: bool = True,
    limit: int = None,
):
    """
    Train a Vision Transformer image classifier on iNaturalist data.

    Args:
        output_dir: Directory to save the trained model
        model_name: Pretrained ViT model to use
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        use_wandb: Whether to use Weights & Biases for logging
        limit: Limit total dataset size before splitting (for quick testing)
    """
    # Initialize Weights & Biases
    if use_wandb:
        config = {
            "model_name": model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "output_dir": output_dir,
        }
        if limit is not None:
            config["limit"] = limit

        wandb.init(project=WANDB_PROJECT, config=config)
        print(f"Weights & Biases initialized. Project: {WANDB_PROJECT}")

    print(f"Loading dataset: blambert/inat_sample_classes")
    dataset = load_dataset("blambert/inat_sample_classes")

    print(f"\nOriginal dataset info:")
    print(f"  Total samples: {len(dataset['train'])}")

    # Shuffle and split the dataset
    print(f"\nShuffling dataset with seed {RANDOM_SEED}...")
    dataset_shuffled = dataset["train"].shuffle(seed=RANDOM_SEED)

    # Apply limit if specified
    if limit is not None:
        print(f"Limiting dataset to {limit} samples...")
        dataset_shuffled = dataset_shuffled.select(
            range(min(limit, len(dataset_shuffled)))
        )
        print(f"  Limited samples: {len(dataset_shuffled)}")

    print(
        f"\nSplitting dataset {TRAIN_SPLIT:.0%} train / {VAL_SPLIT:.0%} val / {TEST_SPLIT:.0%} test..."
    )
    # First split: separate out test set
    train_val_test = dataset_shuffled.train_test_split(
        test_size=TEST_SPLIT, seed=RANDOM_SEED
    )

    # Second split: separate train and val from the remaining data
    train_val = train_val_test["train"].train_test_split(
        test_size=VAL_SPLIT
        / (TRAIN_SPLIT + VAL_SPLIT),  # Adjust val size relative to remaining data
        seed=RANDOM_SEED,
    )

    # Create final dataset with train/val/test splits
    dataset = {
        "train": train_val["train"],
        "val": train_val["test"],
        "test": train_val_test["test"],
    }

    print(f"\nSplit dataset info:")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Val samples: {len(dataset['val'])}")
    print(f"  Test samples: {len(dataset['test'])}")

    # Get label information from the dataset
    print("\nExtracting unique classes from dataset...")
    unique_classes = sorted(set(dataset["train"]["class"]))
    labels = unique_classes
    num_labels = len(labels)
    print(f"  Number of classes: {num_labels}")
    print(f"  Classes: {labels[:5]}..." if len(labels) > 5 else f"  Classes: {labels}")

    # Create label mappings
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    # Print class distribution table
    print_class_distribution(dataset, labels)

    # Load image processor and create transforms
    print(f"\nLoading pretrained model: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    train_transforms, val_transforms = create_transforms(image_processor)

    # Preprocess datasets using partial to make them picklable
    print("\nPreprocessing datasets...")
    dataset["train"].set_transform(
        partial(preprocess_train, transform=train_transforms, label2id=label2id)
    )

    dataset["val"].set_transform(
        partial(preprocess_val, transform=val_transforms, label2id=label2id)
    )

    dataset["test"].set_transform(
        partial(preprocess_val, transform=val_transforms, label2id=label2id)
    )

    # Load model
    print(f"\nInitializing model with {num_labels} output classes...")

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Print model parameter counts
    print_model_parameters(model)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check device type and configure accordingly
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    # Configure data loading workers based on device
    num_workers = 0 if use_mps else 4

    # Enable mixed precision training on CUDA
    if use_cuda:
        fp16 = not torch.cuda.is_bf16_supported()
        bf16 = torch.cuda.is_bf16_supported()
    else:
        fp16 = False
        bf16 = False

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=0.05,  # Evaluate every 5% of training
        save_strategy="steps",
        save_steps=0.05,  # Save every 5% of training
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb" if use_wandb else None,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=False if use_mps else True,
        fp16=fp16,
        bf16=bf16,
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    compute_metrics = compute_metrics_factory(id2label, use_wandb=use_wandb)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],  # Use validation set for training evaluation
        compute_metrics=compute_metrics,
    )

    # Train
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {get_device()}")
    print(f"  Data workers: {num_workers}")
    if fp16:
        print(f"  Mixed precision: fp16")
    elif bf16:
        print(f"  Mixed precision: bf16")

    train_result = trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    image_processor.save_pretrained(output_dir)

    # Evaluate on validation set (final check with best model)
    print("\nFinal validation accuracy...")
    val_results = trainer.evaluate(dataset["val"])

    # Evaluate on test set (held-out evaluation)
    print("Evaluating on test set...")
    test_results = trainer.evaluate(dataset["test"])

    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print(f"  Total training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"  Final training loss: {train_result.metrics['train_loss']:.4f}")
    print(f"  Best validation accuracy: {val_results['eval_accuracy']:.4f}")
    print(f"  Final test accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"  Model saved to: {output_dir}")
    print("=" * 60)

    # Finish wandb run
    if use_wandb:
        wandb.finish()


def create_transforms(image_processor):
    """
    Create image transformations for training and validation.

    Args:
        image_processor: The image processor from the pretrained model

    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    # Get the expected input size and normalization from the processor
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )

    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )

    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    return train_transforms, val_transforms


def preprocess_train(examples, *, transform, label2id):
    """Apply transformations to training examples."""
    examples["pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["image"]
    ]
    examples["labels"] = [label2id[cls] for cls in examples["class"]]
    del examples["image"]
    del examples["class"]
    return examples


def preprocess_val(examples, *, transform, label2id):
    """Apply transformations to validation examples."""
    examples["pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["image"]
    ]
    examples["labels"] = [label2id[cls] for cls in examples["class"]]
    del examples["image"]
    del examples["class"]
    return examples


def compute_metrics_factory(id2label, use_wandb=True):
    """
    Create a compute_metrics function with access to label mappings.

    Args:
        id2label: Dictionary mapping class IDs to class names
        use_wandb: Whether to log per-class metrics to wandb

    Returns:
        compute_metrics function
    """

    def compute_metrics(eval_pred):
        """Compute accuracy and per-class precision/recall metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Overall accuracy
        accuracy = (predictions == labels).mean()

        # Per-class precision and recall
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        # Create metrics dict
        metrics = {"accuracy": accuracy}

        # Add per-class metrics
        if use_wandb:
            # Log per-class metrics to wandb
            class_metrics = {}
            for idx, (prec, rec, f1_score) in enumerate(zip(precision, recall, f1)):
                class_name = id2label.get(idx, f"class_{idx}")
                class_metrics[f"precision/{class_name}"] = prec
                class_metrics[f"recall/{class_name}"] = rec
                class_metrics[f"f1/{class_name}"] = f1_score

            # Log to wandb
            if wandb.run is not None:
                wandb.log(class_metrics)

        # Also compute macro averages
        metrics["precision_macro"] = precision.mean()
        metrics["recall_macro"] = recall.mean()
        metrics["f1_macro"] = f1.mean()

        return metrics

    return compute_metrics


if __name__ == "__main__":
    app()
