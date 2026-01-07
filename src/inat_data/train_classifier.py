import os
from pathlib import Path

import numpy as np
import torch
import typer
from datasets import load_dataset
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
        "google/vit-base-patch16-224",  # or google/vit-base-patch16-224-in21k?
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
        32,
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
    save_steps: int = typer.Option(
        500,
        "--save-steps",
        help="Save checkpoint every N steps",
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
        save_steps=save_steps,
        use_wandb=not no_wandb,
        limit=limit,
    )


def train_vit_classifier(
    output_dir: str = "models/vit-inat-classifier",
    model_name: str = "google/vit-base-patch16-224",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    save_steps: int = 500,
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
        save_steps: Save checkpoint every N steps
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

    # Load image processor and create transforms
    print(f"\nLoading pretrained model: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    train_transforms, val_transforms = create_transforms(image_processor)

    # Preprocess datasets
    print("\nPreprocessing datasets...")
    dataset["train"].set_transform(
        lambda examples: preprocess_train(examples, train_transforms, label2id)
    )

    dataset["val"].set_transform(
        lambda examples: preprocess_val(examples, val_transforms, label2id)
    )

    dataset["test"].set_transform(
        lambda examples: preprocess_val(examples, val_transforms, label2id)
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

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb" if use_wandb else None,
        dataloader_num_workers=0,  # Set to 0 to avoid pickling issues with transforms
    )

    # Initialize trainer
    print("\nInitializing trainer...")
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
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

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


def preprocess_train(examples, transform, label2id):
    """Apply transformations to training examples."""
    examples["pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["image"]
    ]
    # Convert class labels to integer IDs
    examples["labels"] = [label2id[cls] for cls in examples["class"]]
    # Remove raw images to avoid collator errors
    del examples["image"]
    del examples["class"]
    return examples


def preprocess_val(examples, transform, label2id):
    """Apply transformations to validation examples."""
    examples["pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["image"]
    ]
    # Convert class labels to integer IDs
    examples["labels"] = [label2id[cls] for cls in examples["class"]]
    # Remove raw images to avoid collator errors
    del examples["image"]
    del examples["class"]
    return examples


def compute_metrics(eval_pred):
    """Compute accuracy metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


if __name__ == "__main__":
    app()
