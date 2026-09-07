"""Build ne_plant_photos: the nature-labelled photos, each carrying its iNaturalist metadata."""

import json
from pathlib import Path

import polars as pl
import typer
from datasets import Dataset, Image
from vlm_toolkit.hf_dataset import push_card

LABEL = "nature"
TARGET = "blambert/ne_plant_photos"

app = typer.Typer()


@app.command()
def main(
    results_file: Path = typer.Argument(..., help="JSONL written by `vlm process`"),
    pics_tsv: Path = typer.Argument(
        Path("data/ne_pics.tsv"), help="Filter output holding the per-photo metadata"
    ),
    target: str = typer.Option(
        TARGET, "--target", "-t", help="Dataset repo to push to"
    ),
    label: str = typer.Option(
        LABEL, "--label", help="Keep only images carrying this label"
    ),
    card: Path = typer.Option(
        Path("cards/ne_plant_photos.md"),
        "--card",
        help="Dataset card to upload as the README",
    ),
    private: bool = typer.Option(
        False, "--private", help="Create the dataset repo as private"
    ),
):
    """Keep the images labelled `label`, join their metadata, and push to the Hub."""
    images = keep_labelled(results_file, label)
    print(f"{len(images)} images labelled {label}")

    ds = build_dataset(images, pl.read_csv(pics_tsv, separator="\t"))
    ds.push_to_hub(target, private=private, split="train", embed_external_files=True)
    if card:
        push_card(target, card)

    print(f"pushed {len(ds)} rows to https://huggingface.co/datasets/{target}")


def keep_labelled(results_file: Path, label: str) -> list[Path]:
    """Read a classification run and return the paths of the images carrying `label`."""
    with open(results_file) as f:
        items = [json.loads(line) for line in f if line.strip()]
    paths = [
        (results_file.parent / item["file_name"]).resolve()
        for item in items
        if item["output"].strip() == label
    ]

    missing = [p for p in paths if not p.is_file()]
    for path in missing[:10]:
        print(f"skipping missing image: {path}")
    if len(missing) > 10:
        print(f"...and {len(missing) - 10} more")
    return [p for p in paths if p.is_file()]


def build_dataset(images: list[Path], meta: pl.DataFrame) -> Dataset:
    """Pair each image with the metadata row for its photo id, skipping those without one."""
    # images are named after their photo id, which is what the filter output is keyed on
    picked = pl.DataFrame(
        {
            "image": [str(p) for p in images],
            "file_name": [p.name for p in images],
            "photo_id": [int(p.stem) for p in images],
        }
    )
    df = picked.join(meta, on="photo_id", how="inner")
    if len(df) < len(picked):
        missing = picked.join(meta, on="photo_id", how="anti")["file_name"].to_list()
        for name in missing[:10]:
            print(f"skipping image missing metadata: {name}")
        if len(missing) > 10:
            print(f"...and {len(missing) - 10} more")

    return Dataset.from_polars(df).cast_column("image", Image())


if __name__ == "__main__":
    app()
