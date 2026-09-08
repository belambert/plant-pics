import shutil
from collections import defaultdict
from pathlib import Path

import polars as pl
import typer
from tqdm import tqdm

app = typer.Typer()


@app.command()
def main(
    source_dir: str = typer.Argument(
        ..., help="Source directory containing photos named by photo_id"
    ),
    tsv_file: str = typer.Argument(..., help="TSV file with photo_id to name mapping"),
    dest_dir: str = typer.Argument(
        ..., help="Destination directory to create and copy photos to"
    ),
):
    """
    Organize photos by species name.

    Reads photo_id to species name mapping from TSV file and copies photos
    from source directory to destination directory with species names as filenames.
    """
    organize_by_species(source_dir, tsv_file, dest_dir)


def normalize_name(name: str) -> str:
    """Convert species name to lowercase with underscores."""
    return name.lower().replace(" ", "_")


def organize_by_species(source_dir: str, tsv_file: str, dest_dir: str):
    """
    Organize photos by species name.

    Args:
        source_dir: Source directory containing photos named by photo_id
        tsv_file: TSV file with photo_id to name mapping
        dest_dir: Destination directory to create and copy photos to
    """
    # Read the TSV file
    print(f"Reading TSV file: {tsv_file}")
    df = pl.read_csv(tsv_file, separator="\t")

    # Create photo_id to (name, extension) mapping
    photo_mapping = {}
    for row in df.iter_rows(named=True):
        photo_id = str(row["photo_id"])
        name = row["name"]
        extension = row["extension"]
        photo_mapping[photo_id] = (name, extension)

    print(f"Found {len(photo_mapping)} photo mappings")

    # Create destination directory
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Created destination directory: {dest_dir}")

    # Get all files in source directory
    source_path = Path(source_dir)
    source_files = list(source_path.iterdir())
    print(f"Found {len(source_files)} files in source directory")

    # Track name usage for handling duplicates
    name_counts = defaultdict(int)

    # Copy files
    copied = 0
    skipped = 0

    with tqdm(total=len(source_files), desc="Copying files", unit="file") as pbar:
        for source_file in source_files:
            if not source_file.is_file():
                skipped += 1
                pbar.update(1)
                continue

            # Extract photo_id from filename (remove extension)
            photo_id = source_file.stem

            # Look up the mapping
            if photo_id not in photo_mapping:
                pbar.write(f"Warning: No mapping found for photo_id {photo_id}")
                skipped += 1
                pbar.update(1)
                continue

            name, extension = photo_mapping[photo_id]
            normalized_name = normalize_name(name)

            # Handle duplicates by appending counter
            name_counts[normalized_name] += 1
            if name_counts[normalized_name] > 1:
                dest_filename = (
                    f"{normalized_name}_{name_counts[normalized_name]}.{extension}"
                )
            else:
                dest_filename = f"{normalized_name}.{extension}"

            dest_file = dest_path / dest_filename

            # Copy the file
            shutil.copy2(source_file, dest_file)
            copied += 1
            pbar.update(1)

    print(f"\nCompleted:")
    print(f"  Copied: {copied}")
    print(f"  Skipped: {skipped}")
    print(f"  Destination: {dest_dir}")


if __name__ == "__main__":
    app()
