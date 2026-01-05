import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import polars as pl
import typer
from tqdm import tqdm


class ImageSize(str, Enum):
    """Available iNaturalist image sizes."""

    original = "original"  # 2048px
    large = "large"  # 1024px
    medium = "medium"  # 500px
    small = "small"  # 240px
    thumb = "thumb"  # 100px
    square = "square"  # exactly 75x75px, cropped to be square


def download_photo(
    photo_id: int, extension: str, size: ImageSize, output_dir: Path
) -> tuple[int, bool, str]:
    """
    Download a single photo from S3.

    Args:
        photo_id: The photo ID
        extension: The file extension (e.g., 'jpg')
        size: The image size to download
        output_dir: Directory to save the file

    Returns:
        Tuple of (photo_id, success, error_message)
    """
    s3_path = f"s3://inaturalist-open-data/photos/{photo_id}/{size.value}.{extension}"
    output_file = output_dir / f"{photo_id}.{extension}"

    try:
        result = subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", s3_path, str(output_file)],
            capture_output=True,
            text=True,
            check=True,
        )
        return (photo_id, True, "")
    except subprocess.CalledProcessError as e:
        return (photo_id, False, e.stderr)


def download_photos(input_file: str, size: ImageSize, max_workers: int = 50):
    """
    Download iNaturalist photos at specified size.

    Args:
        input_file: Path to the input TSV file (e.g., data/plant_pics.tsv)
        size: The image size to download
        max_workers: Number of parallel downloads
    """
    # Read the TSV file
    df = pl.read_csv(input_file, separator="\t")

    # Create output directory named after the size
    output_dir = Path(size.value)
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading {len(df)} {size.value} photos to {output_dir}/")
    print(f"Using {max_workers} parallel workers")

    # Download files in parallel with progress bar
    failed = 0
    failed_ids = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_photo, row["photo_id"], row["extension"], size, output_dir
            ): row["photo_id"]
            for row in df.iter_rows(named=True)
        }

        with tqdm(total=len(df), desc="Downloading", unit="photo") as pbar:
            for future in as_completed(futures):
                photo_id, success, error = future.result()

                if not success:
                    failed += 1
                    failed_ids.append(photo_id)
                    pbar.write(f"Failed {photo_id}: {error.strip()}")

                pbar.update(1)

    print(f"\nCompleted: {len(df) - failed}/{len(df)}")
    if failed > 0:
        print(f"Failed: {failed}")
        print(f"Failed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")


app = typer.Typer()


@app.command()
def main(
    input_file: str = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
    size: ImageSize = typer.Option(
        ImageSize.large, "--size", "-s", help="Image size to download"
    ),
    workers: int = typer.Option(
        50, "--workers", "-w", help="Number of parallel download workers"
    ),
):
    """
    Download iNaturalist photos from S3 at the specified size.
    """
    download_photos(input_file, size, max_workers=workers)


if __name__ == "__main__":
    app()
