import sys
import typer
import polars as pl

from pathlib import Path

app = typer.Typer()

@app.command()
def main(
    taxa_file: Path = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
    output_file: Path = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
):
    # load the taxa data
    taxa = pl.scan_csv(
        taxa_file,
        separator="\t",
        # these two flags needed to prevent errors
        quote_char=None,
        ignore_errors=True,
    )
    # filter to just the active, plant, species
    plant_taxa = taxa.filter(
        (pl.col("active") == True)
        & (pl.col("ancestry").str.starts_with("48460/47126/"))  # plants
        & (pl.col("rank") == "species")
    )
    plant_taxa.sink_csv(output_file, separator="\t", engine="streaming")


if __name__ == "__main__":
    app()