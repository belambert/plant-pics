import typer
import polars as pl

from pathlib import Path

app = typer.Typer()

@app.command()
def main(
    obs_file: Path = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
    taxa_file: Path = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
    output_file: Path = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
):

    taxa = pl.scan_csv(taxa_file, separator="\t")

    print("processing observations...") 
    # this only keeps memory low if we give it the uncompressed version?
    obs_df = (
        pl.scan_csv(
            obs_file,
            separator="\t",
            low_memory=True,
        )
        .filter(pl.col("quality_grade") == "research")
        .filter((pl.col("latitude") > 41) & (pl.col("latitude") < 45))
        .filter((pl.col("longitude") > -73) & (pl.col("longitude") < -70))
        .select(["taxon_id", "observation_uuid"])
    )
    filtered_obs = obs_df.join(taxa, on="taxon_id", how="inner")
    filtered_obs.sink_csv(output_file, separator="\t", engine="streaming")


if __name__ == "__main__":
    app()