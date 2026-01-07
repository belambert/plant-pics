import typer
import polars as pl

from pathlib import Path

app = typer.Typer()

@app.command()
def main(
    obs_file: Path = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
    pics_file: Path = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
    output_file: Path = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
):

    obs_df = pl.scan_csv(obs_file, separator="\t", low_memory=True)

    # plant_obs = obs_df.join(plant_taxa, on="taxon_id", how="inner")

    print("processing photos...")
    pics_df = (
        pl.scan_csv(
            pics_file,
            separator="\t",
            low_memory=True
        )
        .filter(~pl.col("license").str.contains("ND"))
        .filter((pl.col("width") > 1000) & (pl.col("height") > 1000))
        .filter(pl.col("position") == 1)
    )

    filtered_pics = obs_df.join(pics_df, on="observation_uuid", how="inner")

    # columns = ["taxon_id", "observation_uuid", "name", "photo_id", "extension"]
    # plant_pics = photos_df.select(columns)
    # plant_pics = (
    #     plant_obs.join(photos_df, on="observation_uuid", how="inner")
    #     .group_by("taxon_id")
    #     .head(10)
    # )

    filtered_pics.sink_csv(output_file, separator="\t", engine="streaming")


if __name__ == "__main__":
    app()