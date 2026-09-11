from pathlib import Path

import polars as pl
import typer

OUT_PREFIX = "plant"
MIN_RESOLUTION = 750
# ~20k pics at 10 per species, 152k at 100, 862k unlimited
MAX_PER_SPECIES = 100
# 0 keeps every species the photos cover
MAX_SPECIES = 0
# covers all of NE from NYC to New Brunswick
LAT_MIN, LAT_MAX = 41, 48
LON_MIN, LON_MAX = -74, -67
# what the taxa output carries, one row per species the photos cover
TAXA_COLUMNS = ["taxon_id", "name", "ancestry"]
# everything the dataset build carries alongside each image
PIC_COLUMNS = [
    "photo_id",
    "photo_uuid",
    "extension",
    "license",
    "width",
    "height",
    "observer_id",
    "observation_uuid",
    "observed_on",
    "latitude",
    "longitude",
    "taxon_id",
    "name",
    "ancestry",
    "observations",
]

app = typer.Typer()


@app.command()
def main(
    data_dir: Path = typer.Argument(
        Path("data"), help="Directory holding the iNaturalist CSVs and filter output"
    ),
    out_prefix: str = typer.Option(
        OUT_PREFIX,
        "--out-prefix",
        "-o",
        help="Prefix for the output files, e.g. 'ne' writes ne_taxa.tsv and ne_pics.tsv",
    ),
    lat_min: float = typer.Option(LAT_MIN, "--lat-min", help="Southern bound"),
    lat_max: float = typer.Option(LAT_MAX, "--lat-max", help="Northern bound"),
    lon_min: float = typer.Option(LON_MIN, "--lon-min", help="Western bound"),
    lon_max: float = typer.Option(LON_MAX, "--lon-max", help="Eastern bound"),
    min_resolution: int = typer.Option(
        MIN_RESOLUTION, "--min-resolution", help="Minimum photo width and height"
    ),
    max_per_species: int = typer.Option(
        MAX_PER_SPECIES, "--max-per-species", help="Cap on photos kept per species"
    ),
    max_species: int = typer.Option(
        MAX_SPECIES,
        "--max-species",
        help="Keep only this many species, most observed first; 0 keeps all",
    ),
):
    """Filter iNaturalist metadata down to plant photos within a bounding box."""
    # load the taxa data
    taxa = pl.scan_csv(
        data_dir / "taxa.csv.gz",
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
    plant_taxa = plant_taxa.select(TAXA_COLUMNS)

    print("processing observations...")
    obs_df = (
        pl.scan_csv(
            data_dir / "observations.csv.gz",
            separator="\t",
            low_memory=True,
        )
        .filter(pl.col("quality_grade") == "research")
        .filter((pl.col("latitude") > lat_min) & (pl.col("latitude") < lat_max))
        .filter((pl.col("longitude") > lon_min) & (pl.col("longitude") < lon_max))
        .select(
            ["taxon_id", "observation_uuid", "latitude", "longitude", "observed_on"]
        )
    )
    plant_obs = obs_df.join(plant_taxa, on="taxon_id", how="inner")

    print("processing photos...")
    photos_df = (
        pl.scan_csv(
            data_dir / "photos.csv.gz",
            separator="\t",
            low_memory=True,
        )
        # don't include those that don't allow derivates
        .filter(~pl.col("license").str.contains("ND"))
        .filter(
            (pl.col("width") >= min_resolution) & (pl.col("height") >= min_resolution)
        )
        .filter(pl.col("position") == 1)
    )

    candidates = plant_obs.join(photos_df, on="observation_uuid", how="inner")

    # rank before the per-species cap, so the counts reflect how commonly each
    # species is observed rather than where the cap lands
    print("counting observations per species...")
    counts = (
        candidates.group_by("taxon_id")
        .agg(pl.len().alias("observations"))
        .sort("observations", descending=True)
        .collect()
    )
    if max_species:
        counts = counts.head(max_species)
    print(f"keeping {len(counts)} species")

    plant_pics = (
        candidates.join(counts.lazy(), on="taxon_id", how="inner")
        .group_by("taxon_id")
        .head(max_per_species)
        # position is 1 everywhere after the filter above, so it carries nothing
        .select(PIC_COLUMNS)
    )

    out_file = data_dir / f"{out_prefix}_pics.tsv"
    plant_pics.sink_csv(out_file, separator="\t", engine="streaming")

    n_rows = pl.scan_csv(out_file, separator="\t").select(pl.len()).collect().item()
    print(f"wrote {n_rows} rows to {out_file}")

    # the species the photos actually cover, rather than every plant in the
    # archive, most commonly observed first
    species = (
        pl.scan_csv(out_file, separator="\t")
        .select(TAXA_COLUMNS + ["observations"])
        .unique("taxon_id")
        .sort("observations", descending=True)
        .collect()
    )
    taxa_file = data_dir / f"{out_prefix}_taxa.tsv"
    species.write_csv(taxa_file, separator="\t")
    print(f"wrote {len(species)} species to {taxa_file}")


if __name__ == "__main__":
    app()
