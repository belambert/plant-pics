import polars as pl

MIN_RESOLUTION = 750
# covers all of NE from NYC to New Brunswick
LAT_MIN, LAT_MAX = 41, 48
LON_MIN, LON_MAX = -74, -67

def main():

    # load the taxa data
    taxa = pl.scan_csv(
        "data/taxa.csv.gz",
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
    plant_taxa = plant_taxa.select(["taxon_id", "name"])
    plant_taxa.collect().write_csv("data/plant_taxa.tsv", separator="\t")

    print("processing observations...")
    obs_df = (
        pl.scan_csv(
            "data/observations.csv.gz",
            separator="\t",
            low_memory=True,
        )
        .filter(pl.col("quality_grade") == "research")
        .filter((pl.col("latitude") > LAT_MIN) & (pl.col("latitude") < LAT_MAX))
        .filter((pl.col("longitude") > LON_MIN) & (pl.col("longitude") < LON_MAX))
        .select(["taxon_id", "observation_uuid"])
    )
    plant_obs = obs_df.join(plant_taxa, on="taxon_id", how="inner")

    print("processing photos...")
    photos_df = (
        pl.scan_csv(
            "data/photos.csv.gz",
            separator="\t",
            low_memory=True,
        )
        # don't include those that don't allow derivates
        .filter(~pl.col("license").str.contains("ND"))
        .filter((pl.col("width") >= MIN_RESOLUTION) & (pl.col("height") >= MIN_RESOLUTION))
        .filter(pl.col("position") == 1)
    )
    columns = ["taxon_id", "observation_uuid", "name", "photo_id", "extension"]
    plant_pics = photos_df.select(columns)

    # get ~20k with only 10 per species
    # 862k with unlimited per species
    # 152k with 100 per species
    plant_pics = (
        plant_obs.join(photos_df, on="observation_uuid", how="inner")
        .group_by("taxon_id")
        .head(100)
    )

    plant_pics.sink_csv("data/plant_pics.tsv", separator="\t", engine="streaming")

    n_rows = pl.scan_csv("data/plant_pics.tsv", separator="\t").select(pl.len()).collect().item()
    print(f"wrote {n_rows} rows")


if __name__ == "__main__":
    main()
