import polars as pl


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
    # plant_taxa.collect().write_csv("data/plant_taxa.tsv", separator="\t")

    print("processing observations...")
    # this only keeps memory low if we give it the uncompressed version?
    obs_df = (
        pl.scan_csv(
            "data/observations.csv",
            separator="\t",
            low_memory=True,
        )
        .filter(pl.col("quality_grade") == "research")
        .filter((pl.col("latitude") > 41) & (pl.col("latitude") < 45))
        .filter((pl.col("longitude") > -73) & (pl.col("longitude") < -70))
        .select(["taxon_id", "observation_uuid"])
    )
    plant_obs = obs_df.join(plant_taxa, on="taxon_id", how="inner")

    print("processing photos...")
    photos_df = (
        pl.scan_csv(
            "data/photos.csv",
            separator="\t",
            low_memory=True,
        )
        .filter(~pl.col("license").str.contains("ND"))
        .filter((pl.col("width") > 1000) & (pl.col("height") > 1000))
        .filter(pl.col("position") == 1)
    )
    columns = ["taxon_id", "observation_uuid", "name", "photo_id", "extension"]
    plant_pics = photos_df.select(columns)
    plant_pics = (
        plant_obs.join(photos_df, on="observation_uuid", how="inner")
        .group_by("taxon_id")
        .head(10)
    )

    plant_pics.sink_csv("data/plant_pics.tsv", separator="\t", engine="streaming")


if __name__ == "__main__":
    main()
