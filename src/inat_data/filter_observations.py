import polars as pl


def lazy_len(df) -> int:
    return df.select(pl.len()).collect().item()


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
    plant_taxa = plant_taxa.select(["taxon_id"])
    plant_taxa.collect().write_csv("data/plant_taxa.tsv", separator="\t")

    print("processing observations...")
    # this only keeps memory low if we give it the uncompressed version
    obs_df = (
        pl.scan_csv(
            "data/observations.csv",
            separator="\t",
            low_memory=True,
        )
        .filter(pl.col("quality_grade") == "research")
        # .select(["taxon_id", "observation_uuid"])
    )
    plant_obs = obs_df.join(plant_taxa, on="taxon_id", how="inner")

    plant_obs.sink_csv("data/plant_observations.tsv", separator="\t",engine="streaming")


if __name__ == "__main__":
    main()
