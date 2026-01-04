import polars as pl

from inat_data.filter_observations import lazy_len


def main():

    # load the taxa data
    taxa = pl.scan_csv(
        "data/taxa.csv.gz",
        separator="\t",
        # these two flags needed to prevent errors
        schema_overrides={"rank_level": pl.Float64},
        quote_char=None,
    )
    # filter to just the active, plant, species
    plant_species = taxa.filter(
        (pl.col("active") == True)
        & (pl.col("ancestry").str.starts_with("48460/47126/"))  # plants
        & (pl.col("rank") == "species")
    )
    print(f"Num of active plant species: {lazy_len(plant_species):,}")
    print("sample taxa:")
    print(plant_species.head(10).collect())
    plant_species.collect().write_csv("data/plant_taxa.csv.gz", separator="\t")

    plant_species = pl.scan_csv("data/plant_taxa.csv", separator="\t")

    return

    # load the observations and peek at them
    obs_df = pl.scan_csv("data/observations.csv.gz", separator="\t")
    obs_df = obs_df.select(["observation_uuid", "taxon_id", "quality_grade"])
    obs_df = obs_df.filter(pl.col("quality_grade") == "research")
    obs_df = obs_df.limit(1000)
    # print(f"Total num obs: {lazy_len(obs_df):,}")
    # print("sample observations:")
    # print(obs_df.head(10).collect())

    # join plants with observations
    plant_obs = obs_df.join(plant_species, on="taxon_id", how="inner")
    # print(f"Num of plant obs: {lazy_len(plant_obs):,}")
    # print("sample plant observations:")
    # plant_obs = plant_obs.select(
    #     ["observation_uuid", "latitude", "longitude", "taxon_id", "name"]
    # )
    # print(plant_obs.head(10).collect())

    plant_obs.collect(engine="streaming").write_csv(
        "data/plant_observations.csv.gz", separator="\t"
    )

    return

    plant_obs = pl.scan_csv("data/plant_observations.tsv.gz", separator="\t")

    # load photos
    photos_df = pl.scan_csv("~/inat/photos.csv.gz", separator="\t")
    print("sample photos:")
    print(f"Num photos: {lazy_len(photos_df):,}")
    print(photos_df.head(10).collect())

    # join photos with plant obsevations
    plant_pics = plant_obs.join(photos_df, on="observation_uuid", how="inner")
    print("sample plant pics:")
    print(f"Num plant pics: {lazy_len(plant_pics):,}")
    print(plant_pics.head(10).collect())

    plant_pics.write_csv("plant_pics.tsv.gz", separator="\t")


if __name__ == "__main__":
    main()
