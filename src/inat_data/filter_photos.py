import polars as pl


def main():

    print("loading observations...")
    plant_obs = pl.scan_csv(
        "data/plant_observations.tsv",
        separator="\t",
        low_memory=True,
    ).select(["taxon_id", "observation_uuid"])

    print("processing photos...")
    photos_df = pl.scan_csv(
        "data/photos.csv",
        separator="\t",
        low_memory=True,
    )
    photos_df = photos_df.select(["observation_uuid", "photo_id", "extension"])
    plant_pics = plant_obs.join(photos_df, on="observation_uuid", how="inner")
    plant_pics.sink_csv("data/plant_pics.tsv", separator="\t", engine="streaming")

    # plant_pics.collect(engine="streaming").write_csv("data/plant_pics.tsv", separator="\t")


if __name__ == "__main__":
    main()
