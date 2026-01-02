import polars as pl


def main():

    # load the observations and peek at them
    obs = (
        # pl.read_csv("data/observations.csv.gz", separator="\t")
        pl.scan_csv("data/observations.csv.gz", separator="\t", low_memory=True)
        .filter(pl.col("quality_grade") == "research")
        .select(["taxon_id", "observation_uuid"])
        # .collect(engine="streaming")
        # .sink_csv("data/res_observations.csv.gz")
        # .sink_csv("data/res_observations.csv.gz", optimizations=pl.QueryOptFlags.none())
    )
    # obs.collect(engine="streaming").write_csv("data/res_observations.csv")
    obs.collect(engine="streaming").write_csv("data/res_observations.csv")

    # obs.sink_csv("data/res_observations.csv")

    # obs = obs.head(10)
    # print(obs.collect(streaming=True)) # this is faster
    # print(obs.collect(streaming=False))
    # print(obs.explain(optimized=False))
    # print(obs.explain(optimized=True))
    # print(obs.explain())


if __name__ == "__main__":
    main()
