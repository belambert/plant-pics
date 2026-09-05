import json
import sys
import polars as pl

data = pl.read_csv(
    sys.argv[1],
    separator="\t",
    low_memory=True,
)

names = sorted(data["name"].unique().to_list())
for name in names:
    dict_ = {"prompt": name}
    print(json.dumps(dict_))

