# plant-pics

Builds image datasets of plants from the [iNaturalist open data](https://github.com/inaturalist/inaturalist-open-data)
archive. The filter defaults to a New England bounding box, but nothing in the
pipeline is region-specific.

## Pipeline

`build-dataset.sh` runs the whole sequence. The steps are:

1. **Download the metadata.** The archive lives in a public S3 bucket and is
   refreshed monthly:

        aws s3 cp s3://inaturalist-open-data/taxa.csv.gz data/ --no-sign-request
        aws s3 cp s3://inaturalist-open-data/observations.csv.gz data/ --no-sign-request
        aws s3 cp s3://inaturalist-open-data/photos.csv.gz data/ --no-sign-request

2. **Filter to plants inside the bounding box:**

        uv run src/plant_pics/filter.py data --out-prefix ne \
            --lat-min 41 --lat-max 48 --lon-min -74 --lon-max -67

   `data` is the directory holding the downloaded `*.csv.gz` files; the filter
   writes `<prefix>_taxa.tsv` and `<prefix>_pics.tsv` back into it, so the call
   above produces `data/ne_taxa.tsv` and `data/ne_pics.tsv`. The prefix defaults
   to `plant`.

3. **Download the images:**

        uv run src/plant_pics/download_pics.py data/ne_pics.tsv

4. **Annotate them** with `vlm-toolkit`, which sorts each photo into nature /
   human / magnified / manmade / other:

        uv run vlm process ./large \
            --prompt-file ./prompts/inat_classify.txt \
            --model Qwen/Qwen3.5-9B \
            --output ne_plant_classes.jsonl

5. **Publish the labelled photos** as their own dataset:

        uv run vlm upload-dataset ne_plant_classes.jsonl blambert/ne_plant_classes \
            --card cards/ne_plant_classes.md

6. **Package the `nature` photos into a dataset:**

        uv run src/plant_pics/build_dataset.py ne_plant_classes.jsonl data/ne_pics.tsv \
            --target blambert/ne_plant_photos --card cards/ne_plant_photos.md

   Takes the `vlm process` output, keeps the images it labelled `nature`
   (`--label` picks a different class), joins each one to its row in the filter
   output on the photo id its filename carries, and pushes the result to the
   Hub. `data/ne_pics.tsv` is the only metadata source, so anything the dataset
   should carry belongs in the filter's `PIC_COLUMNS`.

## Prompts

`prompts/` holds the VLM prompts the annotation step runs, copied out of
`vlm-toolkit` so they can be edited here:

- `prompts/inat_classify.txt` - sorts a photo into nature / human / magnified /
  manmade / other

## Dataset Cards

`cards/` holds one card per dataset produced here, named after the dataset:

- `cards/ne_plant_classes.md` - photographs labelled by subject kind
- `cards/ne_plant_photos.md` - the `nature` photos, with taxon and observation metadata

Each card is uploaded to the Hugging Face Hub as that dataset's `README.md`,
which is why the files carry Hub YAML frontmatter.

# Notes

## Archive Size

As of 1/1/2026:

| File               | Rows        |
| ------------------ | ----------- |
| taxa.csv.gz        | 1,615,611   |
| observations.csv.gz| 226,862,366 |
| photos.csv.gz      | 401,313,288 |

416,341 of the taxa rows are plants.

Peek at the raw data without decompressing:

    gzcat taxa.csv.gz | less

## Plant Taxonomy

Taxon 48460 is the root of the tree and 47126 is the Plantae kingdom, so every
plant's ancestry string starts with `48460/47126/`:

    47126	48460	70	kingdom	Plantae	true
