PREFIX=ne

# download the metadata files from inaturalist
mkdir -p data
aws s3 cp s3://inaturalist-open-data/taxa.csv.gz data/ --no-sign-request
aws s3 cp s3://inaturalist-open-data/observations.csv.gz data/ --no-sign-request
aws s3 cp s3://inaturalist-open-data/photos.csv.gz data/ --no-sign-request

# do the filtering (bounds cover NE from NYC to New Brunswick)
uv run plant-filter data --out-prefix "$PREFIX" \
    --lat-min 41 --lat-max 48 --lon-min -74 --lon-max -67 --max-per-species 100

# download images
uv run plant-download "data/${PREFIX}_pics.tsv"

# use vlm-toolkit to annotate the images
uv run vlm process ./large \
    --prompt-file ./prompts/inat_classify.txt \
    --model Qwen/Qwen3.5-9B \
    --output ne_plant_classes.jsonl

# upload the labelled images as a dataset to HF...
uv run vlm upload-dataset ne_plant_classes.jsonl blambert/ne_plant_classes \
    --card cards/ne_plant_classes.md

# ...and the nature ones, with their taxon metadata attached
uv run plant-build-dataset ne_plant_classes.jsonl "data/${PREFIX}_pics.tsv" \
    --target blambert/ne_plant_photos --card cards/ne_plant_photos.md



# optional: train and build a classifier to annotate more images
