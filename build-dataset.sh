PREFIX=ne

# download the metadata files from inaturalist
mkdir -p data
aws s3 cp s3://inaturalist-open-data/taxa.csv.gz data/ --no-sign-request
aws s3 cp s3://inaturalist-open-data/observations.csv.gz data/ --no-sign-request
aws s3 cp s3://inaturalist-open-data/photos.csv.gz data/ --no-sign-request

# do the filtering (bounds cover NE from NYC to New Brunswick)
uv run src/plant_pics/filter.py data --out-prefix "$PREFIX" \
    --lat-min 41 --lat-max 48 --lon-min -74 --lon-max -67

# download images
uv run src/plant_pics/download_pics.py "data/${PREFIX}_pics.tsv"

# use vlm-toolkit to annotate the images
uv run vlm-process ./large \
    --prompt-file ./src/imgproc/prompts/inat_classify.txt \
    --model Qwen/Qwen3.5-9B \
    --output ne_plants.json

# upload as dataset to HF...
uv run upload-dataset ne_plants.json belambert/ne_plants

# optional: train and build a classifier to annotate more images


# re-upload the clean images?




# get common names (need LM or LM API)

# (optional: do image classification)



# package into a dataset
