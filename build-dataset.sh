# download the metadata files from inaturalist
mkdir -p data
aws s3 cp s3://inaturalist-open-data/taxa.csv.gz data/ --no-sign-request
aws s3 cp s3://inaturalist-open-data/observations.csv.gz data/ --no-sign-request
aws s3 cp s3://inaturalist-open-data/photos.csv.gz data/ --no-sign-request

# do the filtering
uv run src/plant_pics/filter.py

# download images
uv run src/plant_pics/download_pics.py data/plant_pics.tsv

# upload to an unannotated dataset on HF?


# use a LM to annotate (some of?) the imgs
# use vlm-tools...


# upload as dataset to HF...







# train and build a classifier to annotate the rest of the images?


# re-upload the clean images?




# get common names (need LM or LM API)

# (optional: do image classification)



# package into a dataset
