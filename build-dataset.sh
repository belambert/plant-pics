


# download the metadata files from inaturalist

aws s3 cp s3://inaturalist-open-data/taxa.csv.gz . --no-sign-request
# aws s3 cp s3://inaturalist-open-data/observations.csv.gz . --no-sign-request
# aws s3 cp s3://inaturalist-open-data/photos.csv.gz . --no-sign-request

# do the filtering

# get common names (need LM or LM API)

# download images

# (optional: do image classification)



# package into a dataset