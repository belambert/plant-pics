# plant-pics

Builds image datasets of plants from the [iNaturalist open data](https://github.com/inaturalist/inaturalist-open-data)
archive. The filter defaults to a New England bounding box, but nothing in the
pipeline is region-specific.

So the full process would look something like this:
- download the metadata files from inaturalist
- do the filtering:

        uv run src/plant_pics/filter.py data --out-prefix ne \
            --lat-min 41 --lat-max 48 --lon-min -74 --lon-max -67

  `data` is the directory holding the downloaded `*.csv.gz` files; the filter
  writes `<prefix>_taxa.tsv` and `<prefix>_pics.tsv` back into it, so the call
  above produces `data/ne_taxa.tsv` and `data/ne_pics.tsv`. The prefix defaults
  to `plant`.
- get common names (need LM or LM API)
- download images
- (optional: do image classification)
- package into a dataset:

        uv run src/plant_pics/build_dataset.py ne_plant_classes.jsonl data/ne_pics.tsv \
            --target blambert/ne_plants --card cards/ne_plants.md

  Takes the `vlm process` output, keeps the images it labelled `nature`
  (`--label` picks a different class), joins each one to its row in the filter
  output on the photo id its filename carries, and pushes the result to the
  Hub. `data/ne_pics.tsv` is the only metadata source, so anything the dataset
  should carry belongs in the filter's `PIC_COLUMNS`.

`build-dataset.sh` runs the whole sequence.

## Prompts

`prompts/` holds the VLM prompts the annotation step runs, copied out of
`vlm-toolkit` so they can be edited here:

- `prompts/inat_classify.txt` - sorts a photo into nature / human / magnified /
  manmade / other

## Dataset Cards

`cards/` holds one card per dataset produced here, named after the dataset:

- `cards/ne_plant_classes.md` - photographs labelled by subject kind
- `cards/ne_plants.md` - the `nature` photos, with taxon and observation metadata

Each card is uploaded to the Hugging Face Hub as that dataset's `README.md`,
which is why the files carry Hub YAML frontmatter.


Classification of 20k images at size "large" (2600 batches of 8)
~3 hours on a100 = $15

Do the classification on the smaller sized images.  Try thumb (100px) or
small (240px).

 401,313,288 photos.csv

## Get the data

Download raw data from S3 (updated monthly?):

    aws s3 cp s3://inaturalist-open-data/taxa.csv.gz . --no-sign-request
    aws s3 cp s3://inaturalist-open-data/observations.csv.gz . --no-sign-request
    aws s3 cp s3://inaturalist-open-data/photos.csv.gz . --no-sign-request

More information about the data here:
https://github.com/inaturalist/inaturalist-open-data

Peek at the raw data from the command line without decompressing like this:

    gzcat taxa.csv.gz | less


Also need:

    wget https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip

Mass list...:

    wget https://plants.sc.egov.usda.gov/DocumentLibrary/Txt/Massachusetts_NRCS_csv.txt

USDA data

    wget https://plants.sc.egov.usda.gov/DocumentLibrary/Txt/plantlst.txt

    # Download the complete database
    wget https://plants.usda.gov/assets/docs/CompletePLANTSList/plantlst.txt

    # State distribution data
    wget https://plants.usda.gov/assets/docs/CompletePLANTSList/statedownload.txt

    # Characteristics data
    wget https://plants.usda.gov/assets/docs/CompletePLANTSList/characteristics.txt


    wget https://plants.sc.egov.usda.gov/DocumentLibrary/Txt/Massachusetts_NRCS_csv.txt



# Notes

On 1/1/2026

Plantae

48460 is the root of the tree
47126 is Plantae kingdom

The ancestry prefix for all plants should be "48460/47126/"

taxa csv has 1,615,611 rows

416,341 are plants


gzcat observations.csv.gz| wc -l
 226,862,366

gzcat photos.csv.gz| wc -l      
 401,313,288



47126	48460	70	kingdom	Plantae	true


https://github.com/inaturalist/inaturalist-open-data