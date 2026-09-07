---
license: other
license_name: mixed-creative-commons
license_link: https://github.com/inaturalist/inaturalist-open-data
task_categories:
- image-classification
size_categories:
- 100K<n<1M
---

# NE Plant Photos

Plant photographs from the [iNaturalist open data](https://github.com/inaturalist/inaturalist-open-data)
archive, filtered to a New England bounding box (latitude 41 to 48, longitude
-74 to -67), each labelled with the species it records and where it was
observed.

This is the `nature` portion of
[blambert/ne_plant_classes](https://huggingface.co/datasets/blambert/ne_plant_classes) -
photos a vision language model judged to show a plant in the field, rather than
a person, a microscope image, or something manmade - so photos of people and
indoor shots are largely gone, but the labelling is model output and has not
been checked against a hand-labelled sample.

114,054 images in a single `train` split, covering the species in the source
archive at up to 100 photos per species.

## Columns

- `image` - the photograph, embedded in the dataset
- `file_name` - the original filename, named after the iNaturalist photo id
- `photo_id`, `photo_uuid`, `extension` - iNaturalist photo identifiers
- `license`, `observer_id` - who took the photo and on what terms
- `observation_uuid`, `observed_on`, `latitude`, `longitude` - the observation
  the photo belongs to
- `taxon_id`, `name`, `ancestry` - the iNaturalist taxon id, its scientific
  name, and the `/`-separated ancestor taxon ids up to the root

Every row is a research-grade observation of an active species-rank taxon under
Plantae, with the photo at least 750 pixels on each side.

## Licensing

The photographs keep whichever license each iNaturalist observer chose, so the
collection is not under a single license and most of it is non-commercial:
CC-BY-NC, CC-BY, CC0, CC-BY-NC-SA and CC-BY-SA, with the exact terms per photo
in the `license` column. Everything but the CC0 portion requires attribution.

## Provenance

Built by [plant-pics](https://github.com/belambert/plant-pics): the archive is
filtered with `filter.py`, the images downloaded with `download_pics.py`,
labelled by running `vlm process` with `prompts/inat_classify.txt` over them,
and assembled here by `build_dataset.py`, which keeps the `nature` label and
joins the filter's metadata back onto each image.
