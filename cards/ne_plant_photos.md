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

![Twelve photos from the dataset, captioned with their species](https://raw.githubusercontent.com/belambert/plant-pics/main/images/ne_plant_photos.jpg)

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

114,075 images in a single `train` split, covering the species in the source
archive at up to 100 photos per species.

![Density map of where the photos were taken across New England](https://raw.githubusercontent.com/belambert/plant-pics/main/images/ne_map.png)

## Label Quality

The `nature` filter is model output, so a few photos here are not field shots
at all: a hand holding a specimen, a ruler beside a plant, an indoor macro
view. A ViT classifier trained on the same labels
([blambert/ne_plant_classes_vit](https://huggingface.co/blambert/ne_plant_classes_vit))
disputes about 0.9% of `nature` labels, and reviewing a sample of those
disagreements suggested the label was wrong about two thirds of the time -
**on the order of 700 photos here, under 1%**. Photos that belong here were
dropped for the same reason, in roughly similar numbers.

Every species label, on the other hand, comes from the iNaturalist archive's
research-grade observations, not from a model.

This is good enough for most uses of a plant photo collection, where a fraction
of a percent of odd images does not change much. If your use needs every image
to be a field photograph, filter again with a model of your own, or hand-check
the sample you use.

## Columns

- `image` - the photograph, embedded in the dataset
- `file_name` - the original filename, named after the iNaturalist photo id
- `photo_id`, `photo_uuid`, `extension` - iNaturalist photo identifiers
- `license` - the terms the photo is published under
- `width`, `height` - the photo's pixel dimensions
- `observed_on`, `latitude`, `longitude` - when and where the observation was made
- `taxon_id`, `name` - the iNaturalist taxon id and its scientific name
- `observations` - how many matching observations the species has, before the
  per-species photo cap

Every row is a research-grade observation of an active species-rank taxon under
Plantae, with the photo at least 750 pixels on each side.

## Licensing

The photographs keep whichever license each iNaturalist observer chose, so the
collection is not under a single license and most of it is non-commercial:
CC-BY-NC, CC-BY, CC0, CC-BY-NC-SA and CC-BY-SA, with the exact terms per photo
in the `license` column. Everything but the CC0 portion requires attribution.
The sample photos on this card are all CC0.

## Provenance

Built by [plant-pics](https://github.com/belambert/plant-pics): the archive is
filtered with `filter.py`, the images downloaded with `download.py`,
labelled by running `vlm process` with `prompts/inat_classify.txt` over them,
and assembled here by `build_dataset.py`, which keeps the `nature` label and
joins the filter's metadata back onto each image.
