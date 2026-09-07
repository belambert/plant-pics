---
license: other
license_name: mixed-creative-commons
license_link: https://github.com/inaturalist/inaturalist-open-data
task_categories:
- image-classification
size_categories:
- 100K<n<1M
---

# NE Plant Classes

Plant photographs from the [iNaturalist open data](https://github.com/inaturalist/inaturalist-open-data)
archive, filtered to a New England bounding box (latitude 41 to 48, longitude
-74 to -67) and each labelled by a vision language model with the kind of
subject it depicts.

Labels come from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B);
they are model output, not human annotation, and have not been checked against
a hand-labelled sample.

151,778 images in a single `train` split.

## Columns

- `image` - the photograph, embedded in the dataset
- `output` - the model's one-word classification
- `file_name` - the original image filename, named after the iNaturalist photo id

## Labels

The prompt sorts each photo into one of five classes, falling back to `other`
when none of the first four fit.

| label     |   count |  share |
|-----------|--------:|-------:|
| nature    | 114,054 |  75.1% |
| human     |  33,675 |  22.2% |
| manmade   |   2,308 |   1.5% |
| magnified |   1,715 |   1.1% |
| other     |      26 |  <0.1% |

## Licensing

The photographs keep whichever license each iNaturalist observer chose, so the
collection is not under a single license and most of it is non-commercial.

| license     |   count |  share |
|-------------|--------:|-------:|
| CC-BY-NC    | 114,506 |  75.4% |
| CC-BY       |  22,860 |  15.1% |
| CC0         |  10,073 |   6.6% |
| CC-BY-NC-SA |   3,288 |   2.2% |
| CC-BY-SA    |   1,051 |   0.7% |

Everything but the CC0 portion requires attribution. Observer and photo
metadata live in the iNaturalist open data archive, joinable on the photo id in
`file_name`.

## Provenance

Labelled by running `vlm process` with `prompts/inat_classify.txt` over the
downloaded images, then uploaded with `vlm upload-dataset`. Both commands come
from the [vlm-toolkit](https://github.com/belambert/vlm-toolkit) repo; the
filtering and download steps are in [plant-pics](https://github.com/belambert/plant-pics).
