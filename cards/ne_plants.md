---
license: cc-by-4.0
task_categories:
- image-classification
---

# NE Plants

Photographs from a New England plants collection, each labelled by a vision
language model with the kind of subject it depicts.

## Columns

- `image` - the photograph, embedded in the dataset
- `output` - the model's one-word classification
- `file_name` - the original image filename

## Labels

| label      | count |
|------------|------:|
| nature     |  1360 |
| human      |   171 |
| magnified  |    14 |
| manmade    |     3 |

## Provenance

Generated with `vlm-process` from the [vlm-toolkit](https://github.com/belambert/vlm-toolkit)
repo and uploaded with its `upload-dataset` command.
