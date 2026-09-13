---
license: apache-2.0
library_name: transformers
pipeline_tag: image-classification
base_model: google/vit-base-patch16-224-in21k
datasets:
- blambert/ne_plant_classes
tags:
- vit
- inaturalist
- plants
---

# NE Plant Classes ViT

A Vision Transformer that sorts plant photographs by the kind of subject they
show: a plant in the field, a hand or person in frame, a microscope or macro
view, or something manmade. It is
[google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
fine-tuned on [blambert/ne_plant_classes](https://huggingface.co/datasets/blambert/ne_plant_classes).

It was built to do cheaply what a vision language model did for that dataset:
separate usable field photos from the rest before assembling a plant dataset
such as [blambert/ne_plant_photos](https://huggingface.co/datasets/blambert/ne_plant_photos).
It does not identify species.

## Labels

| label     | meaning                                         |
|-----------|-------------------------------------------------|
| nature    | a plant in the field                            |
| human     | a hand or person in frame                       |
| magnified | a microscope or macro view                      |
| manmade   | rulers, tags, labs, fences, herbarium sheets    |

The source dataset has a fifth label, `other`, with only 26 examples; it was
dropped before training, so the model never predicts it.

## Usage

```python
from transformers import pipeline

classify = pipeline("image-classification", model="blambert/ne_plant_classes_vit")
classify("photo.jpg")
```

## Training Data

151,919 photographs from the iNaturalist open data archive inside a New England
bounding box (latitude 41 to 48, longitude -74 to -67), split by label:

| label     |   count |  share |
|-----------|--------:|-------:|
| nature    | 114,192 |  75.2% |
| human     |  33,702 |  22.2% |
| manmade   |   2,310 |   1.5% |
| magnified |   1,715 |   1.1% |

The labels are the output of
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), not human
annotation, and have not been checked against a hand-labelled sample. The model
learns to agree with Qwen, mistakes included.

## Training Procedure

- Shuffled with seed 42, then split 60% train / 20% validation / 20% test,
  stratified by label.
- Images augmented with a random resized crop and horizontal flip for training;
  resized and center-cropped for evaluation.
- 10 epochs at batch size 64 and learning rate 2e-5, in bf16 mixed precision.
- Evaluated on the validation split every 5% of training, keeping the
  checkpoint with the best macro F1. Accuracy would reward always predicting
  `nature`, which alone scores 75%.

Training code is `train_classifier.py` in
[plant-pics](https://github.com/belambert/plant-pics).

## Results

<!-- fill in from test_results.json and test_per_label.json after training -->

## Limitations

- Test scores measure agreement with Qwen's labels on held-out photos, not
  accuracy against ground truth.
- `manmade` and `magnified` have under 2,500 examples each, so expect them to
  be the weakest classes.
- All training photos come from New England iNaturalist observations; photos
  from other regions, sources, or camera setups may look different enough to
  hurt accuracy.

## Licensing

The weights are released under Apache 2.0, matching the base model. The
training photographs keep their iNaturalist licenses - mostly CC-BY-NC, with
CC-BY, CC0, CC-BY-NC-SA and CC-BY-SA making up the rest - and are not
redistributed here.
