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
- Whole images resized to 224x224 with no cropping, since the cue for a label
  (a ruler, a hand) is often near the edge of the frame. Training adds a
  random horizontal flip.
- 10 epochs at batch size 64 and learning rate 2e-5, in bf16 mixed precision.
- Evaluated on the validation split every 5% of training, keeping the
  checkpoint with the best macro F1. Accuracy would reward always predicting
  `nature`, which alone scores 75%.

Training code is `train_classifier.py` in
[plant-pics](https://github.com/belambert/plant-pics).

## Results

Scores on the 30,384-photo test split. The best checkpoint came from epoch 3
(step 4,278 of 14,250), with a validation macro F1 of 0.887.

| metric          | score |
|-----------------|------:|
| accuracy        | 0.984 |
| macro F1        | 0.902 |
| macro precision | 0.913 |
| macro recall    | 0.891 |
| loss            | 0.081 |

Per label:

| label     | precision | recall |    F1 | support |
|-----------|----------:|-------:|------:|--------:|
| nature    |     0.990 |  0.991 | 0.990 |  22,839 |
| human     |     0.982 |  0.985 | 0.984 |   6,740 |
| manmade   |     0.744 |  0.686 | 0.714 |     462 |
| magnified |     0.936 |  0.901 | 0.918 |     343 |

Confusion matrix, with true labels as rows and predictions as columns:

| true \ predicted | nature | human | manmade | magnified |
|------------------|-------:|------:|--------:|----------:|
| nature           | 22,629 |   103 |      87 |        20 |
| human            |     78 | 6,642 |      20 |         0 |
| manmade          |    129 |    15 |     317 |         1 |
| magnified        |     30 |     2 |       2 |       309 |

Most errors involve `nature`. Over a quarter of `manmade` photos (129 of 462)
are predicted as `nature`, and `nature` photos account for 87 of the 109 false
`manmade` predictions.

## Limitations

- Test scores measure agreement with Qwen's labels on held-out photos, not
  accuracy against ground truth.
- `manmade` and `magnified` have under 2,500 examples each. `manmade` is by
  far the weakest class (F1 0.714), and roughly 3 in 10 manmade photos slip
  through as `nature`.
- All training photos come from New England iNaturalist observations; photos
  from other regions, sources, or camera setups may look different enough to
  hurt accuracy.

## Licensing

The weights are released under Apache 2.0, matching the base model. The
training photographs keep their iNaturalist licenses - mostly CC-BY-NC, with
CC-BY, CC0, CC-BY-NC-SA and CC-BY-SA making up the rest - and are not
redistributed here.
