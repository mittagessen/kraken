---
authors:
- affiliation: "\xC9cole Pratique des Hautes \xC9tudes, PSL University"
  name: Benjamin Kiessling
base_model:
- https://doi.org/10.5281/zenodo.7050296
datasets:
- https://github.com/OpenITI/arabic_print_data.git
id: 10.5281/zenodo.14585602
language:
- urd
license: Apache-2.0
metrics:
  cer: 4.13
model_type:
- recognition
script:
- Arab
software_hints:
- segmentation=baseline
- version>=2.0
software_name: kraken
summary: Printed Urdu Base Model Trained on the OpenITI Corpus
tags:
- automatic-text-recognition
---
# Printed Urdu Base Model Trained on the OpenITI Corpus

This is a text recognition model trained on the OpenITI dataset of printed
Arabic-script text available [here](https://github.com/OpenITI/arabic_print_data.git) in its state of 2022-09-03. It encompasses
Urdu (~11k lines) material in a variety of typefaces. The model has been
obtained by fine-tuning the [Arabic-script base model](https://doi.org/10.5281/zenodo.7050296) on the purely Urdu
subset of the corpus. 

 The ground truth was lightly normalized to NFD but is otherwise untouched.

## Architecture

The default model architecture and hyperparameters of kraken 4.x where used.

## Uses

The model is trained on a variety of highly diverse typefaces it is mostly
intended as a base model for fine-tuning more specific models from it. In line
with this it has not been extensively verified or optimized.

## How to Get Started with the Model

Follow the instructions on installing and using kraken from the
[website](https://kraken.re).

#### Metrics

CER: 4.13%
