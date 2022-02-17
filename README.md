# Minerva

![GitHub release (latest by date)](https://img.shields.io/github/v/release/Pale-Blue-Dot-97/Minerva) ![GitHub](https://img.shields.io/github/license/Pale-Blue-Dot-97/Minerva) ![Ubuntu-Py39](https://github.com/Pale-Blue-Dot-97/Minerva/actions/workflows/ubuntu_tests_39.yml/badge.svg) ![Ubuntu-Py38](https://github.com/Pale-Blue-Dot-97/Minerva/actions/workflows/ubuntu_tests_38.yml/badge.svg) ![Ubuntu-Py37](https://github.com/Pale-Blue-Dot-97/Minerva/actions/workflows/ubuntu_tests_37.yml/badge.svg) ![Ubuntu-Py36](https://github.com/Pale-Blue-Dot-97/Minerva/actions/workflows/ubuntu_tests_36.yml/badge.svg)

Minerva is a package to aid in the building, fitting and testing of neural network models on geo-spatial
rasterised land cover data.  

## Installation

If one wishes to use [torchgeo](https://pypi.org/project/torchgeo/), installation on Linux is recommended to handle the
compilation of the required C-based libraries.

The recommended installation order is to start with a fresh `conda` environment, specifying the `python`
version and installing `pytorch` upon environment creation:

```shell
conda create env --name minerva-39 python=3.9 pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Then install `torchgeo` via `pip`:

```shell
pip install torchgeo
```

Then proceed with installing `Minerva`'s remaining requirements:

```shell
pip install tensorflow pandas imageio opencv-python seaborn tabulate torchinfo psutil alive-progress inputimeout
```

The `torchgeo` docs also recommend installing `radiant_mlhub` and `zipfile-deflate64`:

```shell
pip install zipfile-deflate64  radiant_mlhub
```

## Requirements

`Minerva` now supports the use of [torchgeo](https://torchgeo.readthedocs.io/en/latest/)
datasets with upcoming support for [torchvision](https://pytorch.org/vision/stable/index.html) datasets.

Required Python modules for `Minerva` are stated in `requirements.txt`.

## Usage

The core functionality of `Minerva` provides the modules to define `models` to fit and test, `loaders` to pre-process,
load and parse data, and a `Trainer` to handle all aspects of a model fitting. Below is a MWE of creating datasets,
initialising a Trainer and model, and fitting and testing that model then outputting the results:

### MWE Driver Script

```python
from Minerva.utils import utils             # Module containing various utility functions
from Minerva.loaders import make_datasets   # Module to handle the creation of datasets and torch loaders
from Minerva.trainer import Trainer         # Class designed to handle fitting of model

# Specify the path to a YAML master config.
config_path = "../../config/config.yml"

# Loads the master config from file as a dict.
config, _ = utils.load_configs(config_path)

# Get the dataset loaders, number of batches, the distribution of classes and an updated config.
datasets, n_batches, class_dist, new_config = make_datasets(**config)

# Initialise a Trainer. Also creates the model.
trainer = Trainer(loaders=datasets, n_batches=n_batches, class_dist=class_dist, **new_config)

# Run the fitting (train and validation epochs).
trainer.fit()

# Run the testing epoch and output results.
trainer.test()
```

See `Minerva\bin\MinervaExp.py` as an example script implementing `Minerva`.

### Config Structure

See `config\example_config.yml` as an example config file.

### Creating a Manifest for your Dataset

Use `Minerva\bin\ManifestMake.py` to construct a manifest to act as a look-up table for a dataset.

## License

Minerva is distributed under a [GNU GPLv3 License](https://choosealicense.com/licenses/gpl-3.0/).

## Authors

Created by Harry Baker as part of a project towards the award of a PhD Computer Science from the
University of Southampton. Funded by the Ordnance Survey Ltd.

I'd like to acknowledge the invaluable supervision and contributions of Dr Jonathon Hare and
Dr Isabel Sargent towards this work.

## Project Status

This project is still very much a WIP alpha state.
