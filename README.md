# Minerva

![GitHub release (latest by date)](https://img.shields.io/github/v/release/Pale-Blue-Dot-97/Minerva?style=for-the-badge) ![GitHub](https://img.shields.io/github/license/Pale-Blue-Dot-97/Minerva?style=for-the-badge) ![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/Pale-Blue-Dot-97/Minerva?style=for-the-badge)  ![GitHub contributors](https://img.shields.io/github/contributors/Pale-Blue-Dot-97/Minerva?style=for-the-badge) ![Snyk Vulnerabilities for GitHub Repo](https://img.shields.io/snyk/vulnerabilities/github/Pale-Blue-Dot-97/Minerva?style=for-the-badge) ![Ubuntu-Py310](https://github.com/Pale-Blue-Dot-97/Minerva/actions/workflows/ubuntu_tests_310.yml/badge.svg) ![Ubuntu-Py39](https://github.com/Pale-Blue-Dot-97/Minerva/actions/workflows/ubuntu_tests_39.yml/badge.svg) ![Ubuntu-Py38](https://github.com/Pale-Blue-Dot-97/Minerva/actions/workflows/ubuntu_tests_38.yml/badge.svg) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="docs/images/Minerva_logo_v3.png" alt="Logo" width="400" height="300">
  </a>
  <p align="center">
    Framework for machine learning in remote sensing
    <br />
    <a href="https://pale-blue-dot-97.github.io/Minerva/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Pale-Blue-Dot-97/Minerva/issues">Report Bug</a>
    ·
    <a href="https://github.com/Pale-Blue-Dot-97/Minerva/issues">Request Feature</a>
  </p>
</div>

## About

Minerva is a package to aid in the building, fitting and testing of neural network models on geo-spatial
rasterised land cover data.

## Getting Started

If one wishes to use [torchgeo](https://pypi.org/project/torchgeo/), installation on Linux is recommended to handle the
compilation of the required C-based libraries.

### Installation

The recommended installation order is to start with a fresh `conda` environment, specifying the `python`
version and installing `pytorch` upon environment creation:

```shell
conda create env --name minerva-310 python=3.10 pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

Clone `minerva` from the git repo.

```shell
git clone https://github.com/Pale-Blue-Dot-97/Minerva.git
```

Then install `minerva` via `pip` in editable mode from within the root level of the cloned repositry:

```shell
pip install -e ./
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Requirements

`minerva` now supports the use of [torchgeo](https://torchgeo.readthedocs.io/en/latest/)
datasets with upcoming support for [torchvision](https://pytorch.org/vision/stable/index.html) datasets.

Required Python modules for `minerva` are stated in `requirements.txt`.

`minerva` currently only supports `python` 3.8 -- 3.10.

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

The core functionality of `minerva` provides the modules to define `models` to fit and test, `loaders` to pre-process,
load and parse data, and a `Trainer` to handle all aspects of a model fitting. Below is a MWE of creating datasets,
initialising a Trainer and model, and fitting and testing that model then outputting the results:

### MWE Driver Script

```python
from minerva.utils import config  # Module containing various utility functions.
from minerva.trainer import Trainer  # Class designed to handle fitting of model.


# Initialise a Trainer. Also creates the model.
trainer = Trainer(**config)

# Run the fitting (train and validation epochs).
trainer.fit()

# Run the testing epoch and output results.
trainer.test()
```

See `minerva\bin\MinervaExp.py` as an example script implementing `minerva`.

### Config Structure

See `inbuilt_cfgs\example_config.yml` as an example config file.

### Creating a Manifest for your Dataset

Use `minerva\bin\ManifestMake.py` to construct a manifest to act as a look-up table for a dataset.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Minerva is distributed under a [GNU GPLv3 License](https://choosealicense.com/licenses/gpl-3.0/).

<p align="right">(<a href="#top">back to top</a>)</p>

## Authors

Created by Harry Baker as part of a project towards for a PhD in Computer Science from the
University of Southampton. Funded by the Ordnance Survey Ltd.

Contributions also provided by:

- [Jo Walsh](https://github.com/metazool)
- [Navid Rahimi](https://github.com/NavidCOMSC)
- [Isabel Sargent](https://github.com/PenguinJunk)
- [Steve Coupland](https://github.com/scoupland-os)
- [Joe Guyatt](https://github.com/joeguyatt97)

## Acknowledgments

I'd like to acknowledge the invaluable supervision and contributions of Prof Jonathon Hare and
Dr Isabel Sargent towards this work.

Credit for new `minerva` logo goes to [DALL-E 2](https://openai.com/dall-e-2/).

<p align="right">(<a href="#top">back to top</a>)</p>

## Project Status

This project is still very much a WIP alpha state.

<p align="right">(<a href="#top">back to top</a>)</p>
