# Minerva

![GitHub release (latest by date)](https://img.shields.io/github/v/release/Pale-Blue-Dot-97/Minerva?)
![GitHub](https://img.shields.io/github/license/Pale-Blue-Dot-97/Minerva?)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/minerva)
![GitHub contributors](https://img.shields.io/github/contributors/Pale-Blue-Dot-97/Minerva?)
[![CodeFactor](https://www.codefactor.io/repository/github/pale-blue-dot-97/minerva/badge)](https://www.codefactor.io/repository/github/pale-blue-dot-97/minerva)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f961aed541494e4db7317bead5f84fef)](https://app.codacy.com/gh/Pale-Blue-Dot-97/Minerva/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
![tests](https://github.com/Pale-Blue-Dot-97/Minerva/actions/workflows/tests.yml/badge.svg)
[![Read the Docs](https://img.shields.io/readthedocs/smp?)](https://pale-blue-dot-97.github.io/Minerva/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Pale-Blue-Dot-97/Minerva/main.svg)](https://results.pre-commit.ci/latest/github/Pale-Blue-Dot-97/Minerva/main)
[![codecov](https://codecov.io/gh/Pale-Blue-Dot-97/Minerva/graph/badge.svg?token=8TUR8A8XZ5)](https://codecov.io/gh/Pale-Blue-Dot-97/Minerva)
[![DOI](https://zenodo.org/badge/333241475.svg)](https://doi.org/10.5281/zenodo.13941041)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Pale-Blue-Dot-97/Minerva">
    <img src="docs/images/Minerva_logo.png" alt="Logo" width="" height="400">
  </a>
  <p align="center">
    <b style="font-size:26px;"> v0.28.0</b>
    <br />
    <a href="https://pale-blue-dot-97.github.io/Minerva/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Pale-Blue-Dot-97/Minerva/issues">Report Bug</a>
    ·
    <a href="https://github.com/Pale-Blue-Dot-97/Minerva/issues">Request Feature</a>
  </p>
</div>

## About 🔎

Minerva is a package to aid in the building, fitting and testing of neural network models on multi-spectral geo-spatial data.

## Getting Started ▶

If one wishes to use [torchgeo](https://pypi.org/project/torchgeo/), installation on Linux is recommended to handle the
compilation of the required C-based libraries -- though `minerva` is also tested with MacOS and Windows runners.

### Installation ⬇

`minerva` is currently not included in any distribution. The recommended install is therefore to install the latest version from `GitHub`.

```shell
pip install git+https://github.com/Pale-Blue-Dot-97/Minerva.git
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Requirements 📌

`minerva` now supports the use of [torchgeo](https://torchgeo.readthedocs.io/en/latest/)
datasets with upcoming support for [torchvision](https://pytorch.org/vision/stable/index.html) datasets.

Required Python modules for `minerva` are stated in the `setup.cfg`.

`minerva` currently only supports `python` 3.10 -- 3.12.

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage 🖥

The core functionality of `minerva` provides the modules to define `models` to fit and test, `loaders` to pre-process,
load and parse data, and a `Trainer` to handle all aspects of a model fitting. Below is a MWE of creating datasets,
initialising a Trainer and model, and fitting and testing that model then outputting the results:

### MWE Driver Script 📄

```python
import hydra
from omegaconf import DictConfig

from minerva.trainer import Trainer  # Class designed to handle fitting of model.


@hydra.main(version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Initialise a Trainer. Also creates the model.
    trainer = Trainer(**cfg)

    # Run the fitting (train and validation epochs).
    trainer.fit()

    # Run the testing epoch and output results.
    trainer.test()
```

See `scripts\MinervaExp.py` as an example script implementing `minerva`.

### Config Structure ⚙

See `minerva\inbuilt_cfgs\example_config.yaml` as an example config file.

### Creating a Manifest for your Dataset 📑

Use `scripts\ManifestMake.py` to construct a manifest to act as a look-up table for a dataset.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing 🤝

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

 1. Fork the Project
 2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
 3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
 4. Push to the Branch (`git push origin feature/AmazingFeature`)
 5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

## License 🔏

Minerva is distributed under a [MIT License](https://choosealicense.com/licenses/mit/).

<p align="right">(<a href="#top">back to top</a>)</p>

## Authors ✒

Created by Harry Baker as part of a project towards for a PhD in Computer Science from the
University of Southampton. Funded by the Ordnance Survey Ltd.

Contributions also provided by:

- [Jo Walsh](https://github.com/metazool)
- [Jonathon Hare](https://github.com/jonhare)
- [Isabel Sargent](https://github.com/PenguinJunk)
- [Navid Rahimi](https://github.com/NavidCOMSC)
- [Steve Coupland](https://github.com/scoupland-os)
- [Joe Guyatt](https://github.com/joeguyatt97)
- [Ben Dickens](https://github.com/BenDickens)
- [Kitty Varghese](https://github.com/kittyvarghese)

## Acknowledgments 📢

I'd like to acknowledge the invaluable supervision and contributions of [Prof Jonathon Hare](https://github.com/jonhare) and
[Dr Isabel Sargent](https://github.com/PenguinJunk) towards this work.

The following modules are adapted from open source third-parites:
| Module | Original Author | License | Link |
|:-------|:----------------|:--------|:-----|
| `pytorchtools` | [Noah Golmant](https://github.com/noahgolmant) | [MIT](https://choosealicense.com/licenses/mit/) | [lars](https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py) |
| `optimisers` | [Bjarte Mehus Sunde](https://github.com/Bjarten) | [MIT](https://choosealicense.com/licenses/mit/) | [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch) |
| `dfc` | [Lukas Liebel](https://github.com/lukasliebel) | [GNU GPL v3.0](https://choosealicense.com/licenses/gpl-3.0/) | [dfc2020_baseline](https://github.com/lukasliebel/dfc2020_baseline/blob/master/code/datasets.py) |

This repositry also contains some small samples from various public datasets for unit testing purposes. These are:

| Dataset | Citation | License | Link |
|:--------|:---------|:--------|:-----|
| ChesapeakeCVPR | Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N, "Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data". Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition (CVPR 2019) | Unknown | [ChesapeakeCVPR](https://lila.science/datasets/chesapeakelandcover) |
| SSL4EO-S12 | Wang Y, Braham N A A, Xiong Z, Liu C, Albrecht C M, Zhu X X, "SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation". arXiv preprint, 2023 | [Apache 2.0](https://github.com/zhu-xlab/SSL4EO-S12/blob/main/LICENSE) | [SSL4E0-S12](https://github.com/zhu-xlab/SSL4EO-S12) |
| DFC2020 | M. Schmitt, L. H. Hughes, C. Qiu, and X. X. Zhu, “SEN12MS – A curated dataset of georeferenced multi-spectral sentinel-1/2 imagery for deep learning and data fusion,” in ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci. IV-2/W7, 2019, pp. 153–160. | [Creative Commons Attribution](https://creativecommons.org/licenses/by/4.0/) | [IEEE DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest#files)
BigEarthNet | G. Sumbul, A. d. Wall, T. Kreuziger, F. Marcelino, H. Costa, P. Benevides, M. Caetano, B. Demir, V. Markl, “BigEarthNet-MM: A Large Scale Multi-Modal Multi-Label Benchmark Archive for Remote Sensing Image Classification and Retrieval”,  IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 3, pp. 174-180, 2021, doi: 10.1109/MGRS.2021.3089174. | [Community Data License Agreement – Permissive, Version 1.0](https://cdla.dev/permissive-1-0/) | [BigEarthNet](https://bigearth.net/)

<p align="right">(<a href="#top">back to top</a>)</p>

## Project Status 🔴🟡🟢

This project is now in release *beta* state. Still expect some bugs and there may be breaking changes in future versions.

<p align="right">(<a href="#top">back to top</a>)</p>
