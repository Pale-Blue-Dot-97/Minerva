---
title: 'Minerva: The Remote Sensing Machine Learning Framework'
tags:
  - Python
  - machine learning
  - artificial intelligence
  - remote sensing
authors:
  - name: Harry Baker
    orcid: 0000-0002-4382-8196
    equal-contrib: true
    affiliation: "1, 2"
    corresponding: true
  - name: Jonathon Hare
    orcid: 0000-0003-2921-4283
    equal-contrib: false
    affiliation: "1"
    corresponding: false
  - name: Isabel Sargent
    orcid: 0000-0002-3982-7318
    equal-contrib: false
    affiliation: "1"
    corresponding: false
  - name: Adam Prugel Bennett
    # orcid:
    equal-contrib: false
    affiliation: "1"
    corresponding: false
  - name: Steve Coupland
    # orcid:
    equal-contrib: false
    affiliation: "2"
    corresponding: false
affiliations:
 - name: University of Southampton, University Road, Southampton, UK, SO17 1BJ
   index: 1
 - name: Ordnance Survey, Explorer House, Adanac Drive, Southampton, UK, S016 0AS
   index: 2
date: 06 January 2025
bibliography: paper.bib

---

# Summary

Remote-sensing and earth observation requires its own distictive formats for data from number of bands in imagery, geo-spatial co-ordinates and applications. Most computer vision machine learning research is performed in domains outside of remote sensing and thus, many of the existing libraries in Python for ML lack the features we need. Thankfully, the `torchgeo` [@Stewart_TorchGeo_2022] library extends the popular `torch` library for remote sensing researchers. `minerva` takes this functionality further introducing a framework for researchers to design and execute remote sensing focussed machine learning experiments at scale. `minerva` includes support for `hydra` for experiment configuration, `wandb` logging support, use of high performance computing via `SLURM`, along with numerous other QoL utilities.

# Statement of need

The `minerva` package is primarily designed for use in training, validating and testing machine learning models in the remote sensing domain. Using `hydra`, experiments are configured via `YAML` files that `minerva` interprets, allowing users a great degree of flexibility. It was orginally conceived in 2021 when it was found that `pytorch-lightning` did not offer the level of flexibility required for our particular use-case. Since then, `minerva` has grown from a PhD research repositry into a fully-fledged package -- albeit still in its beta infancy -- with regular users.

# Package Structure

+-------------------+--------------+-----------------------------------------------+
| Sub-Package       | Module       | Description                                   |
|                   |              |                                               |
+:=================:+:============:+:=============================================:+
| datasets          | collators    | Collation functions designed for `minerva`    |
|                   +--------------+-----------------------------------------------+
|                   | dfc          | Implementation of DFC2020 competition dataset |
|                   +--------------+-----------------------------------------------+
|                   | factory      | Functionality for constructing                |
|                   |              | datasets and `DataLoader` in `minerva`        |
|                   +--------------+-----------------------------------------------+
|                   | naip         | Adapted version of `torchgeo` `NAIP` dataset  |
|                   |              | that works with the NAIP files provided in    |
|                   |              | the ChesapeakeCVPR dataset                    |
|                   +--------------+-----------------------------------------------+
|                   | paired       | Datasets to handle paired sampling            |
|                   |              | for use in Siamese learning                   |
|                   +--------------+-----------------------------------------------+
|                   | ssl4eos12    | Simple adaption of the `torchgeo` Sentinel2   |
|                   |              | dataset for use with the SSL4EO-S12 dataset   |
|                   +--------------+-----------------------------------------------+
|                   | utils        | Utility functions for datasets in `minerva`   |
+-------------------+--------------+-----------------------------------------------+
| logger            | steplog      | Loggers to handle the logging                 |
|                   |              | from each step of a task                      |
|                   +--------------+-----------------------------------------------+
|                   | tasklog      | Loggers designed to handle the logging        |
|                   |              | and analysis for a whole task                 |
+-------------------+--------------+-----------------------------------------------+
| models            | core         | Core utility functions and abstract classes   |
|                   |              | underpinning `models`                         |
|                   +--------------+-----------------------------------------------+
|                   | fcn          | Fully Convolutional Network (FCN) models      |
|                   +--------------+-----------------------------------------------+
|                   | psp          | Pyramid Spatial Pooling Net (PSPNet) adapted  |
|                   |              | for use in `minerva`                          |
|                   +--------------+-----------------------------------------------+
|                   | resnet       | ResNets adapted for use in `minerva`          |
|                   +--------------+-----------------------------------------------+
|                   | siamese      | Siamese models adapted for use in `minerva`   |
|                   +--------------+-----------------------------------------------+
|                   | unet         | Module containing UNet models. Most code      |
|                   |              | from https://github.com/milesial/Pytorch-UNet |
+-------------------+--------------+-----------------------------------------------+
| tasks             | core         | Core functionality of `tasks`, defining the   |
|                   |              | abstract `MinervaTask` class                  |
|                   +--------------+-----------------------------------------------+
|                   | epoch        | Standard epoch style for use with generic     | 
|                   |              | model fitting                                 |
|                   +--------------+-----------------------------------------------+
|                   | knn          | K-Nearest Neighbour (KNN) validation task     |
|                   +--------------+-----------------------------------------------+
|                   | tsne         | TSNE clustering task                          |
+-------------------+--------------+-----------------------------------------------+
| utils             | config_load  | Handles the loading of config files           | 
|                   |              | and checking paths                            |
|                   +--------------+-----------------------------------------------+
|                   | runner       | Generic functionality for running `minerva`   |
|                   |              | scripts, setting up distributed computing,    |
|                   |              | handling SLURM variables and                  |
|                   |              | Weights and Biases logging                    |
|                   +--------------+-----------------------------------------------+
|                   | utils        | Utility functions                             |
|                   +--------------+-----------------------------------------------+
|                   | visutils     | Visualisation utility functionality           |
+-------------------+--------------+-----------------------------------------------+
| _root_            | loss         | Specialised loss functions for `minerva`      |
|                   +--------------+-----------------------------------------------+
|                   | modelio      | Standarised functions to handle various IO    |
|                   |              | structures from `dataloaders` and to models   |
|                   +--------------+-----------------------------------------------+
|                   | optimisers   | Custom `torch` optimisers. Consists soley of  |
|                   |              | adapted LARS optimiser from Noah Golmant      |
|                   +--------------+-----------------------------------------------+
|                   | pytorchtools | `EarlyStopping` functionality to track when   |
|                   |              | the training of a model should stop. By       |
|                   |              | Bjarte Mehus Sunde                            |
|                   +--------------+-----------------------------------------------+
|                   | samplers     | Custom samplers for `torchgeo` datasets       |
|                   +--------------+-----------------------------------------------+
|                   | trainer      | Module containing `Trainer` that handles      | 
|                   |              | the fitting of models                         |
|                   +--------------+-----------------------------------------------+
|                   | transforms   | Custom transforms to handle multi-spectral    | 
|                   |              | imagery and geospatial data                   |
+===================+==============+===============================================+
| Footer                                                                           |
+===================+==============+===============================================+

# Comparison to Similar Projects

Given the rapid expansion and advancement of machine learning research since 2014, it will not surprise the reader that there are a wide variety of open-source libraries that support ML practitioners. However, with regards to the remote-sensing focussed researcher, there is a far smaller selection. The stand out package, which `minerva` heavily relies on, is `torchgeo` [@Stewart_TorchGeo_2022]. Like `minerva`, `torchgeo` has matured significantly over the last few years to become an invaulable tool for remote-sensing AI researchers. Its stand out features include its native support for handling GeoTiffs and geospatial information, making it effortless for a user to currate and manipulate datasets to train a remote-sensing focussed model on. `torchgeo.datamodule` also offers much of the same framework features `minerva` does but takes a slightly different approach as to how an experiment is defined.

`minerva` also bears similarities to `pytorch-lightning` in its internal structure. Like `pytorch-lightning`, the internal workings of performing each step of model training is abstracted away in `minerva` from a user. The major difference between the libraries (other than the former's far superior stability and maturity) is `minerva`'s focus on configuring experiments via `YAML` configuration. This stems largely from `minerva`'s raison d'etre -- to act as a framework to facilitate research experiments. As such, `minerva` does lack the same flexibility that `pytorch-lightning` offers its users.

# Conclusion

# Acknowledgements

This work was possible thanks to a PhD funded by the Ordnance Survey. Thanks must also go to my supervisors, Prof. Jonathon Hare, Dr. Isabel Sargent, Prof. Adam Prugel-Bennett and Steve Coupland, whose guidance and support have contributed to the creation of `minerva` and its associated work. Contributions to `minerva` were also made by several people past and present at the Ordnance Survey not listed above as authors:

* Jo Walsh
* Navid Rahimi
* Joe Guyatt
* Ben Dickens
* Kitty Varghese

# References
