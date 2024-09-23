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
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true
affiliations:
 - name: University of Southampton, University Road, Southampton, UK, SO17 1BJ
   index: 1
 - name: Ordnance Survey, Explorer House, Adanac Drive, Southampton, UK, S016 0AS
   index: 2
date: 21 September 2024
bibliography: paper.bib

---

# Summary

Remote-sensing and earth observation requires its own distictive formats for
data from number of bands in imagery, geo-spatial co-ordinates and applications.
Most computer vision machine learning research is performed in domains outside
of remote sensing and thus, many of the existing libraries in Python for ML lack
the features we need. Thankfully, the `torchgeo` [@Stewart_TorchGeo_2020] library extends the popular
`torch` library for remote sensing researchers. `minerva` takes this
functionality further introducing a framework for researchers to design and
execute remote sensing focussed machine learning experiments at scale. `minerva`
includes support for `hydra` for experiment configuration, `wandb` logging
support, use of high performance computing via `SLURM`, along with numerous
other QoL utilities.

# Statement of need

The `minerva` package is primarily designed for use in training, validating and
testing machine learning models in the remote sensing domain. Using `hydra`,
experiments are configured via `YAML` files that `minerva` interprets, allowing
users a great degree of flexibility. It was orginally conceived in 2021 when it was found that `pytorch-lightning` did not offer the level of flexibility required for our particular use-case. Since then, `minerva` has grown from a PhD research repositry into a fully-fledged package -- albeit still in its beta infancy -- with regualr users. 

# Package Structure

<!-- +-------------------+------------+----------+----------+
| Sub-Package       | Module     | Description         |
|                   |            |                     |
+:=================:+:==========:+:===================:+
| datasets          |    |                     |
|                   +------------+---------------------+
|                   |            |                     |
|                   +------------+---------------------+
|                   |            | - body              |
|                   |            | - elements          |
|                   |            | - here              |
+===================+============+=====================+
| Footer                                               |
+===================+============+=====================+ -->

# Comparison to Similar Projects

Given the rapid expansion and advancement of machine learning research since 2014, it will not surprise the reader that there are a wide variety of open-source libraries that support ML practitioners. However, with regards to the remote-sensing focussed researcher, there is a far smaller selection. The stand out package, which `minerva` heavily relies on, is `torchgeo` [@Stewart_TorchGeo_2020]. Like `minerva`, `torchgeo` has matured significantly over the last few years to become an invaulable tool for remote-sensing AI researchers. Its stand out features include its native support for handling GeoTiffs and geospatial information, making it effortless for a user to currate and manipulate datasets to train a remote-sensing focussed model on. `torchgeo.datamodule` also offers much of the same framework features `minerva` does but takes a slightly different approach as to how an experiment is defined.

`minerva` also bear similarities to `pytorch-lightning` in its internal structure. Like `pytorch-lightning`, the internal workings of performing each step of model training is abstracted away in `minerva` from a user. The major difference between the libraries (other than the former's far superior stability and maturity) is `minerva`'s focus on configuring experiments via `YAML` configuration. This stems largely from `minerva`'s raison d'etre -- to act as a framework to facilitate research experiments. As such, `minerva` does lack the same flexibility that `pytorch-lightning` offers its users.

# Conclusion

# Citations

# Acknowledgements


# References
