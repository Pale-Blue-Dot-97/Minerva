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
date: 08 August 2024
bibliography: paper.bib

---

# Summary

Remote-sensing and earth observation requires its own distictive formats for
data from number of bands in imagery, geo-spatial co-ordinates and applications.
Most computer vision machine learning research is performed in domains outside
of remote sensing and thus, many of the existing libraries in Python for ML lack
the features we need. Thankfully, the ``torchgeo`` library extends the popular
``torch`` library for remote sensing researchers. ``minerva`` takes this
functionality further introducing a framework for researchers to design and
execute remote sensing focussed machine learning experiments at scale. ``minerva``
includes support for ``hydra`` for experiment configuration, ``wandb`` logging
support, use of high performance computing via ``SLURM``, along with numerous
other QoL utilities.

# Statement of need

The `minerva` package is primarily designed for use in training, validating and
testing machine learning models in the remote sensing domain. Using `hydra`,
experiments are configured via `YAML` files that `minerva` interprets, allowing
users to a great degree of flexibility. Experiments are broken down into tasks.
These represent

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

# Conclusion

# Citations

# Acknowledgements


# References
