Experiment Configs
==================

The most comprehensive way to config an experiment in ``minerva`` is to use ``YAML`` files.
This guide will walk through all the possible config options and their structure.
It will also explain how to use the config file for an experiment.


Globals
-------

The following are non-structured (i.e not a :class:`list` or :class:`dict`) global level variables
that can be set within the YAML config file. They can also be provided as command line arguments when
using the provided ``MinervaExp.py`` script.

.. py:data:: model_name

    Name of the model. Should take the form ``{class_name}-{version}`` where ``class_name``
    is a :class:`MinervaModel` class name and ``version`` is any string that can be used
    to differeniate version numbers of models and will be included in the ``exp_name`` used for results.

    :type: str


.. py:data:: model_type

    Type of model. Should be either ``"segmentation"``, ``"scene_classifier"``, ``"mlp"`` or ``"ssl"``.

    :type: str
    :value: "scene_classifier"


.. py:data:: pre_train

    Defines this as a pre-train experiment. In this case, the backbone of the model will be saved
    to the cache at the end of training.

    :type: bool
    :value: False


.. py:data:: fine_tune

    Defines this as a fine-tuning experiment.

    :type: bool
    :value: False


.. py:data:: sample_pairs
    :type: bool
    :value: False


.. py:data:: elim
    :type: bool
    :value: False


.. py:data:: balance
    :type: bool
    :value: False


.. py:data:: patch_size
    :type: tuple


.. py:data:: max_r
    :type: int
    :value: 256


.. py:data:: save_model
    :type: str | bool
    :value: False


.. py:data:: run_tensorboard
    :type: str | bool
    :value: False


.. py:data:: save
    :type: bool
    :value: True


.. py:data:: show
    :type: bool
    :value: False


.. py:data:: p_dist
    :type: bool
    :value: False


.. py:data:: calc_norm
    :type: bool
    :value: False


.. py:data:: plot_last_epoch
    :type: bool
    :value: False


Metrics and Loggers
'''''''''''''''''''
In addition, there are also options for defining the logging, metric calculator
and IO function at the global level:

.. py:data:: logger
    :type: str
    :noindex:


.. py:data:: metrics
    :type: str
    :noindex:


.. py:data:: model_io
    :type: str


Paths
-----

Paths to required directories are contained in the ``dir`` sub-dictionary with these keys:

.. py:data:: data

    Path to the data directory where the input data is stored within. Can be relative or absolute.
    Either defined as a string or a list of sequencial levels describing the path.

    :type: str | list


.. py:data:: cache

    Path to the cache directory storing dataset manifests and a place to output the latest / best version
    of a model. Can be relative or absolute. Either defined as a string or a list of sequencial levels
    describing the path.

    :type: str | list


.. py:data:: results

    Path to the results directory where the results from all experiments will be stored.
    Can be relative or absolute. Either defined as a string or a list of sequencial levels
    describing the path.

    :type: str | list


.. py:data:: configs

    Dictionary with two keys giving the paths to the auxillary configs:
    ``imagery_config`` and ``data_config``.

    :type: dict


Plots Dictionary
----------------

To define which plots to make from the results of testing, use the ``plots`` sub-dictionary with these keys:

.. py:data:: History
    :type: bool


.. py:data:: CM
    :type: bool


.. py:data:: Pred
    :type: bool


.. py:data:: ROC
    :type: bool


.. py:data:: micro
    :type: bool


.. py:data:: macro
    :type: bool


.. py:data:: Mask
    :type: bool
