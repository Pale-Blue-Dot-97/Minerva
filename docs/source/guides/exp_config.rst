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

    Defines this as a experiment using paired sampling e.g. for Siamese architectures.

    :type: bool
    :value: False


.. py:data:: elim

    Will eliminate classes that have no samples in and reorder the class labels so they
    still run from ``0`` to ``n-1`` classes where ``n`` is the reduced number of classes.
    ``minerva`` ensures that labels are converted between the old and new schemes seamlessly.

    :type: bool
    :value: False


.. py:data:: balance

    Activates class balancing. For ``model_type="scene_classifer"`` or ``model_type="mlp"``,
    over and under sampling will be used. For ``model_type="segmentation"``, class weighting will be
    used on the loss function.

    :type: bool
    :value: False


.. py:data:: patch_size

    Define the shape of the patches in the dataset.

    :type: Tuple[int, int]


.. py:data:: max_r

    Only used with *GeoCLR*. Maximum geospatial distance (in pixels) to sample
    the other side of the pair from.

    :type: int
    :value: 256


.. py:data:: save_model

    Whether to save the model at end of testing. Must be ``True``, ``False`` or ``"auto"``.
    Setting ``"auto"`` will automatically save the model to file.
    ``True`` will ask the user whether to or not at runtime.
    ``False`` will not save the model and will not ask the user at runtime.

    :type: str | bool
    :value: False


.. py:data:: run_tensorboard

    Whether to run the Tensorboard logs at end of testing. Must be ``True``, ``False`` or ``"auto"``.
    Setting ``"auto"`` will automatically locate and run the logs on a local browser.
    ``True`` will ask the user whether to or not at runtime.
    ``False`` will not save the model and will not ask the user at runtime.

    :type: str | bool
    :value: False


.. py:data:: save

    Whether to save plots created to file or not.

    :type: bool
    :value: True


.. py:data:: show

    Whether to show plots created in a window or not.

    .. warning::
        Do not use with a terminal-less operation, e.g. SLURM.

    :type: bool
    :value: False


.. py:data:: p_dist

    Whether to print the distribution of classes within the data to ``stdout``.

    :type: bool
    :value: False


.. py:data:: calc_norm

    *Depreciated*: Calculates the gradient norms.

    :type: bool
    :value: False


.. py:data:: plot_last_epoch

    Whether to plot the results from the final validation epoch.

    :type: bool
    :value: False


Metrics and Loggers
'''''''''''''''''''
In addition, there are also options for defining the logging, metric calculator
and IO function at the global level:

.. py:data:: logger

    Specify the logger to use. Must be the name of a :class:`MinervaLogger` class
    within :mod:`logger`.

    :type: str


.. py:data:: metrics

    Specify the metric logger to use. Must be the name of a :class:`MinervaMetrics` class
    within :mod:`metrics`.

    :type: str


.. py:data:: model_io

    Specify the IO function to use to handle IO for the model during fitting. Must be the name
    of a function within :mod:`modelio`.

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

    Plot a graph of the model history. By default, this will plot a graph of any metrics with
    keys containing ``"train"`` or ``"val"``.

    :type: bool


.. py:data:: CM

    Plots a confusion matrix.

    :type: bool


.. py:data:: Pred

    Plots a pie chart of the relative sizes of the classes within the predictions from the model.

    :type: bool


.. py:data:: ROC

    Plots a *Receiver over Operator Curve* (ROC) including *Area Under Curve* (AUC) scores.

    :type: bool


.. py:data:: micro

    Only used with ``ROC=True``. ROC plot includes micro-average ROC.

    .. warning::
        Adding this plot can be very computationally and memory intensive.
        Avoid use with large datasets!

    :type: bool


.. py:data:: macro

    Only used with ``ROC=True``. ROC plot includes macro-average ROC.

    :type: bool


.. py:data:: Mask

    Plots a comparison of predicted segmentation masks, the ground truth
    and original RGB imagery from a random selection of samples put to the model.

    :type: bool
