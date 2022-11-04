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
^^^^^^^^^^^^^^^^^^^

In addition, there are also options for defining the logging, metric calculator
and IO function at the global level:

.. py:data:: logger
    :noindex:

    Specify the logger to use. Must be the name of a :class:`MinervaLogger` class
    within :mod:`logger`.

    :type: str


.. py:data:: metrics
    :noindex:

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

.. code-block:: yaml
    :caption: Example ``dir`` dictionary describing the paths to directories needed in experiment.

    dir:
        data:
            - path
            - to
            - data
            - directory
        configs:
            data_config: ../../inbuilt_cfgs/Chesapeake13.yml
            imagery_config: ../../inbuilt_cfgs/NAIP.yml
        results:
            - path
            - to
            - results
            - directory
        cache: can/also/be/a/string/path/to/cache/directory


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

.. code-block:: yaml
    :caption: Example ``plots`` dictionary.

    plots:
        History: True
        CM: False
        Pred: False
        ROC: False
        micro: False
        macro: True
        Mask: False

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


Dataset Parameters
------------------

To define what datasets to use in the experiment, use the ``dataset_params`` dictionary in the following structure.

.. code-block:: yaml
    :caption: Example ``dataset_params`` defining train, validation and test datasets with image and mask sub-datasets.

    dataset_params:
    # Training Dataset
    train:
        image:
            module: torchgeo.datasets
            name: NAIP
            root: NAIP/NAIP 2013/Train
            params:
                res: 1.0
        mask:
            module: torchgeo.datasets
            name: Chesapeake13
            root: Chesapeake13
            params:
                res: 1.0
                download: False
                checksum: False

    # Validation Dataset
    val:
        image:
            module: torchgeo.datasets
            name: NAIP
            root: NAIP/NAIP 2013/Validation
            params:
                res: 1.0
        mask:
            module: torchgeo.datasets
            name: Chesapeake13
            root: Chesapeake13
            params:
                res: 1.0
                download: False

    # Test Dataset
    test:
        image:
            module: torchgeo.datasets
            name: NAIP
            root: NAIP/NAIP 2013/Test
            params:
                res: 1.0
        mask:
            module: torchgeo.datasets
            name: Chesapeake13
            root: Chesapeake13
            params:
                res: 1.0
                download: False


The first level of keys defines the modes of model fitting: ``"train"``, ``"val"`` and ``"test"``.
The presence (or not) of these keys in ``dataset_params`` defines which modes of fitting will be conducted.
Currently, ``"train"`` and ``"val"`` must be present with ``"test"`` being optional.

The keys within each mode define the sub-datasets that will be intersected together to form
the dataset for that mode. Currently, only ``"image"`` and ``"mask"`` keys are allowed, in line with
:mod:`torchgeo` use of the same keys.

Within each sub-dataset params, there are four possible keys:

.. py:data:: module

    Defines the module the dataset class is in.

    :type: str
    :value: "torchgeo.datasets"


.. py:data:: name

    Name of the dataset class that is within ``module``.

    :type: str


.. py:data:: root

    Path from the data directory (given in ``dir["data"]``) to the directory containing the sub-dataset data.

    :type: str


.. py:data:: params

    Arguments to the dataset class (excluding ``root``).

    :type: Dict[str, Any]


Sampler Parameters
------------------

Defining the samplers used to provide indices from the dataset to samples is done in a
similar structure to ``dataset_params``. The first level is again the modes of model fitting.
These keys *MUST* match those in ``dataset_params``.

.. code-block:: yaml
    :caption: Exampler ``sampler_params``.

    sampler_params:
        # Training Dataset Sampler
        train:
            module: torchgeo.samplers
            name: RandomGeoSampler
            roi: False
            params:
                size: 224
                length: 96000

        # Validation Dataset Sampler
        val:
            module: torchgeo.samplers
            name: RandomGeoSampler
            roi: False
            params:
                size: 224
                length: 3200

        # Test Dataset Sampler
        test:
            module: torchgeo.samplers
            name: RandomGeoSampler
            roi: False
            params:
                size: 224
                length: 9600


There is only a sampler for the overall intersected dataset, not for each sub-dataset.
Within each mode, there are 4 recognised keys again:


.. py:data:: module
    :noindex:

    Module name that the sampler class resides in.

    :type: str


.. py:data:: name
    :noindex:

    Name of sampler class within ``module``.

    :type: str


.. py:data:: roi

    Region-of-interest. Providing ``False`` uses dataset ROI. Else, one can provide a 6-element :class:`tuple`
    in the order ``minx``, ``maxx``, ``miny``, ``maxy``, ``mint`` and ``maxt`` that defines a reduced area
    to sample from that are within the dataset ROI.

    :type: Literal[False] | Tuple[float, float, float, float, float, float]


.. py:data:: params
    :noindex:

    Arguments to sampler constructor (excluding ROI).

    :type: dict


Transform Parameters
--------------------

The ``transform_params`` follows a similar structure as ``dataset_params`` and ``sampler_params`` but with
some extra functionality in order and types of transforms that can be specified.

Again, the first level of keys is the modes of model fitting and these *MUST* match those in ``dataset_params``.

.. code-block:: yaml
    :caption: Example ``transform_params``.

    transform_params:
        # Training Dataset Transforms
        train:
            image:
                Normalise:
                    module: minerva.transforms
                    norm_value: 255
                RandomHorizontalFlip:
                    module: torchvision.transforms
                RandomVerticalFlip:
                    module: torchvision.transforms
                RandomResizedCrop:
                    module: torchvision.transforms
                    size: 224
                GaussianBlur:
                    module: torchvision.transforms
                    kernel_size: 25

        # Validation Dataset Transforms
        val:
            image:
                Normalise:
                    module: minerva.transforms
                    norm_value: 255
                RandomHorizontalFlip:
                    module: torchvision.transforms
                RandomVerticalFlip:
                    module: torchvision.transforms
                RandomResizedCrop:
                    module: torchvision.transforms
                    size: 224
                GaussianBlur:
                    module: torchvision.transforms
                    kernel_size: 25


The transforms are added to each sub-datasets, not the overall intersected dataset of the mode.
So the next level down is the sub-dataset keys which again must match the same ones provided for that
mode in ``dataset_params``.

Within each sub-dataset, the transforms are defined. Each key is the name of the transform class.
The order the transforms are given is respected.

Within each transform :class:`dict`, the ``module`` key again gives the module name.
The default is ``"torchvison.transforms"``. All other keys given are parsed to the transform constructor.

There is one exception to this structure and that is the use of ``torchvision.transforms.RandomApply``.
If the transform key is ``RandomApply`` then transforms can be provided within that :class:`dict` in the same
structure with the addition of a ``p`` key that gives the propability that the transforms within are applied.


Collator params
---------------

The collator is the function that collates the samples from the datset to make a mini-batch. It can be
defined using the simple ``collator`` :class:`dict`.

.. code-block:: yaml
    :caption: Example of ``collator_params``.

    collator:
        module: torchgeo.datasets
        name: stack_samples


.. py:data:: module
    :noindex:

    Name of module that collator function can be imported from.

    :type: str


.. py:data:: name
    :noindex:

    Name of collator function.

    :type: str


Hyperparams
-----------


.. code-block:: yaml
    :caption: Example ``hyperparams``.

    hyperparams:
        params:
            batch_size: 256
            num_workers: 10
            pin_memory: True
        model_params:
            input_size: [4, 224, 224]
        optim_params:
            name: LARS
            module: minerva.optimisers
            lr: 3.0E-4
            weight_decay: 1.0E-4
            max_epochs: 100
        loss_params:
            name: NT_Xent
            module: simclr.modules
            params:
                temperature: 0.07
                batch_size: 16
                world_size: 1
        max_epochs: 5
        stopping:
            patience: 3
            verbose: True



.. py:data:: max_epochs

    Maximum number of epochs of training and validation.

    :type: int
    :value: 25


Dataloader Paramaters
^^^^^^^^^^^^^^^^^^^^^


Model Paramaters
^^^^^^^^^^^^^^^^

Optimiser Parameters
^^^^^^^^^^^^^^^^^^^^

Loss Paramaters
^^^^^^^^^^^^^^^

Early Stopping
^^^^^^^^^^^^^^
