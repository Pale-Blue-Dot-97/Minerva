Experiment Configs
==================

The most comprehensive way to config an experiment in ``minerva`` is to use ``YAML`` files.
This guide will walk through all the possible config options and their structure.
It will also explain how to use the config file for an experiment.

A good example to look at is ``example_config.yaml``

.. code-block:: yaml
    :caption: The ``example_config.yml`` demonstrating how to construct a master config to define an experiment in ``minerva``.

    ---
    #       *       *    __  ________   ____________ _    _____
    #   *        *      /  |/  /  _/ | / / ____/ __ \ |  / /   |  *           *
    #       *          / /|_/ // //  |/ / __/ / /_/ / | / / /| |     *
    #   *       *     / /  / // // /|  / /___/ _, _/| |/ / ___ |               *
    #    *           /_/  /_/___/_/ |_/_____/_/ |_| |___/_/  |_|     *   *
    #
    #                          EXAMPLE MASTER CONFIG FILE
    #
    # === PATHS ===================================================================
    data_root: tests/fixtures/data
    results_dir: tests/tmp/results
    cache_dir: tests/tmp/cache

    # === HYPERPARAMETERS =========================================================
    # ---+ Model Specification +---------------------------------------------------
    # Name of model. This no longer used for model class (see model_params).
    model_name: FCN32ResNet18-test

    # Type of model. Can be mlp, scene classifier, segmentation, ssl or siamese.
    model_type: segmentation

    # ---+ Sizing +----------------------------------------------------------------
    batch_size: 8                               # Number of samples in each batch.
    input_size: [4, 32, 32]   # patch_size plus leading channel dim.
    patch_size: '${to_patch_size: ${input_size}}'  # 2D tuple or float.
    n_classes: 8                                   # Number of classes in dataset.

    # ---+ Experiment Execution +--------------------------------------------------
    max_epochs: 4                         # Maximum number of training epochs.
    pre_train: false                      # Activate pre-training mode.
    fine_tune: false                      # Activate fine-tuning mode.
    elim: true                            # Eliminates empty classes from schema.
    balance: true                         # Balances dataset classes.
    torch_compile: true                   # Wrap model in `torch.compile`.

    # ---+ Optimisers +---------------------------------------------------
    lr: 1.0E-2                            # Learning rate of optimiser.
    optim_func: SGD                       # Name of the optimiser function.

    # ---+ Model Parameters +------------------------------------------------------
    model_params:
    _target_: minerva.models.FCN32ResNet18
    input_size: ${input_size}
    n_classes: ${n_classes}
    # any other params...

    # ---+ Optimiser Parameters +--------------------------------------------------
    optimiser:
    _target_: torch.optim.${optim_func}
    lr: ${lr}

    # ---+ Scheduler Parameters +--------------------------------------------------
    scheduler:
    _target_: torch.optim.lr_scheduler.LinearLR
    start_factor: 1.0
    end_factor: 0.5
    total_iters: 5

    # ---+ Loss Function Parameters +----------------------------------------------
    loss_params:
    _target_: torch.nn.CrossEntropyLoss

    # ---+ Dataloader Parameters +-------------------------------------------------
    loader_params:
    num_workers: 0
    pin_memory: true

    # === MODEL IO & LOGGING ======================================================
    # ---+ wandb Logging +---------------------------------------------------------
    wandb_log: true              # Activates wandb logging.
    project: pytest              # Define the project name for wandb.
    wandb_dir: /test/tmp/wandb   # Directory to store wandb logs locally.

    # ---+ Collator +--------------------------------------------------------------
    collator: torchgeo.datasets.stack_samples

    # === TASKS ===================================================================
    tasks:
    fit-train:
        _target_: minerva.tasks.StandardEpoch
        train: true
        record_float: true

        imagery_config: '${oc.create:${cfg_load: minerva/inbuilt_cfgs/dataset/NAIP.yaml}}'  # yamllint disable-line rule:line-length
        data_config: '${oc.create:${cfg_load: minerva/inbuilt_cfgs/dataset/Chesapeake7.yaml}}'  # yamllint disable-line rule:line-length

        # ---+ Dataset Parameters +----------------------------------------
        dataset_params:
            sampler:
                _target_: torchgeo.samplers.RandomGeoSampler
                roi: false
                size: ${patch_size}
                length: 32

            image:
                transforms: false
                subdatasets:
                    images_1:
                        _target_: minerva.datasets.__testing.TstImgDataset
                        paths: NAIP
                        res: 1.0

                    image2:
                        _target_: minerva.datasets.__testing.TstImgDataset
                        paths: NAIP
                        res: 1.0

            mask:
                transforms: false
                _target_: minerva.datasets.__testing.TstMaskDataset
                paths: Chesapeake7
                res: 1.0

    fit-val:
        _target_: minerva.tasks.StandardEpoch
        train: false
        record_float: true

        imagery_config: '${oc.create:${cfg_load: minerva/inbuilt_cfgs/dataset/NAIP.yaml}}'  # yamllint disable-line rule:line-length
        data_config: '${oc.create:${cfg_load: minerva/inbuilt_cfgs/dataset/Chesapeake7.yaml}}'  # yamllint disable-line rule:line-length

        # ---+ Minerva Inbuilt Logging Functions +-------------------------
        task_logger: minerva.logger.tasklog.SupervisedTaskLogger
        model_io: minerva.modelio.supervised_torchgeo_io

        # ---+ Dataset Parameters +----------------------------------------
        dataset_params:
            sampler:
                _target_: torchgeo.samplers.RandomGeoSampler
                roi: false
                size: ${patch_size}
                length: 32

            image:
                transforms: false
                _target_: minerva.datasets.__testing.TstImgDataset
                paths: NAIP
                res: 1.0

            mask:
                transforms: false
                _target_: minerva.datasets.__testing.TstMaskDataset
                paths: Chesapeake7
                res: 1.0

    test-test:
        _target_: minerva.tasks.StandardEpoch
        record_float: true

        imagery_config: '${oc.create:${cfg_load: minerva/inbuilt_cfgs/dataset/NAIP.yaml}}'  # yamllint disable-line rule:line-length
        data_config: '${oc.create:${cfg_load: minerva/inbuilt_cfgs/dataset/Chesapeake7.yaml}}'  # yamllint disable-line rule:line-length

        # ---+ Minerva Inbuilt Logging Functions +-------------------------
        task_logger: minerva.logger.tasklog.SupervisedTaskLogger
        model_io: minerva.modelio.supervised_torchgeo_io

        # ---+ Dataset Parameters +----------------------------------------
        dataset_params:
            sampler:
                _target_: torchgeo.samplers.RandomGeoSampler
                roi: false
                size: ${patch_size}
                length: 32

            image:
                transforms: false
                _target_: minerva.datasets.__testing.TstImgDataset
                paths: NAIP
                res: 1.0

            mask:
                transforms: false
                _target_: minerva.datasets.__testing.TstMaskDataset
                paths: Chesapeake7
                res: 1.0

    # === PLOTTING OPTIONS ========================================================
    plots:
        History: true   # Plot of the training and validation metrics over epochs.
        CM: true        # Confusion matrix.
        Pred: true      # Pie chart of the distribution of the predicted classes.
        ROC: true       # Receiver Operator Characteristics for each class.
        micro: true     # Include micro averaging in ROC plot.
        macro: true     # Include macro averaging in ROC plot.
        Mask: true      # Plot predicted masks against ground truth and imagery.

    # === MISCELLANEOUS OPTIONS ===================================================
    # ---+ Early Stopping +--------------------------------------------------------
    stopping:
    patience: 1    # No. of val epochs with increasing loss before stopping.
    verbose: true  # Verbosity of early stopping prints to stdout.

    # ---+ Verbosity and Saving +--------------------------------------------------
    verbose: true           # Verbosity of Trainer print statements to stdout.
    save: true              # Saves created figures to file.
    show: false             # Shows created figures in a pop-up window.
    p_dist: true            # Shows the distribution of classes to stdout.
    plot_last_epoch: true   # Plot the results of the last training and val epochs.

    # opt to ask at runtime; auto or True to automatically do so; or False,
    # None etc to not
    save_model: true

    # ---+ Other +-----------------------------------------------------------------
    # opt to ask at runtime; auto or True to automatically do so; or False,
    # None etc to not
    run_tensorboard: false
    calc_norm: false


Paths
-----

Paths to required directories are defined in the ``data_root``, ``results_dir`` and ``cache_dir`` keys.

.. code-block:: yaml
    :caption: Example ``dir`` dictionary describing the paths to directories needed in experiment.

    # === PATHS ===================================================================
    data_root: tests/fixtures/data
    results_dir: tests/tmp/results
    cache_dir: tests/tmp/cache

.. py:data:: data_root

    Path to the data directory where the input data is stored within. Can be relative or absolute.

    :type: str


.. py:data:: cache_dir

    Path to the cache directory storing dataset manifests and a place to output the latest / best version
    of a model. Can be relative or absolute.

    :type: str


.. py:data:: results_dir

    Path to the results directory where the results from all experiments will be stored.
    Can be relative or absolute.

    :type: str


Hyperparameters
---------------

This section of the config file covers hyperparmeters of the model and experiment.
The most important of these are now top-level variables in the config.
Most are also accessible from the CLI.

Model Specification
^^^^^^^^^^^^^^^^^^^

These parameters focus on defining the model, such as class, version and type.

.. code-block:: yaml

    # Name of model. Substring before hyphen is model class.
    model_name: FCN32ResNet18-MkI

    # Type of model.
    model_type: segmentation

.. py:data:: model_name

    Name of the model. Used to create the unique ``exp_name`` that is created dynamically for each experiment run.

    :type: str


.. py:data:: model_type

    Type of model. Can contain these key words seperated by hyphens:
        * ``"segmentation"``
        * ``"scene_classifier"``
        * ``"mlp"``
        * ``"ssl"``
        * ``"siamese"``
        * ``"change_detection"``
        * ``"multilabel"``

    :type: str
    :value: "scene_classifier"


Sizing
^^^^^^

These parameters concern the shapes and sizes of the IO to the model.

.. code-block:: yaml

    batch_size: 8             # Number of samples in each batch.
    patch_size: [32, 32]      # 2D tuple or float.
    input_size: [4, 32, 32]   # patch_size plus leading channel dim.
    n_classes: 8              # Number of classes in dataset.

.. py:data:: batch_size

    Number of samples in each batch.

    :type: int

.. py:data:: patch_size

    Define the shape of the patches in the dataset.

    :type: Tuple[int, int]

.. py:data:: input_size

    The :data:`patch_size` plus the leading channel dimension.

    :type: Tuple[int, int, int]

.. py:data:: n_classes

    Number of possible classes in the dataset.

    :type: int


Experiment Execution
^^^^^^^^^^^^^^^^^^^^

These parameters control the execution of the model fitting
such as the number of epochs, type of job or class balancing.

.. code-block:: yaml

    max_epochs: 5                         # Maximum number of training epochs.
    pre_train: false                      # Activate pre-training mode.
    fine_tune: false                      # Activate fine-tuning mode.
    elim: true                            # Eliminates empty classes from schema.
    balance: true                         # Balances dataset classes.


.. py:data:: max_epochs

    Maximum number of epochs of training and validation.

    :type: int
    :value: 5

.. py:data:: pre_train

    Defines this as a pre-train experiment. In this case, the backbone of the model will be saved
    to the cache at the end of training.

    :type: bool
    :value: False


.. py:data:: fine_tune

    Defines this as a fine-tuning experiment.

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

Loss and Optimisers
^^^^^^^^^^^^^^^^^^^
These parameters set the most important aspects of the loss function and optimiser.

.. code-block:: yaml

    loss_func: CrossEntropyLoss           # Name of the loss function to use.
    lr: 1.0E-2                            # Learning rate of optimiser.
    optim_func: SGD                       # Name of the optimiser function.


.. py:data:: loss_func

    Name of the loss function to use.

    :type: str

.. py:data:: lr

    Learning rate of the optimiser

    :type: float

.. py:data:: optim_func

    Name of the optimiser function.

    :type: str


Model Paramaters
^^^^^^^^^^^^^^^^
These are the parameters parsed to the model class to initiate it.

.. code-block:: yaml

    model_params:
        _target_: minerva.models.FCN32ResNet18
        input_size: ${input_size}
        n_classes: ${n_classes}
        # any other params...

Two common parameters are:

.. py:data:: input_size
    :noindex:

    Shape of the input to the model. Typically in CxHxW format.
    Should align with the values given for ``patch_size``.

    :type: list

.. py:data:: n_classes
    :noindex:

    Number of possible classes to predict in output.
    Best to parse :data:`n_classes` using ``${n_classes}``.

    :type: int

But you can add any other parameters in the ``model_params`` dict that the model expects.

Optimiser Parameters
^^^^^^^^^^^^^^^^^^^^

Here's where to place any additional parameters for the optimiser,
other than the already handled learning rate -- ``lr``. Place them in the ``params`` key.
If using a non-torch optimiser, use the ``module`` key to specify the import path to the optimiser function.

.. code-block:: yaml

    optimiser:
        _target_: torch.optim.${optim_func}
        lr: ${lr}

Loss Paramaters
^^^^^^^^^^^^^^^

Here's where to specify any additional parameters for the loss function in the ``params`` key.
If using a non-torch loss function, you need to specify the import path
with the ``module`` key.

.. code-block:: yaml

    loss:
        _target_: torch.nn.${loss_func}
        # any other params...

Dataloader Paramaters
^^^^^^^^^^^^^^^^^^^^^

Finally, this is where to define parameters for the
:class:`~torch.utils.data.DataLoader`. Unlike other parameters, there is no ``_target_`` field
as it is locked to ``DataLoader``.

.. code-block:: yaml

    loader_params:
        num_workers: 1
        pin_memory: true


Model IO & Logging
------------------

These parameters allow for the configuring how to handle different types of
input/ output to the model and how to handle logging of the model.

wandb Logging
^^^^^^^^^^^^^

Here's where to define how Weights and Biases (``wandb``) behaves in ``minerva``.

.. code-block:: yaml

    wandb_log: true              # Activates wandb logging.
    project: pytest              # Define the project name for wandb.
    wandb_dir: /test/tmp/wandb   # Directory to store wandb logs locally.


Minerva Inbuilt Logging Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition, there are also options for defining the logging, metric calculator
and IO function using inbuilt ``minerva`` functionality:

.. code-block:: yaml

    task_logger: minerva.logger.tasklog.SupervisedTaskLogger
    step_logger:
        _target_: minerva.logger.steplog.SupervisedStepLogger
        # any other params...

    model_io: minerva.modelio.supervised_torchgeo_io

    record_int: true    # Store integer results in memory.
    record_float: true  # Store floating point results too. Beware memory overload!


.. py:data:: logger
    :noindex:

    Specify the logger to use. Must be the name of a :class:`~minerva.logger.MinervaLogger` class
    within :mod:`logger`.

    :type: str


.. py:data:: metrics
    :noindex:

    Specify the metric logger to use. Must be the name of a :class:`~minerva.metrics.MinervaMetrics` class
    within :mod:`metrics`.

    :type: str


.. py:data:: model_io

    Specify the IO function to use to handle IO for the model during fitting. Must be the name
    of a function within :mod:`modelio`.

    :type: str


.. py:data:: record_int

    Store the integer results of each epoch in memory such the predictions, ground truth etc.

    :type: bool


.. py:data:: record_float

    Store the floating point results of each epoch in memory such as the raw predicted probabilities.

    .. warning::
        Could cause a memory overload issue with large datasets or systems with small RAM capacity.


Collator
^^^^^^^^

The collator is the function that collates the samples from the datset to make a mini-batch. It can be
defined using the simple ``collator`` param at the global-level.

.. code-block:: yaml

    collator: torchgeo.datasets.stack_samples


.. py:data:: collator

    Dot-based import path to the desired collator.

    :type: str


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


Miscellaneous Options
---------------------

And finally, this section holds various other options.

Early Stopping
^^^^^^^^^^^^^^

Here's where to define the behaviour of early stopping functionality.

.. code-block:: yaml

    stopping:
        patience: 2    # No. of val epochs with increasing loss before stopping.
        verbose: true  # Verbosity of early stopping prints to stdout.

.. py:data:: stopping

    Dictionary to hold the parameters defining the early stopping functionality.
    If no dictionary is given, it is assumed that there will be no early stopping.

    :type: dict


.. py:data:: patience

    Number of validation epochs with increasing loss from
    the lowest recorded validation loss before stopping the experiment.

    :type: int

.. py:data:: verbose
    :noindex:

    Verbosity of the early stopping prints to stdout.

    :type: bool


Verbosity and Saving
^^^^^^^^^^^^^^^^^^^^

These parameters dictate the behaviour of the outputs to stdout and saving results.

.. code-block:: yaml

    verbose: true           # Verbosity of Trainer print statements to stdout.
    save: true              # Saves created figures to file.
    show: false             # Shows created figures in a pop-up window.
    p_dist: true            # Shows the distribution of classes to stdout.
    plot_last_epoch: true   # Plot the results of the last training and val epochs.

    # opt to ask at runtime; auto or True to automatically do so; or False,
    # None etc to not
    save_model: true

.. py:data:: verbose

    Verbosity of :class:`~trainer.Trainer` prints to stdout.

    :type: bool


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


.. py:data:: plot_last_epoch

    Whether to plot the results from the final validation epoch.

    :type: bool
    :value: False

.. py:data:: save_model

    Whether to save the model at end of testing. Must be ``True``, ``False`` or ``"auto"``.
    Setting ``"auto"`` will automatically save the model to file.
    ``True`` will ask the user whether to or not at runtime.
    ``False`` will not save the model and will not ask the user at runtime.

    :type: str | bool
    :value: False

Other
^^^^^

All other options belong in this section.

.. code-block:: yaml

    # opt to ask at runtime; auto or True to automatically do so; or False,
    # None etc to not
    run_tensorboard: false
    calc_norm: false

.. py:data:: run_tensorboard

    Whether to run the Tensorboard logs at end of testing. Must be ``True``, ``False`` or ``"auto"``.
    Setting ``"auto"`` will automatically locate and run the logs on a local browser.
    ``True`` will ask the user whether to or not at runtime.
    ``False`` will not save the model and will not ask the user at runtime.

    :type: str | bool
    :value: False

.. py:data:: calc_norm

    *Depreciated*: Calculates the gradient norms.

    :type: bool
    :value: False
