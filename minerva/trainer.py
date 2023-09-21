# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2023 Harry Baker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Module containing the class :class:`~trainer.Trainer` to handle the fitting of neural networks."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
from __future__ import annotations

__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = ["Trainer"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd
import torch
import yaml
from inputimeout import TimeoutOccurred, inputimeout
from nptyping import Int, NDArray
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter

from torchinfo import summary
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from minerva.datasets import make_loaders
from minerva.metrics import MinervaMetrics
from minerva.models import (
    MinervaBackbone,
    MinervaDataParallel,
    MinervaModel,
    MinervaOnnxModel,
    MinervaWrapper,
)
from minerva.pytorchtools import EarlyStopping
from minerva.tasks import MinervaTask, TSNEVis, get_task
from minerva.utils import universal_path, utils, visutils

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Default time till timeout waiting for a user input in seconds.
_timeout = 30
_tensorflow_exist = utils.check_optional_import_exist("tensorflow")
TENSORBOARD_WRITER: Optional[Callable[..., Any]]
try:
    TENSORBOARD_WRITER = utils._optional_import(
        "torch.utils.tensorboard.writer",
        name="SummaryWriter",
        package="tensorflow",
    )
except ImportError as err:  # pragma: no cover
    print(err)
    print("Disabling TensorBoard logging")
    TENSORBOARD_WRITER = None


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class Trainer:
    """Helper class to handle the entire fitting and evaluation of a model.

    Attributes:
        params (dict[str, ~typing.Any]): Dictionary describing all the parameters that define how the model will be
            constructed, trained and evaluated. These should be defined via config ``YAML`` files.
        model (MinervaModel): Model to be fitted of a class contained within :mod:`~minerva.models`.
        max_epochs (int): Number of epochs to train the model for.
        batch_size (int): Size of each batch of samples supplied to the model.
        loaders (dict[str, ~torch.utils.data.DataLoader]): :class:`dict` containing
            :class:`~torch.utils.data.DataLoader` (s) for each dataset.
        n_batches (dict[str, int]): Dictionary of the number of batches to supply to the model for train,
            validation and testing.
        metrics (dict[str, ~typing.Any]): Dictionary to hold the loss and accuracy results from training,
            validation and testing.
        device: The CUDA device on which to fit the model.
        gpu (int): CUDA GPU device number this process is running on.
        verbose (bool): Provides more prints to stdout if ``True``.
        fine_tune (bool): Assumes this is a fine-tuning job if ``True``. Will attempt to load model weights
            from cache.
        class_dist (~typing.Any): Distribution of classes within the data.
        exp_name (~pathlib.Path): :class:`~pathlib.Path` to the unique results directory for this experiment.
        sample_pairs (bool): Whether samples are paired together for Siamese learning.
        modes (tuple[str, ...]): The different *modes* of fitting in this experiment specified by the config.
        writer (~torch.utils.tensorboard.writer.SummaryWriter | ~wandb.sdk.wandb_run.Run | None): The *writer*
            to perform logging for this experiment. For use with either :mod:`tensorboard` or :mod:`wandb`.
        stopper (~pytorchtools.EarlyStopping | None): Early stopping function.
        early_stop (bool): Whether early stopping has been triggered. Will end model training if ``True``.
        n_samples (dict[str, int]): Number of samples in each mode of model fitting.
        metric_logger (~logger.MinervaLogger): Object to calculate and log metrics to track the performance
            of the model.
        modelio_func (~typing.Callable[..., ~typing.Any]): Function to handle the input/ output to the model.
        steps (dict[str, int]): :class:`dict` to hold the global step number for each mode of model fitting.
        model_type (str): Type of the model that determines how to handle IO, metric calculations etc.

    Args:
        gpu (int): Optional; CUDA GPU device number. For use in distributed computing. Defaults to ``0``.
        rank (int): Optional; The rank of this process across all devices in the distributed run.
        world_size (int): Optional; The total number of processes across the distributed run.
        verbose (bool): Turns messages to stdout off/on.
        wandb_run (~wandb.sdk.wandb_run.Run | RunDisabled): Optional; Run object for Weights and Biases.
        params (dict[str, ~typing.Any]): Dictionary describing all the parameters that define how the model will be
            constructed, trained and evaluated. These should be defined via config ``YAML`` files.

    Keyword Args:
        batch_size (int): Number of samples in each batch.
        elim (bool): Will eliminate classes that have no samples in and reorder the class labels so they
            still run from ``0`` to ``n-1`` classes where ``n`` is the reduced number of classes.
            :mod:`minerva` ensures that labels are converted between the old and new schemes seamlessly.
        model_type (str): Defines the type of the model. If ``siamese``, ensures inappropiate functionality is not used.
        dir (dict[str, ~typing.Any]): Dictionary providing the paths to directories needed.
            Must include the ``data`` path.
        loader_params (dict[str, ~typing.Any]): Parameters for the :class:`~torch.utils.data.DataLoader`.
            Unlike the other ``x_params`` dicts, parameters are placed at the immediate ``loader_params`` level
            (not in a ``params`` key).
        dataset_params (dict[str, ~typing.Any]): Parameters to construct each dataset.
            See documentation on structure of these.
        sampler_params (dict[str, ~typing.Any]): Parameters to construct the samplers for each mode of model fitting.
        transform_params (dict[str, ~typing.Any]): Parameters to construct the transforms for each dataset.
            See documentation for the structure of these.
        collator (dict[str, ~typing.Any]): Defines the collator to use that will collate samples together into batches.
            Contains the ``module`` key to define the import path and the ``name`` key
            for name of the collation function.
        sample_pairs (bool): Activates paired sampling for Siamese models. Only used for ``train`` datasets.
        model_name (str): Name of the model to be used in filenames of results.
        model_params (dict[str, ~typing.Any]): Parameters parsed to the model class to initiate it.
        max_epochs (int): Number of epochs to train the model for.
        val_freq (int): Perform a validation epoch with KNN for every ``val_freq``
            training epochs for SSL or Siamese models.
        knn_k (int): Top-k most similar images used to predict the image for KNN validation.
        fine_tune (bool): Activate fine-tuning mode.
        wandb_log (bool): Activates :mod:`wandb` logging.
        stopping (dict[str, ~typing.Any]): Dictionary to hold the parameters defining the early stopping functionality.
            If no dictionary is given, it is assumed that there will be no early stopping.
        pre_train_name (str): Name of the pre-trained model to use.
        reload (bool): Reloads the weights in the cache matching ``pre_train_name`` to continue model fitting.
        loss_func (str): Name of the loss function to use.
        optim_func (str): Name of the optimiser function to use.
        lr (float): Learning rate of optimiser.
        optim_params (dict[str, ~typing.Any]): :class:`dict` to hold any additional parameters for the optimiser,
            other than the already handled learning rate -- ``lr``. Place them in the ``params`` key.
            If using a non-torch optimiser, use the ``module`` key to specify the import path to the optimiser function.
        loss_params (dict[str, ~typing.Any]): :class:`dict` to hold any additional parameters for the loss function
            in the ``params`` key. If using a non-torch loss function, you need to specify the import path
            with the ``module`` key.
        balance (bool): Activates class balancing. For ``model_type="scene classifer"`` or ``model_type="mlp"``,
            over and under sampling will be used. For ``model_type="segmentation"``, class weighting will be
            used on the loss function.
        patch_size (tuple[float, float]): Defines the shape of the patches in the dataset.
        input_size (tuple[int, ...]): Shape of the input to the model. Typically in CxHxW format.
            Should align with the values given for ``patch_size``.
        metrics (str): Specify the metric logger to use. Must be the name of a :class:`~metrics.MinervaMetric` class
            within :mod:`metrics`.
        logger (str): Specify the logger to use. Must be the name of a :class:`~logger.MinervaLogger` class
            within :mod:`logger`.
        modelio (str): Specify the IO function to use to handle IO for the model during fitting. Must be the name
            of a function within :mod:`modelio`.
        record_int (bool): Store the integer results of each epoch in memory such the predictions, ground truth etc.
        record_float (bool): Store the floating point results of each epoch in memory
            such as the raw predicted probabilities.
        plots (dict[str, bool]): :class:`dict` to define plots to make from results of testing. Possible plot types are:

            * History: Plot a graph of any metrics with keys containing ``"train"`` or ``"val"`` over epochs.
            * CM: Plots a confusion matrix of the predictions against ground truth.
            * Pred: Plots a pie chart of the relative sizes of the classes within the predictions from the model.
            * ROC: Plots a *Receiver over Operator Curve* (ROC) including *Area Under Curve* (AUC) scores.
            * micro: Only used with ``ROC=True``. ROC plot includes micro-average ROC.
            * macro: Only used with ``ROC=True``. ROC plot includes macro-average ROC.
            * Mask: Plots a comparison of predicted segmentation masks, ground truth and original RGB imagery.
        plot_last_epoch (bool): Plot the results from the final validation epoch.
        save (bool): Save plots created to file or not.
        show (bool): Show plots created in a window or not.
        verbosity (bool): Verbosity of :class:`~trainer.Trainer` prints to stdout.
        p_dist (bool): Prints the distribution of classes within the data to ``stdout``.
        save_model (bool | str): Whether to save the model at end of testing. Must be ``True``, ``False`` or ``"auto"``.
            Setting ``"auto"`` will automatically save the model to file.
            ``True`` will ask the user whether to or not at runtime.
            ``False`` will not save the model and will not ask the user at runtime.
        run_tensorboard (bool | str): Whether to run the Tensorboard logs at end of testing.
            Must be ``True``, ``False`` or ``"auto"``.
            Setting ``"auto"`` will automatically locate and run the logs on a local browser.
            ``True`` will ask the user whether to or not at runtime.
            ``False`` will not save the model and will not ask the user at runtime.

    .. warning::
        Using ``record_float=True`` may cause a memory overload issue with large datasets
        or systems with small RAM capacity.

    .. warning::
        Using ``plots={ROC: True, micro: True}`` can be very computationally and memory intensive.
        Avoid use with large datasets!

    """

    def __init__(
        self,
        gpu: int = 0,
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True,
        wandb_run: Optional[Union[Run, RunDisabled]] = None,
        **params: Dict[str, Any],
    ) -> None:
        assert not isinstance(wandb_run, RunDisabled)

        # Gets the datasets, number of batches, class distribution and the modfied parameters for the experiment.
        # loaders, n_batches, class_dist, new_params = make_loaders(
        #    rank, world_size, **params
        # )

        # Sets the global GPU number for distributed computing. In single process, this will just be 0.
        self.gpu: int = gpu
        self.rank = rank
        self.world_size = world_size

        # Finds and sets the CUDA device to be used.
        self.device = utils.get_cuda_device(gpu)

        # Verbose level. Always 0 if this is not the primary GPU to avoid duplicate stdout statements.
        self.verbose: bool = verbose if gpu == 0 else False

        if self.gpu == 0 and self.verbose:
            # Prints config to stdout.
            print(
                "\n==+ Experiment Parameters +====================================================="
            )
            utils.print_config(new_params)

        self.params: Dict[str, Any] = new_params
        # self.class_dist = class_dist
        # self.loaders: Dict[str, DataLoader[Iterable[Any]]] = loaders
        # self.n_batches = n_batches
        self.batch_size: int = self.params["batch_size"]
        self.model_type: str = self.params["model_type"]
        self.val_freq: int = self.params.get("val_freq", 1)
        self.sample_pairs: bool = self.params.get("sample_pairs", False)

        # Sets the max number of epochs of fitting.
        self.max_epochs = self.params.get("max_epochs", 25)

        self.modes = self.params["dataset_params"].keys()

        # Flag for a fine-tuning experiment.
        self.fine_tune = self.params.get("fine_tune", False)

        # Sets the timestamp of the experiment.
        self.params["timestamp"] = utils.timestamp_now(fmt="%d-%m-%Y_%H%M")

        # Sets experiment name and adds this to the path to the results' directory.
        self.params["exp_name"] = "{}_{}".format(
            self.params["model_name"], self.params["timestamp"]
        )

        self.params["dir"]["results"] = universal_path(self.params["dir"]["results"])
        results_dir = self.params["dir"]["results"] / self.params["exp_name"]

        if self.gpu == 0:
            # Makes a directory for this experiment.
            utils.mkexpdir(self.params["exp_name"])

        # Path to experiment directory and experiment name.
        self.exp_fn: Path = results_dir / self.params["exp_name"]

        self.writer: Optional[Union[SummaryWriter, Run]] = None
        if self.params.get("wandb_log", False):
            # Sets the `wandb` run object (or None).
            self.writer = wandb_run
            self.init_wandb_metrics()
        else:
            if _tensorflow_exist:
                assert TENSORBOARD_WRITER

                # Initialise TensorBoard logger.
                self.writer = TENSORBOARD_WRITER(results_dir)
            else:  # pragma: no cover
                self.writer = None

        self.model: Union[MinervaModel, MinervaDataParallel, MinervaBackbone]
        if Path(self.params.get("pre_train_name", "none")).suffix == ".onnx":
            # Loads model from `onnx` format.
            self.model = self.load_onnx_model()
        else:
            # Creates model (and loss function) from specified parameters in params.
            self.model = self.make_model()

        # Determines the output shape of the model.
        sample_pairs: Union[bool, Any] = self.sample_pairs
        if not isinstance(sample_pairs, bool):
            sample_pairs = False
            self.params["sample_pairs"] = False

        assert isinstance(sample_pairs, bool)
        self.sample_pairs = sample_pairs
        self.model.determine_output_dim(sample_pairs=sample_pairs)

        print(self.device)
        # Transfer to GPU.
        self.model.to(self.device)

        # Sets up the early stopping functionality.
        self.stopper = None
        self.early_stop = False
        if "stopping" in self.params:
            self.stopper = EarlyStopping(
                path=f"{self.exp_fn}.pt",
                trace_func=self.print,
                **self.params["stopping"],
            )

        # Calculates number of samples in each mode of fitting.
        self.n_samples = {
            mode: self.n_batches[mode] * self.batch_size for mode in self.modes
        }

        # Initialise the metric logger and model IO for the experiment.
        # self.metric_logger = self.make_metric_logger()
        # self.modelio_func = self.get_io_func()

        # Stores the step number for that mode of fitting. To be used for logging.
        self.step_num = {mode: 0 for mode in self.modes}

        # Creates and sets the optimiser for the model.
        self.make_optimiser()

        if self.gpu == 0:
            if isinstance(self.writer, Run):
                self.writer.config.update(self.params)

            # Determines the input size of the model.
            input_size = self.get_input_size()

            if self.verbose:
                # Print model summary.
                summary(self.model, input_size=input_size)

            if _tensorflow_exist:
                if (
                    (
                        torch.cuda.device_count() == 1
                        or self.device == torch.device("cpu")
                    )
                    and isinstance(
                        self.writer, utils.extract_class_type(TENSORBOARD_WRITER)
                    )
                    and self.writer
                ):
                    # Adds a graphical layout of the model to the TensorBoard logger.
                    try:
                        self.writer.add_graph(  # type: ignore[attr-defined]
                            self.model,
                            input_to_model=torch.rand(*input_size, device=self.device),
                        )
                    except RuntimeError as err:  # pragma: no cover
                        print(err)
                        print("ABORT adding graph to writer")

        # If writer is `wandb`, `watch` the model to log gradients.
        if isinstance(self.writer, Run):
            self.writer.watch(self.model)

        # Checks if multiple GPUs detected. If so, wraps model in DistributedDataParallel for multi-GPU use.
        if torch.cuda.device_count() > 1:  # pragma: no cover
            self.print(f"{torch.cuda.device_count()} GPUs detected")
            self.model = torch.nn.modules.SyncBatchNorm.convert_sync_batchnorm(  # type: ignore
                self.model
            )
            self.model = MinervaDataParallel(self.model, DDP, device_ids=[gpu])

    def init_wandb_metrics(self) -> None:
        """Setups up separate step counters for :mod:`wandb` logging of train, val, etc."""
        if isinstance(self.writer, Run):
            for mode in self.n_batches:
                self.writer.define_metric(f"{mode}/step")
                self.writer.define_metric(f"{mode}/*", step_metric=f"{mode}/step")

    def get_input_size(self) -> Tuple[int, ...]:
        """Determines the input size of the model.

        Returns:
            tuple[int, ...]: :class:`tuple` describing the input shape of the model.
        """
        input_shape: Optional[Tuple[int, ...]] = self.model.input_size  # type: ignore
        assert input_shape is not None
        input_size: Tuple[int, ...] = (self.batch_size, *input_shape)

        if self.sample_pairs:
            input_size = (2, *input_size)

        return input_size

    def get_model_cache_path(self) -> Path:
        """Get the path to where to cache this model to.

        Returns:
            ~pathlib.Path: :class:`~pathlib.Path` to cache directory and the filename
            (model name excluding version and file extension).
        """
        cache_dir = universal_path(self.params["dir"]["cache"])
        return Path(cache_dir / Path(self.params["model_name"].split("-")[0]))

    def get_weights_path(self) -> Path:
        """Get the path to the cached version of the pre-trained model.

        Returns:
            ~pathlib.Path: :class:`~pathlib.Path` to the cached model (excluding file extension).
        """
        cache_dir = universal_path(self.params["dir"]["cache"])
        return Path(cache_dir / Path(self.params["pre_train_name"]).with_suffix(""))

    def make_model(self) -> MinervaModel:
        """Creates a model from the parameters specified by config.

        Returns:
            MinervaModel: Initialised model.
        """
        model_params: Dict[str, Any] = self.params["model_params"]

        module = model_params.pop("module", "minerva.models")
        if not module:
            module = "minerva.models"
        is_minerva = True if module == "minerva.models" else False

        # Gets the model requested by config parameters.
        _model = utils.func_by_str(module, self.params["model_name"].split("-")[0])

        if self.fine_tune:
            # Add the path to the pre-trained weights to the model params.
            model_params["backbone_weight_path"] = f"{self.get_weights_path()}.pt"

        params = model_params.get("params", {})
        if "n_classes" in params.keys():
            # Updates the number of classes in case it has been altered by class balancing.
            params["n_classes"] = self.params["n_classes"]

        if "num_classes" in params.keys():
            # Updates the number of classes in case it has been altered by class balancing.
            params["num_classes"] = self.params["n_classes"]

        # Initialise model.
        model: MinervaModel
        if is_minerva:
            model = _model(self.make_criterion(), **params)
        else:
            model = MinervaWrapper(
                _model,
                self.make_criterion(),
                **params,
            )

        if self.params.get("reload", False):
            model.load_state_dict(
                torch.load(f"{self.get_weights_path()}.pt", map_location=self.device)
            )

        return model

    def load_onnx_model(self) -> MinervaModel:
        """Loads and returns a :mod:`onnx` model from the cache in :mod:`torch` form.

        Assumes that the :mod:`onnx` model came from :mod:`minerva`.

        Returns:
            MinervaModel: Loaded model ready for use.
        """
        onnx_load = utils._optional_import(
            "onnx",
            name="load",
            package="onnx",
        )
        convert = utils._optional_import(
            "onnx2torch",
            name="convert",
            package="onnx2torch",
        )

        model_params = self.params["model_params"].get("params", {})

        onnx_model = convert(onnx_load(f"{self.get_weights_path()}.onnx"))
        model = MinervaOnnxModel(onnx_model, self.make_criterion(), **model_params)
        assert isinstance(model, MinervaModel)
        return model

    def make_criterion(self) -> Any:
        """Creates a :mod:`torch` loss function based on config parameters.

        Returns:
            ~typing.Any: Initialised :mod:`torch` loss function specified by config parameters.
        """
        # Gets the loss function requested by config parameters.
        loss_params: Dict[str, Any] = self.params["loss_params"].copy()
        module = loss_params.pop("module", "torch.nn")
        criterion: Callable[..., Any] = utils.func_by_str(
            module, self.params["loss_func"]
        )

        if not utils.check_dict_key(loss_params, "params"):
            loss_params["params"] = {}

        if self.params.get("balance", False) and self.model_type == "segmentation":
            weights_dict = utils.class_weighting(self.class_dist, normalise=False)

            weights = []
            if self.params.get("elim", False):
                for i in range(len(weights_dict)):
                    weights.append(weights_dict[i])
            else:
                for i in range(self.params["n_classes"]):
                    weights.append(weights_dict.get(i, 0.0))

            loss_params["params"]["weight"] = Tensor(weights)

            return criterion(**loss_params["params"])

        else:
            return criterion(**loss_params["params"])

    def make_optimiser(self) -> None:
        """Creates a :mod:`torch` optimiser based on config parameters and sets optimiser."""

        # Gets the optimiser requested by config parameters.
        optimiser_params: Dict[str, Any] = self.params["optim_params"].copy()
        module = optimiser_params.pop("module", "torch.optim")
        optimiser = utils.func_by_str(module, self.params["optim_func"])

        if not utils.check_dict_key(optimiser_params, "params"):
            optimiser_params["params"] = {}

        # Add learning rate from top-level of config to the optimiser parameters.
        optimiser_params["params"]["lr"] = self.params["lr"]

        # Constructs and sets the optimiser for the model based on supplied config parameters.
        self.model.set_optimiser(  # type: ignore
            optimiser(self.model.parameters(), **optimiser_params["params"])
        )

    def fit(self) -> None:
        """Fits the model by running ``max_epochs`` number of training and validation epochs."""
        fit_params = deepcopy(self.params["tasks"]["fit"])

        tasks: Dict[str, MinervaTask] = {}
        for mode in ("train", "val"):
            tasks[mode] = get_task(
                fit_params[mode].pop("type"),
                self.model,
                self.batch_size,
                self.device,
                self.exp_fn,
                self.gpu,
                self.rank,
                self.world_size,
                self.writer,
                fit_params[mode],
            )

        for epoch in range(self.max_epochs):
            self.print(
                f"\nEpoch: {epoch + 1}/{self.max_epochs} =========================================================="
            )

            # Conduct training or validation epoch.
            for mode in ("train", "val"):
                # Only run a validation epoch at set frequency of epochs. Goes to next epoch if not.
                if mode == "val" and (epoch + 1) % self.val_freq != 0:
                    break

                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval()

                results: Optional[Dict[str, Any]]

                results = tasks[mode](mode)

                # Add epoch number to metrics.
                self.metric_logger.log_epoch_number(mode, epoch)

                # Print epoch results.
                if self.gpu == 0:
                    if mode == "val":
                        epoch_no = epoch // self.val_freq
                    else:
                        epoch_no = epoch
                    self.metric_logger.print_epoch_results(mode, epoch_no)

                # Sends validation loss to the stopper and updates early stop bool.
                if mode == "val" and self.stopper is not None:
                    if mode == "val":
                        epoch_no = epoch // self.val_freq
                    else:
                        epoch_no = epoch

                    val_loss = self.metric_logger.get_metrics["val_loss"]["y"][epoch_no]
                    self.stopper(val_loss, self.model)
                    self.early_stop = self.stopper.early_stop

                # Special case for final train/ val epoch to plot results if configured so.
                if epoch == (self.max_epochs - 1) or self.early_stop:
                    if self.early_stop and mode == "val":  # pragma: no cover
                        self.print("\nEarly stopping triggered")

                    # Ensures that plots likely to cause memory issues are not attempted.
                    plots: Dict[str, bool] = self.params.get("plots", {}).copy()
                    plots["CM"] = False
                    plots["ROC"] = False

                    if not self.params.get("plot_last_epoch", False):
                        # If not plotting results, ensure that only history plotting will remain
                        # if originally set to do so.
                        plots["Mask"] = False
                        plots["Pred"] = False

                    # Create a subset of metrics which drops the testing results for plotting model history.
                    sub_metrics = self.metric_logger.get_sub_metrics()

                    # Ensures masks are not plotted for model types that do not yield such outputs.
                    if utils.check_substrings_in_string(
                        self.model_type, "scene classifier", "mlp", "MLP"
                    ):
                        plots["Mask"] = False

                    # Amends the results' directory to add a new level for train or validation.
                    results_dir = self.exp_fn.parent / mode

                    assert results is not None

                    if self.gpu == 0:
                        # Plots the results of this epoch.
                        visutils.plot_results(
                            plots,
                            metrics=sub_metrics,
                            class_names=self.params.get("classes"),
                            colours=self.params.get("colours"),
                            save=True,
                            show=False,
                            model_name=self.params["model_name"],
                            timestamp=self.params["timestamp"],
                            results_dir=results_dir,
                            **results,
                        )

                # If early stopping has been triggered, loads the last model save to replace current model,
                # ready for testing.
                if self.early_stop:  # pragma: no cover
                    if self.gpu == 0:
                        self.model.load_state_dict(torch.load(f"{self.exp_fn}.pt"))
                    return

    def test(self, save: bool = True, show: bool = False) -> None:
        """Tests the model by running a testing epoch then taking the results and orchestrating the plotting and
        analysis of them.

        Args:
            save (bool): Optional; Determines whether to save the plots created to file.
            show (bool): Optional; Determines whether to show the plots created.

        Notes:
            ``save=True``, ``show=False`` regardless of input for plots made for each sample such as PvT or Mask plots.

        """
        self.print("\r\nTESTING")

        test_params = deepcopy(self.params["tasks"]["tests"])

        for _task in test_params:
            task = get_task(
                test_params[_task].pop("type"),
                self.model,
                self.batch_size,
                self.device,
                self.exp_fn,
                self.gpu,
                self.rank,
                self.world_size,
                self.writer,
                test_params[_task],
            )

            # Runs test epoch on model, returning the predicted labels, ground truth labels supplied
            # and the IDs of the samples supplied.
            results = task("test")

            assert results is not None

            # Prints test loss and accuracy to stdout.
            self.metric_logger.print_epoch_results("test", 0)

            # Add epoch number to testing results.
            self.metric_logger.log_epoch_number("test", 0)

            if self.gpu == 0:
                if "z" in results and "y" in results:
                    self.print("\nMAKING CLASSIFICATION REPORT")
                    self.compute_classification_report(results["z"], results["y"])

                # Gets the dict from params that defines which plots to make from the results.
                plots = self.params.get("plots", {}).copy()

                # Ensure history is not plotted again.
                plots["History"] = False

                if utils.check_substrings_in_string(
                    self.model_type, "scene classifier", "mlp", "MLP"
                ):
                    plots["Mask"] = False

                # Amends the results' directory to add a new level for test results.
                results_dir = self.exp_fn.parent / "test"

                # Plots the results.
                visutils.plot_results(
                    plots,
                    mode="test",
                    class_names=self.params["classes"],
                    colours=self.params["colours"],
                    save=save,
                    show=show,
                    model_name=self.params["model_name"],
                    timestamp=self.params["timestamp"],
                    results_dir=results_dir,
                    **results,
                )

        # Now experiment is complete, saves model parameters and config file to disk in case error is
        # encountered in plotting of results.
        self.close()

        if self.gpu == 0:
            # Checks whether to run TensorBoard on the log from the experiment. If defined as optional in the config,
            # a user confirmation is required to run TensorBoard with a 60s timeout.
            if self.params.get("run_tensorboard", False) in (
                "opt",
                "optional",
                "OPT",
                "Optional",
            ):
                try:  # pragma: no cover
                    res = inputimeout(
                        prompt="Run TensorBoard Logs? (Y/N): ", timeout=_timeout
                    )
                    if res in ("Y", "y", "yes", "Yes", "YES", "run", "RUN", "Run"):
                        self.run_tensorboard()
                        return
                    elif res in ("N", "n", "no", "No", "NO"):
                        pass
                    else:
                        self.print("\n*Input not recognised*. Please try again")
                except TimeoutOccurred:  # pragma: no cover
                    self.print(
                        "Input timeout elapsed. TensorBoard logs will not be run."
                    )

            # With auto set in the config, TensorBoard will automatically run without asking for user confirmation.
            elif self.params.get("run_tensorboard", False) in (
                True,
                "auto",
                "Auto",
            ):  # pragma: no cover
                self.run_tensorboard()
                return

            # If the user declined, optional or auto wasn't defined in the config or a timeout occurred,
            # the user is informed how to run TensorBoard on the logs using RunTensorBoard.py.
            self.print(
                "\nTensorBoard logs will not be run but still can be by using RunTensorBoard.py and"
            )
            self.print(
                "providing the path to this experiment's results directory and unique experiment ID"
            )

    def tsne_cluster(self, mode: str = "test") -> None:
        """Perform TSNE clustering on the embeddings from the model and visualise.

        Passes a batch from the test dataset through the model in eval mode to get the embeddings.
        Passes these embeddings to :mod:`visutils` to train a TSNE algorithm and then visual the cluster.

        Args:
            mode (str): The mode of model fitting that the embeddings come from.
        """
        task = TSNEVis(
            self.model,
            self.batch_size,
            self.device,
            self.exp_fn,
            self.gpu,
            self.rank,
            self.world_size,
            self.writer,
        )

        task(mode)

    def close(self) -> None:
        """Closes the experiment, saving experiment parameters and model to file."""
        if _tensorflow_exist:
            if (
                isinstance(self.writer, utils.extract_class_type(TENSORBOARD_WRITER))
                and self.writer
            ):
                # Ensure the TensorBoard logger is closed.
                self.writer.close()  # type: ignore[attr-defined]
        if isinstance(self.writer, Run):
            # Ensures all the `wandb` runs finish and sync.
            self.writer.finish()

        if self.gpu == 0:
            self.print("\nSAVING EXPERIMENT CONFIG TO FILE")
            # Outputs the modified YAML parameters config file used for this experiment to file.
            with open(f"{self.exp_fn}.yml", "w") as outfile:
                yaml.dump(self.params, outfile)

            # Writes the recorded training and validation metrics of the experiment to file.
            self.print("\nSAVING METRICS TO FILE")
            try:
                sub_metrics = self.metric_logger.get_sub_metrics()
                metrics_df = pd.DataFrame(
                    {key: sub_metrics[key]["y"] for key in sub_metrics.keys()}
                )
                metrics_df["Epoch"] = sub_metrics["train_loss"]["x"]
                metrics_df.set_index("Epoch", inplace=True, drop=True)
                metrics_df.to_csv(f"{self.exp_fn}_metrics.csv")

            except (ValueError, KeyError) as err:  # pragma: no cover
                self.print(err)
                self.print("\n*ERROR* in saving metrics to file.")

            # Checks whether to save the model parameters to file.
            if self.params.get("save_model", False) in (
                "opt",
                "optional",
                "OPT",
                "Optional",
            ):
                try:  # pragma: no cover
                    res = inputimeout(
                        prompt="\nSave model to file? (Y/N): ", timeout=_timeout
                    )
                    if res in ("Y", "y", "yes", "Yes", "YES", "save", "SAVE", "Save"):
                        # Saves model state dict to PyTorch file.
                        self.save_model_weights()
                        self.print("MODEL PARAMETERS SAVED")
                    elif res in ("N", "n", "no", "No", "NO"):
                        self.print("Model will NOT be saved to file")
                        pass
                    else:
                        self.print("Input not recognised. Please try again")
                except TimeoutOccurred:  # pragma: no cover
                    self.print("Input timeout elapsed. Model will not be saved")

            elif self.params.get("save_model", False) in (True, "auto", "Auto"):
                self.print("\nSAVING MODEL PARAMETERS TO FILE")
                # Saves model state dict to PyTorch file.
                self.save_model_weights()

    def compute_classification_report(
        self, predictions: Sequence[int], labels: Sequence[int]
    ) -> None:
        """Creates and saves to file a classification report table of precision, recall, f-1 score and support.

        Args:
            predictions (~typing.Sequence[int]): List of predicted labels.
            labels (~typing.Sequence[int]): List of corresponding ground truth label masks.
        """
        # Ensures predictions and labels are flattened.
        preds: NDArray[Any, Int] = utils.batch_flatten(predictions)
        targets: NDArray[Any, Int] = utils.batch_flatten(labels)

        # Uses utils to create a classification report in a DataFrame.
        cr_df = utils.make_classification_report(preds, targets, self.params["classes"])

        # Saves classification report DataFrame to a .csv file at fn.
        cr_df.to_csv(f"{self.exp_fn}_classification-report.csv")

    def extract_model_from_distributed(self) -> MinervaModel:
        """Extracts the actual model from any distributed wrapping if this is a distributed run.

        Returns:
            MinervaModel: Unwrapped model.
        """
        model = self.model

        # Checks if this is a distributed run.
        if isinstance(model, MinervaDataParallel):  # type: ignore[attr-defined]  # pragma: no cover
            # Extracts the actual model instance from the distributed wrapping.
            model = model.model.module  # type: ignore[assignment]

        assert isinstance(model, MinervaModel)
        return model

    def save_model_weights(self, fn: Optional[str] = None) -> None:
        """Saves model state dict to :mod:`torch` file.

        Args:
            fn (str): Optional; Filename and path (excluding extension) to save weights to.
        """
        if fn is None:
            fn = str(self.exp_fn)

        model = self.extract_model_from_distributed()

        torch.save(model.state_dict(), f"{fn}.pt")

    def save_model(
        self, fn: Optional[Union[Path, str]] = None, fmt: str = "pt"
    ) -> None:
        """Saves the model object itself to :mod:`torch` file.

        Args:
            fn (~pathlib.Path | str): Optional; Filename and path (excluding extension) to save model to.
            fmt (str): Optional; Format to save model to. ``pt`` for :mod:`torch`, or :mod:`onnx` for ONNX.

        Raises:
            ValueError: If format is not recognised.
        """
        model = self.extract_model_from_distributed()

        if fn is None:
            fn = str(self.exp_fn)

        if fmt == "pt":
            torch.save(model, f"{fn}.pt")
        elif fmt == "onnx":
            x = torch.rand(*self.get_input_size(), device=self.device)
            torch.onnx.export(model, (x,), f"{fn}.onnx")
        else:
            raise ValueError(f"format {fmt} unrecognised!")

    def save_backbone(self) -> None:
        """Readies the model for use in downstream tasks and saves to file."""
        # Checks that model has the required method to ready it for use on downstream tasks.
        assert hasattr(self.model, "get_backbone")
        pre_trained_backbone: Module = self.model.get_backbone()  # type: ignore[operator]

        cache_dir = universal_path(self.params["dir"]["cache"])

        # Saves the pre-trained backbone to the cache.
        cache_fn = cache_dir / self.params["model_name"]

        try:
            os.mkdir(cache_dir)
        except FileExistsError:
            pass

        torch.save(pre_trained_backbone.state_dict(), f"{cache_fn}.pt")

    def run_tensorboard(self) -> None:
        """Opens :mod:`tensorboard` log of the current experiment in a locally hosted webpage."""
        utils.run_tensorboard(  # pragma: no cover
            path=self.params["dir"]["results"].parent,
            env_name="env2",
            exp_name=self.params["exp_name"],
            host_num=6006,
        )

    def print(self, msg: object) -> None:
        """Print function that will only print the object if this is run on the main device.

        Args:
            msg (object): Object or message to print.
        """
        if self.verbose and self.gpu == 0:
            print(msg)
