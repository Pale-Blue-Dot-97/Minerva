# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

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
"""Core functionality of :mod:`tasks`, defining the abstract :class:`MinervaTask` class

.. versionadded:: 0.27
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter
else:  # pragma: no cover
    SummaryWriter = None

import pandas as pd
import torch
import torch.distributed as dist
from nptyping import Int, NDArray
from omegaconf import OmegaConf
from torch import Tensor
from torch._dynamo.eval_frame import OptimizedModule
from wandb.sdk.wandb_run import Run

from minerva.datasets import make_loaders
from minerva.logger.tasklog import MinervaTaskLogger
from minerva.models import (
    MinervaDataParallel,
    MinervaModel,
    extract_wrapped_model,
    wrap_model,
)
from minerva.utils import utils, visutils
from minerva.utils.utils import fallback_params, func_by_str


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaTask(ABC):
    """An abstract definition of a task to fit or evalulate a model within :mod:`minerva`.

    Attributes:
        name (str): The name of the task.
        params (dict[str, ~typing.Any]): Dictionary describing all the parameters that define how the model will be
            constructed, trained and evaluated. These should be defined via config ``YAML`` files.
        model (MinervaModel): Model to be fitted of a class contained within :mod:`~minerva.models`.
        loaders (dict[str, ~torch.utils.data.DataLoader]): :class:`dict` containing
            :class:`~torch.utils.data.DataLoader` (s) for each dataset.
        n_batches (dict[str, int]): Dictionary of the number of batches to supply to the model for train,
            validation and testing.
        batch_size (int): Number of samples in each batch.
        device: The CUDA device on which to fit the model.
        verbose (bool): Provides more prints to stdout if ``True``.
        class_dist (~typing.Any): Distribution of classes within the data.
        sample_pairs (bool): Whether samples are paired together for Siamese learning.
        modes (tuple[str, ...]): The different *modes* of fitting in this experiment specified by the config.
        writer (~torch.utils.tensorboard.writer.SummaryWriter | ~wandb.sdk.wandb_run.Run | None): The *writer*
            to perform logging for this experiment. For use with either :mod:`tensorboard` or :mod:`wandb`.
        n_samples (dict[str, int]): Number of samples in each mode of model fitting.
        logger (~logger.tasklog.MinervaTaskLogger): Object to calculate and log metrics to track the performance
            of the model.
        modelio_func (~typing.Callable[..., ~typing.Any]): Function to handle the input/ output to the model.
        step_num (int): Holds the step number for this task.
        model_type (str): Type of the model that determines how to handle IO, metric calculations etc.
        record_int (bool): Store the integer results of each epoch in memory such the predictions, ground truth etc.
        record_float (bool): Store the floating point results of each epoch in memory
            such as the raw predicted probabilities.
        train (bool): Mark this as a training task. Will activate backward passes.
        task_dir (~pathlib.Path): Path to the sub-directory to hold the results of this task.
        task_fn (~pathlib.Path): Path and filename prefix for this task.

    Args:
        name (str): The name of the task. Should match the key for the task
            in the ``tasks`` dictionary of the experiment params.
        model (MinervaModel): Model to be fitted of a class contained within :mod:`~minerva.models`.
        device: The CUDA device on which to fit the model.
        exp_fn (~pathlib.Path): The path to the parent directory for the results of the experiment.
        gpu (int): Optional; CUDA GPU device number. For use in distributed computing. Defaults to ``0``.
        rank (int): Optional; The rank of this process across all devices in the distributed run.
        world_size (int): Optional; The total number of processes across the distributed run.
        writer (~wandb.sdk.wandb_run.Run | RunDisabled): Optional; Run object for Weights and Biases.
        record_int (bool): Store the integer results of each epoch in memory such the predictions, ground truth etc.
        record_float (bool): Store the floating point results of each epoch in memory
            such as the raw predicted probabilities.
        train (bool): Mark this as a training task. Will activate backward passes.

    Keyword Args:
        elim (bool): Will eliminate classes that have no samples in and reorder the class labels so they
            still run from ``0`` to ``n-1`` classes where ``n`` is the reduced number of classes.
            :mod:`minerva` ensures that labels are converted between the old and new schemes seamlessly.
        balance (bool): Activates class balancing. For ``model_type="scene classifer"`` or ``model_type="mlp"``,
            over and under sampling will be used. For ``model_type="segmentation"``, class weighting will be
            used on the loss function.
        model_type (str): Defines the type of the model. If ``siamese``, ensures inappropiate functionality is not used.
        dataset_params (dict[str, ~typing.Any]): Parameters to construct each dataset.
            See documentation on structure of these.
        collator (dict[str, ~typing.Any]): Defines the collator to use that will collate samples together into batches.
            Contains the ``module`` key to define the import path and the ``name`` key
            for name of the collation function.
        sample_pairs (bool): Activates paired sampling for Siamese models. Only used for ``train`` datasets.
        loss_func (str): Name of the loss function to use.
        optim_func (str): Name of the optimiser function to use.
        lr (float): Learning rate of optimiser.
        optim_params (dict[str, ~typing.Any]): :class:`dict` to hold any additional parameters for the optimiser,
            other than the already handled learning rate -- ``lr``. Place them in the ``params`` key.
            If using a non-torch optimiser, use the ``module`` key to specify the import path to the optimiser function.
        loss_params (dict[str, ~typing.Any]): :class:`dict` to hold any additional parameters for the loss function
            in the ``params`` key. If using a non-torch loss function, you need to specify the import path
            with the ``module`` key.
        patch_size (tuple[float, float]): Defines the shape of the patches in the dataset.
        input_size (tuple[int, ...]): Shape of the input to the model. Typically in CxHxW format.
            Should align with the values given for ``patch_size``.
        tasklogger (str): Specify the task logger to use. Must be the name of a
            :class:`~logger.tasklog.MinervaTaskLogger` class within :mod:`~logger.tasklog`.
        steplogger (str): Specify the step logger to use. Must be the name of a
            :class:`~logger.steplog.MinervaStepLogger` class within :mod:`~logger.steplog`.
        modelio (str): Specify the IO function to use to handle IO for the model during fitting. Must be the name
            of a function within :mod:`modelio`.
        target_key (str): Optional; Name of the target key (if supervised task). Either ``mask`` or ``label``.

    .. versionadded:: 0.27
    """

    logger_cls: str = "SupervisedTaskLogger"
    model_io_name: str = "sup_tg"

    def __init__(
        self,
        name: str,
        model: Union[MinervaModel, MinervaDataParallel, OptimizedModule],
        device: torch.device,
        exp_fn: Path,
        gpu: int = 0,
        rank: int = 0,
        world_size: int = 1,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        record_int: bool = True,
        record_float: bool = False,
        train: bool = False,
        **global_params,
    ) -> None:
        self.name = name

        self.model = model

        # Gets the datasets, number of batches, class distribution and the modfied parameters for the experiment.
        loaders, n_batches, class_dist, new_params = make_loaders(
            rank, world_size, task_name=name, **global_params
        )

        # If there are multiple modes and therefore number of batches, just take the value of the first one.
        if isinstance(n_batches, dict):
            n_batches = next(iter(n_batches.values()))

        global_params["tasks"][name] = new_params

        # ``global_params`` is the whole experiment parameters.
        self.global_params = global_params

        # ``params`` is the parameters for just this task.
        self.params = new_params

        # Modify `exp_fn` with a sub-directory for this task.
        self.task_dir = exp_fn / self.name
        self.task_fn = self.task_dir / exp_fn.name

        self.train = self.params.get("train", train)
        if "train" in self.params:
            del self.params["train"]

        self.gpu = gpu

        self.loaders = loaders
        self.class_dist = class_dist

        # Try to find parameters first in the task params then fall back to the global level params.
        self.batch_size = fallback_params("batch_size", self.params, self.global_params)

        # Corrects the batch size if this is a distributed job to account for batches being split across devices.
        if dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]  # pragma: no cover
            self.batch_size = self.batch_size // dist.get_world_size()  # type: ignore[attr-defined]

        self.n_batches = n_batches
        self.model_type = fallback_params("model_type", self.params, self.global_params)
        self.sample_pairs = fallback_params(
            "sample_pairs", self.params, self.global_params
        )
        self.n_classes = fallback_params("n_classes", self.params, self.global_params)

        self.output_size = model.output_shape

        self.record_int = utils.fallback_params(
            "record_int", self.params, self.global_params, record_int
        )
        self.record_float = utils.fallback_params(
            "record_float", self.params, self.global_params, record_float
        )

        self.elim = fallback_params("elim", self.params, self.global_params, False)
        self.balance = fallback_params(
            "balance", self.params, self.global_params, False
        )

        # Ensure the model IO function is treated as static not a class method.
        self.modelio = staticmethod(self.get_io_func()).__func__

        self.device = device
        self.writer = writer
        self.step_num = 0

        # Initialise the Weights and Biases metrics for this task.
        self.init_wandb_metrics()

        # Update the loss function of the model.
        self.model.set_criterion(self.make_criterion())

        # Loss function needs to be on the same device as the rest of the model and the data.
        self.model.to(self.device)

        # To eliminate classes, we're going to have to do a fair bit of rebuilding of the model...
        if self.elim:
            # Update the stored number of classes within the model and
            # then rebuild the classification layers that are dependent on the number of classes.
            self.model.update_n_classes(self.n_classes)

            # The optimser is dependent on the model parameters so shall have to be rebuilt.
            self.make_optimiser()

            # The new parts of the model will need transfering to the GPU.
            self.model.to(self.device)

            # And finally the model will need re-wrapping for distributed computing and/ or torch compilation.
            self.model = wrap_model(
                extract_wrapped_model(self.model),
                gpu,
                torch_compile=fallback_params(
                    "torch_compile", self.params, self.global_params, False
                ),
            )

        # Make the logger for this task.
        self.logger: MinervaTaskLogger = self.make_logger()

    def init_wandb_metrics(self) -> None:
        """Setups up the step counter for :mod:`wandb` logging."""
        if isinstance(self.writer, Run):
            self.writer.define_metric(f"{self.name}/step")
            self.writer.define_metric(f"{self.name}/*", step_metric=f"{self.name}/step")

    def make_criterion(self) -> Any:
        """Creates a :mod:`torch` loss function based on config parameters.

        Returns:
            ~typing.Any: Initialised :mod:`torch` loss function specified by config parameters.
        """
        # Gets the loss function requested by config parameters.
        loss_params: Dict[str, Any] = deepcopy(
            fallback_params("loss_params", self.params, self.global_params)
        )
        if OmegaConf.is_config(loss_params):
            loss_params = OmegaConf.to_object(loss_params)  # type: ignore[assignment]

        module = loss_params.pop("module", "torch.nn")
        criterion: Callable[..., Any] = func_by_str(module, loss_params["name"])

        if not utils.check_dict_key(loss_params, "params"):
            loss_params["params"] = {}

        if self.balance and utils.check_substrings_in_string(
            self.model_type, "segmentation"
        ):
            weights_dict = utils.class_weighting(self.class_dist, normalise=False)

            weights = []
            if self.elim:
                for i in range(len(weights_dict)):
                    weights.append(weights_dict[i])
            else:
                for i in range(self.n_classes):
                    weights.append(weights_dict.get(i, 0.0))

            loss_params["params"]["weight"] = Tensor(weights)

            return criterion(**loss_params["params"])

        else:
            return criterion(**loss_params["params"])

    def make_optimiser(self) -> None:
        """Creates a :mod:`torch` optimiser based on config parameters and sets optimiser."""

        # Gets the optimiser requested by config parameters.
        optimiser_params: Dict[str, Any] = deepcopy(self.params["optim_params"])

        if OmegaConf.is_config(optimiser_params):
            optimiser_params = OmegaConf.to_object(optimiser_params)  # type: ignore[assignment]

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

    def make_logger(self) -> MinervaTaskLogger:
        """Creates an object to calculate and log the metrics from the experiment, selected by config parameters.

        Returns:
            ~logger.tasklog.MinervaTaskLogger: Constructed task logger.
        """

        # Gets constructor of the metric logger from name in the config.
        _logger_cls = func_by_str(
            "minerva.logger.tasklog",
            utils.fallback_params(
                "task_logger", self.params, self.global_params, self.logger_cls
            ),
        )

        # Initialises the metric logger with arguments.
        logger: MinervaTaskLogger = _logger_cls(
            self.name,
            self.n_batches,
            self.batch_size,
            self.output_size,
            step_logger_params=utils.fallback_params(
                "step_logger", self.params, self.global_params, {}
            ),
            record_int=self.record_int,
            record_float=self.record_float,
            writer=self.writer,
            model_type=self.model_type,
            sample_pairs=self.sample_pairs,
            n_classes=self.n_classes,
        )

        return logger

    def get_io_func(self) -> Callable[..., Any]:
        """Fetches a func to handle IO for the type of model used in the experiment.

        Returns:
            ~typing.Callable[..., ~typing.Any]: Model IO function requested from parameters.
        """
        io_func: Callable[..., Any] = func_by_str(
            "minerva.modelio",
            utils.fallback_params(
                "model_io", self.params, self.global_params, self.model_io_name
            ),
        )
        return io_func

    @abc.abstractmethod
    def step(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def _generic_step(self, epoch_no: int) -> Optional[Dict[str, Any]]:
        self.step()

        # Send the logs to the metric logger.
        self.logger.calc_metrics(epoch_no)

        if self.record_int or self.record_float:
            results = self.logger.get_results
        else:
            results = None

        self.logger.refresh_step_logger()

        return results

    def __call__(self, epoch_no: int) -> Any:
        return self._generic_step(epoch_no)

    @property
    def get_logs(self) -> Dict[str, Any]:
        return self.logger.get_logs

    @property
    def get_metrics(self) -> Dict[str, Any]:
        return self.logger.get_metrics

    def log_null(self, epoch_no: int) -> None:
        """Log :attr:`numpy.NAN` for this epoch.

        Useful for logging null when a validation epoch was skipped so that
        the length of the logs remains the same as the training logs.

        Args:
            epoch_no (int): Epoch number.
        """
        self.logger.log_null()
        self.logger.log_epoch_number(epoch_no)

    def print_epoch_results(self, epoch_no: int) -> None:
        self.logger.print_epoch_results(epoch_no)

    def plot(
        self,
        results: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        save: bool = True,
        show: bool = False,
    ) -> None:
        # Gets the dict from params that defines which plots to make from the results.
        plots = utils.fallback_params(
            "plots", self.params, self.global_params, {}
        ).copy()

        if not utils.fallback_params(
            "plot_last_epoch", self.params, self.global_params, False
        ):
            # If not plotting results, ensure that only history plotting will remain
            # if originally set to do so.
            plots["Mask"] = False
            plots["Pred"] = False

        # Ensures masks are not plotted for model types that do not yield such outputs.
        if utils.check_substrings_in_string(
            self.model_type, "scene classifier", "mlp", "MLP"
        ):
            plots["Mask"] = False

        if metrics is None:
            plots["History"] = False

        elif len(list(metrics.values())[0]["x"]) <= 1:
            plots["History"] = False

        else:
            pass

        visutils.plot_results(
            plots,
            task_name=self.name,
            metrics=metrics,
            class_names=utils.fallback_params(
                "classes", self.params, self.global_params
            ),
            colours=utils.fallback_params("colours", self.params, self.global_params),
            save=save,
            show=show,
            model_name=self.params["model_name"],
            timestamp=self.params["timestamp"],
            results_dir=self.task_dir,
            cfg=self.params,
            **results,
        )

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

        # Ensure the parent directories for the classification report exist.
        self.task_fn.parent.mkdir(parents=True, exist_ok=True)

        # Saves classification report DataFrame to a .csv file at fn.
        cr_df.to_csv(f"{self.task_fn}_classification-report.csv")

    def save_metrics(self) -> None:
        print("\nSAVING METRICS TO FILE")
        try:
            metrics = self.get_metrics
            metrics_df = pd.DataFrame(
                {key: metrics[key]["y"] for key in metrics.keys()}
            )

            # Assumes that the length of each metric is the same.
            metrics_df["Epoch"] = list(metrics.values())[0]["x"]
            metrics_df.set_index("Epoch", inplace=True, drop=True)
            metrics_df.to_csv(f"{self.task_fn}_metrics.csv")

        except (ValueError, KeyError) as err:  # pragma: no cover
            print(err)
            print("\n*ERROR* in saving metrics to file.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}-{self.name}"


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_task(task: str, *args, **params) -> MinervaTask:
    """Get the requested :class:`MinervaTask` by name.

    Args:
        task (str): Name of the task.
        params (dict[str, ~typing.Any]): Parameters for the task to be initialised.

    Returns:
        MinervaTask: Constructed :class:`MinervaTask` object.
    """
    _task = func_by_str("minerva.tasks", task)

    task = _task(*args, **params)
    assert isinstance(task, MinervaTask)
    return task
