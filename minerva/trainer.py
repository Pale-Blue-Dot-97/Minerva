# -*- coding: utf-8 -*-
# Copyright (C) 2022 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Module containing the class :class:`Trainer` to handle the fitting of neural networks."""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import os
from contextlib import nullcontext

from nptyping import NDArray, Int
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from alive_progress import alive_bar
from inputimeout import TimeoutOccurred, inputimeout
from simclr.modules import NT_Xent
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from minerva.datasets import make_loaders
from minerva.logger import MinervaLogger
from minerva.metrics import MinervaMetrics
from minerva.models import MinervaDataParallel, MinervaModel
from minerva.pytorchtools import EarlyStopping
from minerva.utils import utils, visutils

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"


# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Default time till timeout waiting for a user input in seconds.
_timeout = 30

__all__ = ["Trainer"]


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class Trainer:
    """Helper class to handle the entire fitting and evaluation of a model.

    Attributes:
        params (dict): Dictionary describing all the parameters that define how the model will be constructed, trained
            and evaluated. These should be defined via config YAML files.
        model: Model to be fitted of a class contained within `minerva.models`.
        max_epochs (int): Number of epochs to train the model for.
        batch_size (int): Size of each batch of samples supplied to the model.
        loaders (dict[DataLoader]): Dictionary containing DataLoaders for each dataset.
        n_batches (dict[int]): Dictionary of the number of batches to supply to the model for train, validation and
            testing.
        metrics (dict): Dictionary to hold the loss and accuracy results from training, validation and testing.
        device: The CUDA device on which to fit the model.

    Args:
        gpu (int, optional): CUDA GPU device number. For use in distributed computing. Defaults to 0.
        verbose (bool): Turns messages to stdout off/on.
        params (Dict[str, Any]): Dictionary describing all the parameters that define how the model will be
            constructed, trained and evaluated. These should be defined via config YAML files.

    Keyword Args:
        results (list[str]): Path to the results' directory to save plots to.
        model_name (str): Name of the model to be used in filenames of results.
        batch_size (int): Size of each batch of samples supplied to the model.
        max_epochs (int): Number of epochs to train the model for.
    """

    def __init__(
        self,
        gpu: int = 0,
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True,
        **params: Dict[str, Any],
    ) -> None:
        # Gets the datasets, number of batches, class distribution and the modfied parameters for the experiment.
        loaders, n_batches, class_dist, new_params = make_loaders(
            rank, world_size, **params
        )

        # Sets the global GPU number for distributed computing. In single process, this will just be 0.
        self.gpu = gpu

        # Verbose level. Always 0 if this is not the primary GPU to avoid duplicate stdout statements.
        self.verbose = verbose if gpu == 0 else False

        if self.gpu == 0:
            # Prints config to stdout.
            print(
                "\n==+ Experiment Parameters +====================================================="
            )
            utils.print_config(new_params)

        self.params = new_params
        self.class_dist = class_dist
        self.loaders = loaders
        self.n_batches = n_batches

        self.modes = params["dataset_params"].keys()

        # Flag for a fine-tuning experiment.
        self.fine_tune = self.params.get("fine_tune", False)

        # Sets the timestamp of the experiment.
        self.params["timestamp"] = utils.timestamp_now(fmt="%d-%m-%Y_%H%M")

        # Sets experiment name and adds this to the path to the results' directory.
        self.params["exp_name"] = "{}_{}".format(
            self.params["model_name"], self.params["timestamp"]
        )
        self.params["dir"]["results"].append(self.params["exp_name"])

        # Path to experiment directory and experiment name.
        self.exp_fn = os.sep.join(
            self.params["dir"]["results"] + [self.params["exp_name"]]
        )

        self.batch_size: int = params["hyperparams"]["params"]["batch_size"]

        # Finds and sets the CUDA device to be used.
        self.device = utils.get_cuda_device(gpu)

        # Creates model (and loss function) from specified parameters in params.
        self.model = self.make_model()

        # Determines the output shape of the model.
        sample_pairs: Union[bool, Any] = params.get("sample_pairs", False)
        if type(sample_pairs) != bool:
            sample_pairs = False
        self.model.determine_output_dim(sample_pairs=sample_pairs)

        # Transfer to GPU
        self.model.to(self.device)

        # Sets up the early stopping functionality.
        self.stopper = None
        self.early_stop = False
        if "stopping" in self.params["hyperparams"]:
            self.stopper = EarlyStopping(
                path=f"{self.exp_fn}.pt", **self.params["hyperparams"]["stopping"]
            )

        # Sets the max number of epochs of fitting.
        self.max_epochs = params["hyperparams"].get("max_epochs", 25)

        # Calculates number of samples in each mode of fitting.
        self.n_samples = {
            mode: self.n_batches[mode] * self.batch_size for mode in self.modes
        }

        # Initialise the metric logger and model IO for the experiment.
        self.make_metric_logger()
        self.modelio_func = self.get_io_func()

        # Stores the step number for that mode of fitting. To be used for TensorBoard logging.
        self.step_num = {mode: 0 for mode in self.modes}

        # Initialise TensorBoard logger
        self.writer = SummaryWriter(os.sep.join(self.params["dir"]["results"]))

        # Creates and sets the optimiser for the model.
        self.make_optimiser()

        if self.gpu == 0:
            # Determines the input size of the model.
            input_size: Tuple[int, ...]
            if self.params["model_type"] in ["MLP", "mlp"]:
                input_size = (self.batch_size, self.model.input_shape)
            else:
                input_size = (self.batch_size, *self.model.input_shape)

            if sample_pairs:
                input_size = (2, *input_size)

            # Print model summary.
            summary(self.model, input_size=input_size)

            # Adds a graphical layout of the model to the TensorBoard logger.
            self.writer.add_graph(
                self.model, input_to_model=torch.rand(*input_size, device=self.device)
            )

        # Checks if multiple GPUs detected. If so, wraps model in DistributedDataParallel for multi-GPU use.
        if torch.cuda.device_count() > 1:
            self.print(f"{torch.cuda.device_count()} GPUs detected")
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = MinervaDataParallel(self.model, DDP, device_ids=[gpu])

        else:
            # Adds a graphical layout of the model to the TensorBoard logger.
            try:
                self.writer.add_graph(
                    self.model,
                    input_to_model=torch.rand(*input_size, device=self.device),
                )
            except RuntimeError as err:
                print(err)
                print("ABORT adding graph to writer")

    def make_model(self) -> MinervaModel:
        """Creates a model from the parameters specified by config.

        Returns:
            MinervaModel: Initialised model.
        """
        model_params = self.params["hyperparams"]["model_params"]

        # Gets the model requested by config parameters.
        _model = utils.func_by_str(
            "minerva.models", self.params["model_name"].split("-")[0]
        )

        if self.fine_tune:
            # Define path to the cached version of the desired pre-trained model.
            weights_path = os.sep.join(
                self.params["dir"]["cache"] + [self.params["pre_train_name"]]
            )

            # Add the path to the pre-trained weights to the model params.
            model_params["backbone_weight_path"] = f"{weights_path}.pt"

        # Initialise model.
        model: MinervaModel = _model(self.make_criterion(), **model_params)
        return model

    def make_criterion(self) -> Any:
        """Creates a PyTorch loss function based on config parameters.

        Returns:
            Any: Initialised PyTorch loss function specified by config parameters.
        """
        # Gets the loss function requested by config parameters.
        loss_params: Dict[str, Any] = self.params["hyperparams"]["loss_params"].copy()
        module = loss_params.pop("module", "torch.nn")
        criterion: Callable[..., Any] = utils.func_by_str(module, loss_params["name"])

        if criterion is NT_Xent:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1

            loss_params["params"]["batch_size"] = int(self.batch_size / world_size)
            loss_params["params"]["world_size"] = world_size

        criterion_params_exist = utils.check_dict_key(loss_params, "params")

        if (
            self.params.get("balance", False)
            and self.params["model_type"] == "segmentation"
        ):
            weights_dict = utils.class_weighting(self.class_dist, normalise=False)

            weights = []
            for i in range(len(weights_dict)):
                weights.append(weights_dict[i])

            if not criterion_params_exist:
                loss_params["params"] = {"weight": torch.Tensor(weights)}
            else:
                loss_params["params"]["weight"] = torch.Tensor(weights)
            return criterion(**loss_params["params"])
        else:
            if not criterion_params_exist:
                return criterion()
            else:
                return criterion(**loss_params["params"])

    def make_optimiser(self) -> None:
        """Creates a PyTorch optimiser based on config parameters and sets optimiser."""

        # Gets the optimiser requested by config parameters.
        optimiser_params: Dict[str, Any] = self.params["hyperparams"][
            "optim_params"
        ].copy()
        module = optimiser_params.pop("module", "torch.optim")
        optimiser = utils.func_by_str(module, optimiser_params["name"])

        # Constructs and sets the optimiser for the model based on supplied config parameters.
        self.model.set_optimiser(
            optimiser(self.model.parameters(), **optimiser_params["params"])
        )

    def make_metric_logger(self) -> None:
        """Creates an object to calculate and log the metrics from the experiment, selected by config parameters."""

        # Gets the size of the input data to the network (without batch dimension).
        data_size = self.params["hyperparams"]["model_params"]["input_size"]

        # Gets constructor of the metric logger from name in the config.
        metric_logger: Callable[..., Any] = utils.func_by_str(
            "minerva.metrics", self.params["metrics"]
        )

        # Initialises the metric logger with arguments.
        self.metric_logger: MinervaMetrics = metric_logger(
            self.n_batches,
            batch_size=self.batch_size,
            data_size=data_size,
            model_type=self.params["model_type"],
        )

    def get_logger(self) -> Callable[..., Any]:
        """Creates an object to log the results from each step of model fitting during an epoch.

        Returns:
            Callable[..., Any]: The constructor of logger to be intialised within the epoch.
        """
        logger: Callable[..., Any] = utils.func_by_str(
            "minerva.logger", self.params["logger"]
        )
        return logger

    def get_io_func(self) -> Callable[..., Any]:
        """Fetches a func to handle IO for the type of model used in the experiment.

        Returns:
            Callable: Model IO function requested from parameters.
        """
        return utils.func_by_str("minerva.modelio", self.params["model_io"])

    def epoch(
        self, mode: str, record_int: bool = False, record_float: bool = False
    ) -> Optional[Dict[str, Any]]:
        """All encompassing function for any type of epoch, be that train, validation or testing.

        Args:
            mode (str): Either train, val or test. Defines the type of epoch to run on the model.
            record_int (bool): Optional; Whether to record the integer results
                (i.e. ground truth and predicted labels).
            record_float (bool): Optional; Whether to record the floating point results i.e. class probabilities.

        Returns:
            Dict[str, Any] | None: If ``record_int=True`` or ``record_float=True``, returns the predicted
            and ground truth labels, and the patch IDs supplied to the model. Else, returns ``None``.
        """
        batch_size = self.batch_size
        if dist.is_available() and dist.is_initialized():
            batch_size = self.batch_size // dist.get_world_size()

        # Calculates the number of samples
        n_samples = self.n_batches[mode] * batch_size

        # Creates object to log the results from each step of this epoch.
        _epoch_logger: Callable[..., Any] = self.get_logger()
        epoch_logger: MinervaLogger = _epoch_logger(
            self.n_batches[mode],
            batch_size,
            n_samples,
            self.model.output_shape,
            self.model.n_classes,
            record_int=record_int,
            record_float=record_float,
        )

        # Initialises a progress bar for the epoch.
        with alive_bar(
            self.n_batches[mode], bar="blocks"
        ) if self.gpu == 0 else nullcontext() as bar:
            # Sets the model up for training or evaluation modes.
            if mode == "train":
                self.model.train()
            else:
                self.model.eval()

            # Core of the epoch.
            for batch in self.loaders[mode]:
                results = self.modelio_func(
                    batch, self.model, self.device, mode, **self.params
                )

                if dist.is_available() and dist.is_initialized():
                    loss = results[0].data.clone()
                    dist.all_reduce(loss.div_(dist.get_world_size()))
                    results = (loss, *results[1:])

                epoch_logger.log(mode, self.step_num[mode], self.writer, *results)

                self.step_num[mode] += 1

                # Updates progress bar that batch has been processed.
                if self.gpu == 0:
                    bar()

        # Updates metrics with epoch results.
        self.metric_logger(mode, epoch_logger.get_logs)

        # If configured to do so, calculates the grad norms.
        if self.params.get("calc_norm", False):
            _ = utils.calc_grad(self.model)

        # Returns the results of the epoch if configured to do so. Else, returns None.
        if record_int or record_float:
            return epoch_logger.get_results
        else:
            return None

    def fit(self) -> None:
        """Fits the model by running ``max_epochs`` number of training and validation epochs."""
        for epoch in range(self.max_epochs):
            self.print(
                f"\nEpoch: {epoch + 1}/{self.max_epochs} =========================================================="
            )

            # Conduct training or validation epoch.
            for mode in ("train", "val"):

                results: Dict[str, Any] = {}

                # If final epoch and configured to plot, runs the epoch with recording of integer results turned on.
                if epoch == (self.max_epochs - 1) and self.params.get(
                    "plot_last_epoch", False
                ):
                    result: Optional[Dict[str, Any]] = self.epoch(mode, record_int=True)
                    assert result is not None
                    results = result

                else:
                    self.epoch(mode)

                # Add epoch number to metrics.
                self.metric_logger.log_epoch_number(mode, epoch)

                print(self.metric_logger.get_metrics)

                # Print epoch results.
                if self.gpu == 0:
                    self.metric_logger.print_epoch_results(mode, epoch)

                # Sends validation loss to the stopper and updates early stop bool.
                if mode == "val" and self.stopper is not None:
                    val_loss = self.metric_logger.get_metrics["val_loss"]["y"][epoch]
                    self.stopper(val_loss, self.model)
                    self.early_stop = self.stopper.early_stop

            # Special case for final train/ val epoch to plot results if configured so.
            if epoch == (self.max_epochs - 1) or self.early_stop:
                if self.early_stop:
                    self.print("\nEarly stopping triggered")

                # Ensures that plots likely to cause memory issues are not attempted.
                plots: Dict[str, bool] = self.params.get("plots", {}).copy()
                plots["CM"] = False
                plots["ROC"] = False

                if not self.params("plot_last_epoch", False):
                    # If not plotting results, ensure that only history plotting will remain
                    # if originally set to do so.
                    plots["Mask"] = False
                    plots["Pred"] = False

                # Create a subset of metrics which drops the testing results for plotting model history.
                sub_metrics = self.metric_logger.get_sub_metrics()

                # Ensures masks are not plotted for model types that do not yield such outputs.
                if self.params["model_type"] in ("scene classifier", "mlp", "MLP"):
                    plots["Mask"] = False

                # Amends the results' directory to add a new level for train or validation.
                results_dir: List[str] = self.params["dir"]["results"].copy()
                results_dir.append(mode)

                if self.gpu == 0:
                    # Plots the results of this epoch.
                    visutils.plot_results(
                        plots,
                        metrics=sub_metrics,
                        class_names=self.params["classes"],
                        colours=self.params["colours"],
                        save=True,
                        show=False,
                        model_name=self.params["model_name"],
                        timestamp=self.params["timestamp"],
                        results_dir=results_dir,
                        **results,
                    )

                # If early stopping has been triggered, loads the last model save to replace current model,
                # ready for testing.
                if self.early_stop:
                    if self.gpu == 0:
                        self.model.load_state_dict(torch.load(f"{self.exp_fn}.pt"))
                    break

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

        # Runs test epoch on model, returning the predicted labels, ground truth labels supplied
        # and the IDs of the samples supplied.
        results: Optional[Dict[str, Any]] = self.epoch(
            "test", record_int=True, record_float=True
        )
        assert results is not None

        # Prints test loss and accuracy to stdout.
        self.metric_logger.print_epoch_results("test", 0)

        # Add epoch number to testing results.
        self.metric_logger.log_epoch_number("test", 0)

        # Now experiment is complete, saves model parameters and config file to disk in case error is
        # encountered in plotting of results.
        self.close()

        if self.gpu == 0:
            if "z" in results and "y" in results:
                self.print("\nMAKING CLASSIFICATION REPORT")
                self.compute_classification_report(results["z"], results["y"])

            # Gets the dict from params that defines which plots to make from the results.
            plots = self.params["plots"]

            # Ensure history is not plotted again.
            plots["History"] = False

            if self.params["model_type"] in ("scene classifier", "mlp", "MLP"):
                plots["Mask"] = False

            # Amends the results' directory to add a new level for test results.
            results_dir: List[str] = self.params["dir"]["results"]
            results_dir.append("test")

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

            # Checks whether to run TensorBoard on the log from the experiment. If defined as optional in the config,
            # a user confirmation is required to run TensorBoard with a 60s timeout.
            if self.params.get("run_tensorboard", False) in (
                "opt",
                "optional",
                "OPT",
                "Optional",
            ):
                try:
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
                except TimeoutOccurred:
                    self.print(
                        "Input timeout elapsed. TensorBoard logs will not be run."
                    )

            # With auto set in the config, TensorBoard will automatically run without asking for user confirmation.
            elif self.params.get("run_tensorboard", False) in (True, "auto", "Auto"):
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

    def close(self) -> None:
        """Closes the experiment, saving experiment parameters and model to file."""
        # Ensure the TensorBoard logger is closed.
        self.writer.close()

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

            except (ValueError, KeyError) as err:
                self.print(err)
                self.print("\n*ERROR* in saving metrics to file.")

            # Checks whether to save the model parameters to file.
            if self.params.get("save_model", False) in (
                "opt",
                "optional",
                "OPT",
                "Optional",
            ):
                try:
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
                except TimeoutOccurred:
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
            predictions (ArrayLike): List of predicted labels.
            labels (ArrayLike): List of corresponding ground truth label masks.
        """
        # Ensures predictions and labels are flattened.
        preds: NDArray[Any, Int] = utils.batch_flatten(predictions)
        targets: NDArray[Any, Int] = utils.batch_flatten(labels)

        # Uses utils to create a classification report in a DataFrame.
        cr_df = utils.make_classification_report(preds, targets, self.params["classes"])

        # Saves classification report DataFrame to a .csv file at fn.
        cr_df.to_csv(f"{self.exp_fn}_classification-report.csv")

    def save_model_weights(self, fn: Optional[str] = None) -> None:
        """Saves model state dict to PyTorch file.

        Args:
            fn (str): Optional; Filename and path (excluding extension) to save weights to.
        """
        if fn is None:
            fn = self.exp_fn
        torch.save(self.model.state_dict(), f"{fn}.pt")

    def save_model(self, fn: Optional[str] = None) -> None:
        """Saves the model object itself to PyTorch file.

        Args:
            fn (str): Optional; Filename and path (excluding extension) to save model to.
        """
        if fn is None:
            fn = self.exp_fn
        torch.save(self.model, f"{fn}.pt")

    def save_backbone(self) -> None:
        """Readies the model for use in downstream tasks and saves to file."""
        # Checks that model has the required method to ready it for use on downstream tasks.
        assert hasattr(self.model, "get_backbone")
        pre_trained_backbone: Module = self.model.get_backbone()

        # Saves the pre-trained backbone to the cache.
        cache_fn = os.sep.join(
            self.params["dir"]["cache"] + [self.params["model_name"]]
        )
        torch.save(pre_trained_backbone.state_dict(), f"{cache_fn}.pt")

    def run_tensorboard(self) -> None:
        """Opens TensorBoard log of the current experiment in a locally hosted webpage."""
        utils.run_tensorboard(
            path=self.params["dir"]["results"][:-1],
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
