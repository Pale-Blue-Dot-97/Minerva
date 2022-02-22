"""Module containing the class Trainer to handle the fitting of neural networks.

    Copyright (C) 2022 Harry James Baker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program in LICENSE.txt. If not,
    see <https://www.gnu.org/licenses/>.

Author: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd.

Attributes:
    _timeout (int): Default time till timeout waiting for a user input in seconds.

TODO:
    * Add ability to plot the final training and validation results as well as testing
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Optional, Tuple, List, Dict, Iterable
try:
    from numpy.typing import ArrayLike
except ModuleNotFoundError or ImportError:
    ArrayLike = Iterable
import os
import yaml
from Minerva.utils import visutils, utils
import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from inputimeout import inputimeout, TimeoutOccurred

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Default time till timeout waiting for a user input in seconds.
_timeout = 30


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class Trainer:
    """Helper class to handle the entire fitting and evaluation of a model.

    Attributes:
        params (dict): Dictionary describing all the parameters that define how the model will be constructed, trained
            and evaluated. These should be defined via config YAML files.
        model: Model to be fitted of a class contained within Minerva.models.
        max_epochs (int): Number of epochs to train the model for.
        batch_size (int): Size of each batch of samples supplied to the model.
        loaders (dict[DataLoader]): Dictionary containing DataLoaders for each dataset.
        n_batches (dict[int]): Dictionary of the number of batches to supply to the model for train, validation and
            testing.
        metrics (dict): Dictionary to hold the loss and accuracy results from training, validation and testing.
        device: The CUDA device on which to fit the model.
    """

    def __init__(self, loaders: Dict[str, DataLoader], n_batches: Dict[str, int], class_dist: Optional[List[Tuple[int, int]]] = None,
                 **params) -> None:
        """Initialises the Trainer.

        Args:
            loaders (dict[DataLoader]): Dictionary containing DataLoaders for each dataset.
            n_batches (dict): Dictionary of the number of batches to supply to the model for train, validation and
                testing.
        Keyword Args:
            results (list[str]): Path to the results' directory to save plots to.
            model_name (str): Name of the model to be used in filenames of results.
            batch_size (int): Size of each batch of samples supplied to the model.
            max_epochs (int): Number of epochs to train the model for.
        """
        self.params = params
        self.class_dist = class_dist

        # Sets the timestamp of the experiment.
        self.params['timestamp'] = utils.timestamp_now(fmt='%d-%m-%Y_%H%M')

        # Sets experiment name and adds this to the path to the results' directory.
        self.params['exp_name'] = '{}_{}'.format(self.params['model_name'], self.params['timestamp'])
        self.params['dir']['results'].append(self.params['exp_name'])

        self.batch_size = params['hyperparams']['params']['batch_size']

        self.max_pixel_value = params['max_pixel_value']

        # Creates model (and loss function) from specified parameters in params.
        self.model = self.make_model()

        self.model.determine_output_dim()

        # Checks if multiple GPUs detected. If so, wraps model in DataParallel for multi-GPU use.
        if torch.cuda.device_count() > 1:
            print(f'{torch.cuda.device_count()} GPUs detected')
            self.model = torch.nn.DataParallel(self.model)

        self.max_epochs = params['hyperparams']['max_epochs']
        self.loaders = loaders
        self.n_batches = n_batches
        self.data_size = params['hyperparams']['model_params']['input_size']

        # Stores the step number for that mode of fitting. To be used for TensorBoard logging.
        self.step_num = {
            'train': 0,
            'val': 0,
            'test': 0
        }

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics = {
            'train_loss': {'x': [], 'y': []},
            'val_loss': {'x': [], 'y': []},
            'test_loss': {'x': [], 'y': []},

            'train_acc': {'x': [], 'y': []},
            'val_acc': {'x': [], 'y': []},
            'test_acc': {'x': [], 'y': []}
        }

        # Initialise TensorBoard logger
        self.writer = SummaryWriter(os.sep.join(self.params['dir']['results']))

        # Creates and sets the optimiser for the model.
        self.make_optimiser()

        self.device = utils.get_cuda_device()

        # Transfer to GPU
        self.model.to(self.device)

        if self.params['model_type'] in ['MLP', 'mlp']:
            input_size = (self.batch_size, self.model.input_size)
        else:
            input_size = (self.batch_size, *self.model.input_shape)

        # Print model summary
        summary(self.model, input_size=input_size)

        # Adds a graphical layout of the model to the TensorBoard logger.
        self.writer.add_graph(self.model, input_to_model=torch.rand(*input_size, device=self.device))

    def make_model(self) -> torch.nn.Module:
        """Creates a model from the parameters specified by config.

        Returns:
            Initialised model.
        """
        model_params = self.params['hyperparams']['model_params']

        # Gets the model requested by config parameters.
        model = utils.func_by_str('Minerva.models', self.params['model_name'].split('-')[0])

        # Initialise model
        return model(self.make_criterion(), **model_params)

    def make_criterion(self):
        """Creates a PyTorch loss function based on config parameters.

        Returns:
            Initialised PyTorch loss function specified by config parameters.
        """
        # Gets the loss function requested by config parameters.
        criterion = utils.func_by_str('torch.nn', self.params['hyperparams']['loss_name'])

        if self.params['balance'] and self.params['model_type'] == 'segmentation':
            weights_dict = utils.class_weighting(self.class_dist, normalise=False)

            weights = []
            for i in range(len(weights_dict)):
                weights.append(weights_dict[i])
            return criterion(weight=torch.Tensor(weights))
        else:
            return criterion()

    def make_optimiser(self) -> None:
        """Creates a PyTorch optimiser based on config parameters and sets optimiser."""

        # Gets the optimiser requested by config parameters.
        optimiser = utils.func_by_str('torch.optim', self.params['hyperparams']['optim_name'])

        # Constructs and sets the optimiser for the model based on supplied config parameters.
        self.model.set_optimiser(optimiser(self.model.parameters(), **self.params['hyperparams']['optim_params']))

    def epoch(self, mode: str, record_int: bool = False,
              record_float: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, List[str],
                                                            np.ndarray, np.ndarray]]:
        """All encompassing function for any type of epoch, be that train, validation or testing.

        Args:
            mode (str): Either train, val or test. Defines the type of epoch to run on the model.
            record_int (bool): Optional; Whether to record the integer results
                (i.e. ground truth and predicted labels).
            record_float (bool): Optional; Whether to record the floating point results i.e. class probabilities.

        Returns:
            If a test epoch, returns the predicted and ground truth labels and the patch IDs supplied to the model.
        """
        # Initialises variables to hold overall epoch results.
        total_loss = 0.0
        total_correct = 0.0

        labels = None
        predictions = None
        probs = None
        ids = []
        bounds = None

        n_samples = self.n_batches[mode] * self.batch_size

        if record_int:
            labels = np.empty((self.n_batches[mode], self.batch_size, *self.model.output_shape), dtype=np.uint8)
            predictions = np.empty((self.n_batches[mode], self.batch_size, *self.model.output_shape), dtype=np.uint8)

        if record_float:
            try:
                probs = np.empty((self.n_batches[mode], self.batch_size, self.model.n_classes,
                                  *self.model.output_shape), dtype=np.float16)
            except MemoryError:
                print('Dataset too large to record probabilities of predicted classes!')

            try:
                bounds = np.empty((self.n_batches[mode], self.batch_size), dtype=object)
            except MemoryError:
                print('Dataset too large to record bounding boxes of samples!')

        # Initialises a progress bar for the epoch.
        with alive_bar(self.n_batches[mode], bar='blocks') as bar:
            # Sets the model up for training or evaluation modes
            if mode == 'train':
                self.model.train()
            else:
                self.model.eval()

            batch_num = 0

            # Core of the epoch.
            for sample in self.loaders[mode]:
                x_batch = sample['image'] / self.max_pixel_value
                y_batch = sample['mask']

                x_batch = x_batch.to(torch.float)
                y_batch = np.squeeze(y_batch, axis=1)
                y_batch = y_batch.type(torch.long)

                # Transfer to GPU.
                x, y = x_batch.to(self.device), y_batch.to(self.device)

                # Runs a training epoch.
                if mode == 'train':
                    loss, z = self.model.training_step(x, y)

                # Runs a validation epoch.
                elif mode == 'val':
                    loss, z = self.model.validation_step(x, y)

                # Runs a testing epoch.
                elif mode == 'test':
                    loss, z = self.model.testing_step(x, y)

                if record_int:
                    # Arg max the estimated probabilities and add to predictions.
                    predictions[batch_num] = torch.argmax(z, 1).cpu().numpy()

                    # Add the labels and sample IDs to lists.
                    labels[batch_num] = y.cpu().numpy()
                    batch_ids = []
                    for i in range(batch_num * self.batch_size, (batch_num + 1) * self.batch_size):
                        batch_ids.append(str(i).zfill(len(str(n_samples))))
                    ids.append(batch_ids)

                if record_float:
                    # Add the estimated probabilities to probs.
                    probs[batch_num] = z.detach().cpu().numpy()
                    bounds[batch_num] = sample['bbox']

                self.step_num[mode] += 1

                ls = loss.item()
                correct = (torch.argmax(z, 1) == y).sum().item()

                total_loss += ls
                total_correct += correct

                self.writer.add_scalar(tag='{}_loss'.format(mode), scalar_value=ls,
                                       global_step=self.step_num[mode])
                self.writer.add_scalar(tag='{}_acc'.format(mode),
                                       scalar_value=correct / len(torch.flatten(y_batch)),
                                       global_step=self.step_num[mode])

                batch_num += 1

                # Updates progress bar that sample has been processed.
                bar()

        # Updates metrics with epoch results.
        self.metrics['{}_loss'.format(mode)]['y'].append(total_loss / self.n_batches[mode])

        if self.params['model_type'] == 'segmentation':
            self.metrics['{}_acc'.format(mode)]['y'].append(total_correct / (self.n_batches[mode] * self.batch_size *
                                                                             self.data_size[1] * self.data_size[2]))
        else:
            self.metrics['{}_acc'.format(mode)]['y'].append(total_correct / (self.n_batches[mode] * self.batch_size))

        if self.params['calc_norm']:
            _ = utils.calc_grad(self.model)

        if record_int:
            return predictions, labels, ids, probs, bounds
        else:
            return

    def fit(self) -> None:
        """Fits the model by running max_epochs number of training and validation epochs."""
        for epoch in range(self.max_epochs):
            print(f'\nEpoch: {epoch + 1}/{self.max_epochs} ==========================================================')

            # Conduct training or validation epoch.
            for mode in ('train', 'val'):

                # Special case for final train/ val epoch to plot results if configured so.
                if epoch == (self.max_epochs - 1) and self.params['plot_last_epoch']:
                    predictions, labels, ids, _, _ = self.epoch(mode, record_int=True)

                    # Ensures that the model history will not be plotted.
                    # That should be done with the plotting of test results.
                    plots = self.params['plots'].copy()
                    plots['History'] = False
                    plots['CM'] = False
                    plots['ROC'] = False

                    # Ensures that inappropriate plots are not attempted for incompatible outputs.
                    if self.params['model_type'] in ('scene classifier', 'segmentation'):
                        plots['PvT'] = False

                    if self.params['model_type'] in ('scene classifier', 'mlp', 'MLP'):
                        plots['Mask'] = False

                    # Amends the results' directory to add a new level for train or validation.
                    results_dir = self.params['dir']['results'].copy()
                    results_dir.append(mode)

                    # Plots the results of this epoch.
                    visutils.plot_results(plots, predictions, labels, ids=ids, class_names=self.params['classes'],
                                          colours=self.params['colours'], save=True, show=False,
                                          model_name=self.params['model_name'], timestamp=self.params['timestamp'],
                                          results_dir=results_dir)

                else:
                    self.epoch(mode)

                # Add epoch number to training metrics.
                self.metrics['{}_loss'.format(mode)]['x'].append(epoch + 1)
                self.metrics['{}_acc'.format(mode)]['x'].append(epoch + 1)

                # Print training epoch results.
                print('{} | Loss: {} | Accuracy: {}% \n'.format(mode, self.metrics['{}_loss'.format(mode)]['y'][epoch],
                                                                self.metrics['{}_acc'.format(mode)]['y'][epoch]
                                                                * 100.0))

    def test(self, save: bool = True, show: bool = False) -> None:
        """Tests the model by running a testing epoch then taking the results and orchestrating the plotting and
        analysis of them.

        Args:
            save (bool): Optional; Determines whether to save the plots created to file.
            show (bool): Optional; Determines whether to show the plots created.

        Notes:
            save = True, show = False regardless of input for plots made for each sample such as PvT or Mask plots.

        Returns:
            None
        """
        print('\r\nTESTING')

        # Runs test epoch on model, returning the predicted labels, ground truth labels supplied
        # and the IDs of the samples supplied.
        predictions, labels, test_ids, probabilities, bounds = self.epoch('test', record_int=True, record_float=True)

        # Prints test loss and accuracy to stdout.
        print('Test | Loss: {} | Accuracy: {}% \n'.format(self.metrics['test_loss']['y'][0],
                                                          self.metrics['test_acc']['y'][0] * 100.0))

        # Add epoch number to testing results.
        self.metrics['{}_loss'.format('test')]['x'].append(1)
        self.metrics['{}_acc'.format('test')]['x'].append(1)

        # Now experiment is complete, saves model parameters and config file to disk in case error is
        # encountered in plotting of results.
        self.close()

        print('\nMAKING CLASSIFICATION REPORT')
        self.compute_classification_report(predictions, labels)

        # Create a subset of metrics which drops the testing results for plotting model history.
        sub_metrics = {k: self.metrics[k] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}

        # Gets the dict from params that defines which plots to make from the results.
        plots = self.params['plots']

        # Ensures that inappropriate plots are not attempted for incompatible outputs.
        if self.params['model_type'] in ('scene classifier', 'segmentation'):
            plots['PvT'] = False

        if self.params['model_type'] in ('scene classifier', 'mlp', 'MLP'):
            plots['Mask'] = False

        # Amends the results' directory to add a new level for test results.
        results_dir = self.params['dir']['results']
        results_dir.append('test')

        # Plots the results.
        visutils.plot_results(plots, predictions, labels, metrics=sub_metrics, ids=test_ids, mode='test',
                              bounds=bounds, probs=probabilities, class_names=self.params['classes'],
                              colours=self.params['colours'], save=save, show=show,
                              model_name=self.params['model_name'], timestamp=self.params['timestamp'],
                              results_dir=results_dir)

        # Checks whether to run TensorBoard on the log from the experiment.
        # If defined as optional in the config, a user confirmation is required to run TensorBoard with a 60s timeout.
        if self.params['run_tensorboard'] in ('opt', 'optional', 'OPT', 'Optional'):
            try:
                res = inputimeout(prompt='Run TensorBoard Logs? (Y/N): ', timeout=_timeout)
                if res in ('Y', 'y', 'yes', 'Yes', 'YES', 'run', 'RUN', 'Run'):
                    self.run_tensorboard()
                    return
                elif res in ('N', 'n', 'no', 'No', 'NO'):
                    pass
                else:
                    print('\n*Input not recognised*. Please try again')
            except TimeoutOccurred:
                print('Input timeout elapsed. TensorBoard logs will not be run.')

        # With auto set in the config, TensorBoard will automatically run without asking for user confirmation.
        elif self.params['run_tensorboard'] in (True, 'auto', 'Auto'):
            self.run_tensorboard()
            return

        # If the user declined, optional or auto wasn't defined in the config or a timeout occurred,
        # the user is informed how to run TensorBoard on the logs using RunTensorBoard.py.
        print('\nTensorBoard logs will not be run but still can be by using RunTensorBoard.py and')
        print('providing the path to this experiment\'s results directory and unique experiment ID')

    def close(self) -> None:
        """Closes the experiment, saving experiment parameters and model to file."""
        # Ensure the TensorBoard logger is closed.
        self.writer.close()

        # Path to experiment directory and experiment name.
        fn = os.sep.join(self.params['dir']['results'] + [self.params['exp_name']])

        print('\nSAVING EXPERIMENT CONFIG TO FILE')
        # Outputs the modified YAML parameters config file used for this experiment to file.
        with open(f'{fn}.yml', 'w') as outfile:
            yaml.dump(self.params, outfile)

        # Writes the recorded training and validation metrics of the experiment to file.
        print('\nSAVING METRICS TO FILE')
        try:
            sub_metrics = {k: self.metrics[k]['y'] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}
            metrics_df = pd.DataFrame(sub_metrics)
            metrics_df['Epoch'] = self.metrics['train_loss']['x']
            metrics_df.set_index('Epoch', inplace=True, drop=True)
            metrics_df.to_csv(f'{fn}_metrics.csv')

        except (ValueError, KeyError):
            print('\n*ERROR* in saving metrics to file.')

        # Checks whether to save the model parameters to file.
        if self.params['save_model'] in ('opt', 'optional', 'OPT', 'Optional'):
            try:
                res = inputimeout(prompt='\nSave model to file? (Y/N): ', timeout=_timeout)
                if res in ('Y', 'y', 'yes', 'Yes', 'YES', 'save', 'SAVE', 'Save'):
                    # Saves model state dict to PyTorch file.
                    torch.save(self.model.state_dict(), f'{fn}.pt')
                    print('MODEL PARAMETERS SAVED')
                elif res in ('N', 'n', 'no', 'No', 'NO'):
                    print('Model will NOT be saved to file')
                    pass
                else:
                    print('Input not recognised. Please try again')
            except TimeoutOccurred:
                print('Input timeout elapsed. Model will not be saved')

        elif self.params['save_model'] in (True, 'auto', 'Auto'):
            print('\nSAVING MODEL PARAMETERS TO FILE')
            # Saves model state dict to PyTorch file.
            torch.save(self.model.state_dict(), f'{fn}.pt')

    def compute_classification_report(self, predictions: ArrayLike, labels: ArrayLike) -> None:
        """Creates and saves to file a classification report table of precision, recall, f-1 score and support.

        Args:
            predictions (list or np.ndarray): List of predicted labels.
            labels (list or np.ndarray): List of corresponding ground truth label masks.

        Returns:
            None
        """
        # Ensures predictions and labels are flattened.
        predictions = utils.batch_flatten(predictions)
        labels = utils.batch_flatten(labels)

        # Uses utils to create a classification report in a DataFrame.
        cr_df = utils.make_classification_report(predictions, labels, self.params['classes'])

        # Defines the filename and path for the classification report to be saved to.
        fn = os.sep.join(self.params['dir']['results'] + [self.params['exp_name']])

        # Saves classification report DataFrame to a .csv file at fn.
        cr_df.to_csv(f'{fn}_classification-report.csv')

    def run_tensorboard(self) -> None:
        """Opens TensorBoard log of the current experiment in a locally hosted webpage."""
        utils.run_tensorboard(path=self.params['dir']['results'][:-1],
                              env_name='env2',
                              exp_name=self.params['exp_name'],
                              host_num=6006)
