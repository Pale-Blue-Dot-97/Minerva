"""Module containing the class Trainer to handle the fitting of neural networks.

    Copyright (C) 2021 Harry James Baker

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

Created under a project funded by the Ordnance Survey Ltd

TODO:

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
import yaml
from Minerva.utils import visutils, utils
import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from itertools import islice
from alive_progress import alive_bar


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

    def __init__(self, loaders, n_batches: dict, class_dist: dict = None, **params):
        """Initialises the Trainer.

        Args:
            loaders (dict[DataLoader]): Dictionary containing DataLoaders for each dataset.
            n_batches (dict): Dictionary of the number of batches to supply to the model for train, validation and
                testing.
        Keyword Args:
            results (list[str]): Path to the results directory to save plots to.
            model_name (str): Name of the model to be used in filenames of results.
            batch_size (int): Size of each batch of samples supplied to the model.
            max_epochs (int): Number of epochs to train the model for.
        """
        self.params = params
        self.class_dist = class_dist

        # Sets the timestamp of the experiment.
        self.params['timestamp'] = utils.timestamp_now(fmt='%d-%m-%Y_%H%M')

        # Sets experiment name and adds this to the path to the results directory.
        self.params['exp_name'] = '{}_{}'.format(self.params['model_name'], self.params['timestamp'])
        self.params['dir']['results'].append(self.params['exp_name'])

        self.batch_size = params['hyperparams']['params']['batch_size']

        # Creates model (and loss function) from specified parameters in params.
        self.model = self.make_model()

        self.max_epochs = params['hyperparams']['max_epochs']
        self.loaders = loaders
        self.n_batches = n_batches
        self.data_size = params['hyperparams']['model_params']['input_size']

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],

            'train_acc': [],
            'val_acc': [],
            'test_acc': []
        }

        # Initialise TensorBoard logger
        self.writer = SummaryWriter(os.path.join(*self.params['dir']['results']))

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

    def make_model(self):
        """Creates a model from the parameters specified by config.

        Returns:
            Initialised model.
        """
        model_params = self.params['hyperparams']['model_params']

        if self.params['model_type'] == 'segmentation':
            model_params['batch_size'] = self.batch_size

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

    def make_optimiser(self):
        """Creates a PyTorch optimiser based on config parameters and sets optimiser."""

        # Gets the optimiser requested by config parameters.
        optimiser = utils.func_by_str('torch.optim', self.params['hyperparams']['optim_name'])

        # Constructs and sets the optimiser for the model based on supplied config parameters.
        self.model.set_optimiser(optimiser(self.model.parameters(), **self.params['hyperparams']['optim_params']))

    def epoch(self, mode):
        """All encompassing function for any type of epoch, be that train, validation or testing.

        Args:
            mode (str): Either train, val or test. Defines the type of epoch to run on the model.

        Returns:
            If a test epoch, returns the predicted and ground truth labels and the patch IDs supplied to the model.
        """
        # Initialises variables to hold overall epoch results.
        total_loss = 0.0
        total_correct = 0.0
        test_labels = []
        test_predictions = []
        test_ids = []

        # Initialises a progress bar for the epoch.
        with alive_bar(self.n_batches[mode], bar='blocks') as bar:
            # Sets the model up for training or evaluation modes
            if mode is 'train':
                self.model.train()
            else:
                self.model.eval()

            # Core of the epoch.
            for x_batch, y_batch, patch_id in islice(self.loaders[mode], self.n_batches[mode]):

                # Transfer to GPU.
                x, y = x_batch.to(self.device), y_batch.to(self.device)

                # Runs a training epoch.
                if mode is 'train':
                    loss, z = self.model.training_step(x, y)

                # Runs a validation epoch.
                elif mode is 'val':
                    loss, z = self.model.validation_step(x, y)

                # Runs a testing epoch.
                elif mode is 'test':
                    loss, z = self.model.testing_step(x, y)

                    test_predictions.append(torch.argmax(z, 1).cpu().numpy())
                    test_labels.append(y.cpu().numpy())
                    test_ids.append(patch_id)

                ls = loss.item()
                correct = (torch.argmax(z, 1) == y).sum().item()

                total_loss += ls
                total_correct += correct

                self.writer.add_scalar('{}_loss'.format(mode), ls)
                self.writer.add_scalar('{}_acc'.format(mode), correct / len(torch.flatten(y_batch)))

                # Updates progress bar that sample has been processed.
                bar()

        # Updates metrics with epoch results.
        self.metrics['{}_loss'.format(mode)].append(total_loss / self.n_batches[mode])
        if self.params['model_type'] == 'segmentation':
            self.metrics['{}_acc'.format(mode)].append(total_correct / (self.n_batches[mode] * self.batch_size *
                                                                        self.data_size[1] * self.data_size[2]))
        else:
            self.metrics['{}_acc'.format(mode)].append(total_correct / (self.n_batches[mode] * self.batch_size))

        total_norm = utils.calc_grad(self.model)

        if mode is 'test':
            return test_predictions, test_labels, test_ids
        else:
            return

    def fit(self):
        """Fits the model by running max_epochs number of training and validation epochs."""
        for epoch in range(self.max_epochs):
            print(f'Epoch: {epoch + 1}/{self.max_epochs}')

            self.epoch('train')
            self.epoch('val')

            print('Training loss: {} | Validation Loss: {}'.format(self.metrics['train_loss'][epoch],
                                                                   self.metrics['val_loss'][epoch]))
            print('Train Accuracy: {}% | Validation Accuracy: {}% \n'.format(self.metrics['train_acc'][epoch] * 100.0,
                                                                             self.metrics['val_acc'][epoch] * 100.0))

    def test(self, plots, save=True):
        """Tests the model by running a testing epoch then taking the results and orchestrating the plotting and
        analysis of them.

        Args:
            plots (dict): Dictionary defining which plots of the test results to create.
            save (bool): Optional; Determines whether or not to save the plots created to file.

        Returns:
            Test predicted and ground truth labels along with the patch IDs supplied to the model during testing.
        """
        print('\r\nTESTING')
        predictions, labels, ids = self.epoch('test')

        z = []

        try:
            z = np.array(predictions).flatten()

        except ValueError:
            for i in range(len(predictions)):
                for j in range(len(predictions[i])):
                    z.append(predictions[i][j])

        y = []
        try:
            y = np.array(labels).flatten()

        except ValueError:
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    y.append(labels[i][j])

        print('Test Loss: {} | Test Accuracy: {}% \n'.format(self.metrics['test_loss'][0],
                                                             self.metrics['test_acc'][0] * 100.0))

        sub_metrics = {k: self.metrics[k] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}

        # Plots the results.
        visutils.plot_results(sub_metrics, plots, z, y, self.params['classes'], self.params['colours'],
                              save=save, show=False, model_name=self.params['model_name'],
                              timestamp=self.params['timestamp'], results_dir=self.params['dir']['results'])

        return predictions, labels, ids

    def close(self):
        """Closes the experiment, saving experiment parameters and model to file."""
        # Ensure the TensorBoard logger is closed.
        self.writer.close()

        # Path to experiment directory and experiment name.
        fn = os.path.join(*self.params['dir']['results'], self.params['exp_name'])

        # Outputs the modified YAML parameters config file used for this experiment to file.
        with open('{}.yml'.format(fn), 'w') as outfile:
            yaml.dump(self.params, outfile)

        # Saves model state dict to PyTorch file.
        torch.save(self.model.state_dict(), '{}.pt'.format(fn))

    def run_tensorboard(self):
        """Opens TensorBoard log of the current experiment."""
        os.chdir(os.path.join(*self.params['dir']['results'][:-1]))
        os.system('conda activate env2')
        os.system('tensorboard --logdir={}'.format(self.params['exp_name']))
